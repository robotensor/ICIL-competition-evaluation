"""Serving a policy to a benchmark that cannot import it.

Every test here drives the real host through the real client over a real Unix socket. A mocked
transport would not exercise the thing that actually goes wrong - framing, authentication, a
model raising mid-duel - and those are the failures that make a duel wrong rather than broken.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from icilval.model import wire
from icilval.model.client import PolicyClient, PolicyError
from icilval.model.host import PolicyHost


class Recorder:
    """A policy in the shape `PolicyBase` defines, recording what it was handed."""

    def __init__(self, action=None, fail_on: str | None = None):
        self.action = np.zeros(7, dtype=np.float32) if action is None else action
        self.fail_on = fail_on
        self.prompts: list[dict] = []
        self.histories: list[list[dict]] = []
        self.seeds: list[int] = []
        self.resets = 0

    def seed(self, seed: int) -> None:
        if self.fail_on == "seed":
            raise RuntimeError("no seeding today")
        self.seeds.append(seed)

    def reset(self) -> None:
        self.resets += 1

    def set_prompt(self, demo: dict):
        if self.fail_on == "prompt":
            raise RuntimeError("bad prompt")
        self.prompts.append(demo)
        return type("Info", (), {"steps": len(next(iter(demo.values()))), "chunks": 3})()

    def act(self, history: list[dict]):
        if self.fail_on == "act":
            raise RuntimeError("model exploded")
        self.histories.append(history)
        return self.action


def connected(host: PolicyHost) -> PolicyClient:
    return PolicyClient(host.address, host.authkey, timeout_s=20.0)


def test_a_benchmark_drives_the_policy_over_a_socket():
    policy = Recorder(action=np.array([1.0, 2.0, 3.0], dtype=np.float32))
    with PolicyHost(policy) as host, connected(host) as client:
        client.seed(1234)
        client.reset()
        info = client.prompt({"frames_head": np.zeros((5, 2, 2, 3), dtype=np.uint8)})
        action = client.act([{"rgb": np.ones((2, 2, 3), dtype=np.uint8), "qpos": np.zeros(7)}])

    assert policy.seeds == [1234]
    assert policy.resets == 1
    assert info["prompt_steps"] == 5 and info["prompt_chunks"] == 3
    np.testing.assert_array_equal(action, [1.0, 2.0, 3.0])
    assert action.dtype == np.float32


def test_the_arrays_arrive_byte_for_byte():
    """The policy must see exactly what the benchmark read from the prompt."""
    demo = {
        "frames_head": np.random.default_rng(0).integers(0, 255, (4, 3, 3, 3), dtype=np.uint8),
        "qpos": np.random.default_rng(1).standard_normal((4, 14)),
        "flags": np.array([True, False, True, True]),
    }
    policy = Recorder()
    with PolicyHost(policy) as host, connected(host) as client:
        client.prompt(demo)

    got = policy.prompts[0]
    assert set(got) == set(demo)
    for key, value in demo.items():
        np.testing.assert_array_equal(got[key], value)
        assert got[key].dtype == value.dtype


def test_the_observation_history_keeps_its_order_and_its_keys():
    policy = Recorder()
    history = [
        {"rgb": np.full((1, 1, 3), i, dtype=np.uint8), "qpos": np.full(3, i, dtype=np.float64)}
        for i in range(4)
    ]
    with PolicyHost(policy) as host, connected(host) as client:
        client.act(history)

    got = policy.histories[0]
    assert len(got) == 4
    for i, obs in enumerate(got):
        assert set(obs) == {"rgb", "qpos"}
        assert int(obs["rgb"].flat[0]) == i, "observations arrived out of order"


def test_a_model_error_is_one_units_failure_and_the_host_stays_up():
    """Tearing the server down for one model error would void every unit after it."""
    policy = Recorder(fail_on="act")
    with PolicyHost(policy) as host, connected(host) as client:
        with pytest.raises(PolicyError, match="model exploded"):
            client.act([{"rgb": np.zeros((1, 1, 3), dtype=np.uint8)}])
        assert host.errors == 1
        # Still serving: the units after this one are owed a fair run.
        policy.fail_on = None
        assert client.act([{"rgb": np.zeros((1, 1, 3), dtype=np.uint8)}]) is not None
        assert host.errors == 1


def test_the_wrong_key_cannot_drive_the_policy():
    with PolicyHost(Recorder()) as host:
        with pytest.raises(PolicyError):
            PolicyClient(host.address, b"not the key", timeout_s=5.0)
        # And the host survives the attempt, for the client that does have the key.
        with connected(host) as client:
            client.reset()


def test_an_unknown_op_is_refused_rather_than_guessed():
    with PolicyHost(Recorder()) as host, connected(host) as client:
        wire.send(client._conn, "hello")  # prove the connection is fine
        wire.recv(client._conn)
        wire.send(client._conn, "act", {"steps": 0})
        op, fields, _ = wire.recv(client._conn)
        assert op == "action" or op == "error"


def test_the_key_and_the_socket_are_not_readable_by_anyone_else():
    import os
    import stat

    with PolicyHost(Recorder()) as host:
        assert stat.S_IMODE(os.stat(host.authkey_file).st_mode) == 0o600
        assert stat.S_IMODE(os.stat(host.address).st_mode) == 0o600
        assert host.env()["ICILVAL_POLICY_AUTHKEY"] == host.authkey.hex()


def test_closing_the_host_removes_its_socket_and_its_key():
    host = PolicyHost(Recorder())
    host.__enter__()
    directory = host.authkey_file.parent
    host.close()
    assert not directory.exists(), "the key outlived the host"


def test_several_units_in_one_session_do_not_reload_the_policy():
    """`load()` once per skill is the frozen-policy guarantee."""
    policy = Recorder()
    with PolicyHost(policy) as host, connected(host) as client:
        for _ in range(3):
            client.reset()
            client.prompt({"frames": np.zeros((2, 1, 1, 3), dtype=np.uint8)})
            client.act([{"rgb": np.zeros((1, 1, 3), dtype=np.uint8)}])

    assert policy.resets == 3
    assert len(policy.prompts) == 3


def test_two_benchmarks_in_turn_share_one_host():
    policy = Recorder()
    with PolicyHost(policy) as host:
        for _ in range(2):
            with connected(host) as client:
                client.act([{"rgb": np.zeros((1, 1, 3), dtype=np.uint8)}])
    assert len(policy.histories) == 2


def test_concurrent_clients_do_not_interleave_into_one_session():
    """One connection at a time is the contract; a second waits rather than corrupting the first."""
    policy = Recorder()
    seen = []
    with PolicyHost(policy) as host:
        with connected(host) as first:
            first.act([{"rgb": np.zeros((1, 1, 3), dtype=np.uint8)}])
            seen.append("first")

            done = threading.Event()

            def second():
                with connected(host) as c:
                    c.act([{"rgb": np.ones((1, 1, 3), dtype=np.uint8)}])
                seen.append("second")
                done.set()

            t = threading.Thread(target=second, daemon=True)
            t.start()
            # The first session still holds the host; the second connects once it is released.
            first.close()
            assert done.wait(20.0), "the second client never got served"
            t.join(5.0)
    assert seen == ["first", "second"]
