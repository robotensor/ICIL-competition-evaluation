"""Which policy a skill gets, and what decides it.

The registry is keyed by `skills.<skill>.architecture`, not by `skills.<skill>.simulator`. That
is the whole content of these tests: a skill whose benchmark lives in another repository has no
policy class there to be asked for, and the orchestrator is the side that holds the weights.
"""

from __future__ import annotations

import pytest

from icilval import arch, simulators
from icilval.model import architectures


def test_every_shipped_architecture_has_a_policy_and_a_template(spec):
    """A template with no policy is a skill that cannot be duelled; a policy with no template is
    weights that cannot be checked. Both are caught here rather than at the start of a run."""
    root = spec.path.parent
    for name in architectures.names():
        assert arch.exists(root / "arch", name), f"{name} has no template in arch/"


def test_the_robotwin_conversion_is_registered_for_its_own_architecture():
    from icilval.model.bpp_robotwin import ARCHITECTURE

    assert architectures.implemented(ARCHITECTURE)
    assert architectures.get(ARCHITECTURE) is not None


def test_a_policy_is_looked_up_by_architecture_and_not_by_simulator(spec, monkeypatch):
    """Two skills on different simulators but the same architecture get the same policy, and one
    skill's simulator changing does not change which policy it gets."""
    seen: list[tuple[str, str]] = []
    monkeypatch.setitem(
        architectures.REGISTRY,
        "fake_arch_v1",
        lambda model_dir, arch_dir, spec_, skill, device="cuda": seen.append((skill, device)),
    )

    class _Spec:
        @staticmethod
        def architecture(skill: str) -> str:
            return "fake_arch_v1"

    architectures.make_policy("models", "arch", _Spec(), "any_skill", device="cpu")
    assert seen == [("any_skill", "cpu")]


def test_an_unimplemented_architecture_names_the_ones_there_are():
    with pytest.raises(KeyError) as exc:
        architectures.get("no_such_arch_v9")
    message = str(exc.value)
    assert "no_such_arch_v9" in message and "bpp_robotwin_v1" in message


def test_registering_an_architecture_twice_is_refused(monkeypatch):
    monkeypatch.setitem(architectures.REGISTRY, "twice_v1", lambda *a, **k: None)
    with pytest.raises(ValueError, match="registered twice"):
        architectures.register("twice_v1", lambda *a, **k: None)


def test_a_simulator_no_longer_carries_a_policy():
    """It could not, for the benchmarks this layer exists to run: a plugged benchmark's policy is
    served to it over a socket, from a process the orchestrator owns."""
    assert not hasattr(simulators.get("libero"), "make_policy")
    assert not hasattr(simulators, "make_policy")
