"""Discovering a benchmark that lives in another repository.

No such benchmark is installed here, so every test drives `importlib.metadata.entry_points`
through a fake - which is also the point: the validator must behave correctly on a host where a
benchmark is absent, because that is CI, a laptop, and the dashboard's contract check.
"""

from __future__ import annotations

import pytest

from icilval import simulators
from icilval.simulators import MissingBenchmark, Simulator


def _sim(name: str, **kw) -> Simulator:
    base = dict(
        make_policy=lambda *a, **k: None,
        run_units=lambda *a, **k: None,
        build_stage=lambda *a, **k: None,
        make_unit=lambda *a, **k: None,
        demo_frames=lambda demo: [],
        validate_skill=lambda skill, doc: [],
    )
    base.update(kw)
    return Simulator(name=name, **base)


class _Dist:
    def __init__(self, name: str):
        self.name = name


class _EntryPoint:
    def __init__(self, name: str, value: str, on_load, distribution: str = "fake-benchmark-icil"):
        self.name = name
        self.value = value
        self.dist = _Dist(distribution)
        self._on_load = on_load

    def load(self):
        return self._on_load()


@pytest.fixture
def entry_points(monkeypatch):
    """Install a fake entry point group, and undo whatever it registered afterwards."""
    before = dict(simulators.REGISTRY)
    holder: list[_EntryPoint] = []

    def fake(*, group: str):
        assert group == simulators.ENTRY_POINT_GROUP
        return list(holder)

    monkeypatch.setattr("importlib.metadata.entry_points", fake)
    monkeypatch.setattr(simulators, "_LOADED", False)
    yield holder
    simulators.REGISTRY.clear()
    simulators.REGISTRY.update(before)
    simulators._LOADED = True


def test_a_plugin_that_fails_to_import_names_itself(entry_points):
    def boom():
        raise ImportError("no module named sapien")

    entry_points.append(_EntryPoint("brokensim", "brokenpkg.plugin", boom))
    with pytest.raises(MissingBenchmark, match="brokensim.*did not import.*sapien"):
        simulators.load_plugins()


def test_discovery_runs_once_and_is_not_repeated(entry_points):
    calls: list[int] = []

    def once():
        calls.append(1)
        return _Plugin("countsim")

    entry_points.append(_EntryPoint("countsim", "countpkg.plugin", once))
    simulators.names()
    simulators.names()
    simulators.get("countsim")
    assert calls == [1]


# ------------------------------------------------------------------ require and audit


class _Spec:
    """Just enough of `Spec` for the registry: the skills and what each one runs on."""

    def __init__(self, mapping: dict[str, str], benchmarks: dict | None = None):
        self._mapping = mapping
        self.raw = {"benchmarks": benchmarks} if benchmarks is not None else {}

    @property
    def all_skills(self):
        return tuple(self._mapping)

    def simulator(self, skill: str) -> str:
        return self._mapping[skill]


def test_require_passes_when_every_benchmark_is_installed(entry_points):
    """No benchmark ships here, so one is registered for the test rather than assumed present."""
    entry_points.append(_EntryPoint("fakesim", "fakepkg.plugin", lambda: _Plugin()))
    simulators.load_plugins()
    simulators.require(_Spec({"rt_stacking": "fakesim"}))


def test_require_names_the_distribution_to_install():
    """`nosuchsim` rather than a real name: whether a benchmark happens to be pip-installed in
    this environment must not decide whether the test passes."""
    spec = _Spec(
        {"rt_stacking": "nosuchsim"},
        benchmarks={"nosuchsim": {"distribution": "some-benchmark-icil"}},
    )
    with pytest.raises(MissingBenchmark) as exc:
        simulators.require(spec)
    message = str(exc.value)
    assert "some-benchmark-icil" in message
    assert "rt_stacking" in message
    assert simulators.ENTRY_POINT_GROUP in message


def test_require_says_so_even_when_the_spec_declares_nothing():
    with pytest.raises(MissingBenchmark, match="an unknown distribution"):
        simulators.require(_Spec({"rt_stacking": "nosuchsim"}))


def test_audit_reports_every_simulator_a_skill_names(entry_points):
    entry_points.append(_EntryPoint("fakesim", "fakepkg.plugin", lambda: _Plugin()))
    simulators.load_plugins()
    spec = _Spec(
        {"rt_sm_stacking": "fakesim", "rt_stacking": "nosuchsim"},
        benchmarks={
            "fakesim": {"distribution": "fake-benchmark-icil"},
            "nosuchsim": {"distribution": "some-benchmark-icil"},
        },
    )
    rows = {r["simulator"]: r for r in simulators.audit(spec)}
    assert rows["fakesim"]["installed"]
    assert rows["fakesim"]["skills"] == ["rt_sm_stacking"]
    # A registered plugin whose *distribution* cannot be found is still reported: the audit
    # checks both, because a benchmark imported from somewhere unpinned is the thing the
    # distribution pin exists to refuse.
    assert rows["fakesim"]["problems"] == ["distribution fake-benchmark-icil not found"]
    assert not rows["nosuchsim"]["installed"]
    assert "not installed" in rows["nosuchsim"]["problems"]
    assert rows["nosuchsim"]["declared"]


def test_audit_never_raises_so_an_operator_sees_everything_at_once():
    spec = _Spec({"a": "nosuch", "b": "alsomissing"})
    rows = simulators.audit(spec)
    assert {r["simulator"] for r in rows} == {"nosuch", "alsomissing"}
    assert all(r["problems"] for r in rows)


# ------------------------------------------------------------------ adapting a plugin


class _Plugin:
    """A benchmark as a plugin actually arrives: an object, not something that registered itself."""

    api_version = 1

    def __init__(self, name: str = "fakesim"):
        self.id = name

    def info(self):
        return {
            "id": self.id,
            "api_version": self.api_version,
            "demo_channels": {"video": ["frames_"], "actions": ["actions"]},
        }

    def catalogue(self):
        return {}

    def derive_units(self, *, seed_material, count, suite):
        return []

    def verify_prompt(self, *, path, unit):
        return {"ok": True, "sha256": "0" * 64, "problems": []}

    def read_result(self, *, out_dir):
        return {"success": True, "void": False, "steps": 1, "error": None}

    def materialize_command(self, *, unit, out_dir):
        return ["fake", "materialize"]

    def run_command(self, *, unit, prompt, out_dir, policy_address, **kw):
        return ["fake", "run-unit"]


def test_a_plugin_is_adapted_rather_than_asked_to_register(entry_points):
    """A plugin may not import `icilval`, so it cannot call `register` itself. It exposes an
    object and the orchestrator wraps it - which is the only arrangement that keeps the
    dependency one-directional."""
    plugin = _Plugin()
    entry_points.append(_EntryPoint("fakesim", "fakepkg.plugin", lambda: plugin))
    assert simulators.load_plugins() == ("fakesim",)
    sim = simulators.get("fakesim")
    assert sim.benchmark is plugin
    assert sim.distribution == "fake-benchmark-icil"


def test_a_module_exposing_BENCHMARK_is_accepted(entry_points):
    module = type("M", (), {"BENCHMARK": _Plugin()})
    entry_points.append(_EntryPoint("fakesim", "fakepkg.plugin", lambda: module))
    simulators.load_plugins()
    assert simulators.get("fakesim").benchmark is module.BENCHMARK


def test_the_channel_map_travels_so_a_view_can_be_applied(entry_points):
    """The orchestrator redacts by channel and only the benchmark knows what its arrays mean."""
    entry_points.append(_EntryPoint("fakesim", "fakepkg.plugin", lambda: _Plugin()))
    simulators.load_plugins()
    assert simulators.get("fakesim").demo_channels == {
        "video": ("frames_",),
        "actions": ("actions",),
    }


def test_an_entry_point_exposing_no_benchmark_is_refused(entry_points):
    entry_points.append(_EntryPoint("fakesim", "fakepkg.plugin", lambda: type("M", (), {})))
    with pytest.raises(MissingBenchmark, match="exposes no benchmark"):
        simulators.load_plugins()


def test_a_plugin_that_does_not_satisfy_the_abi_is_refused(entry_points):
    broken = _Plugin()
    broken.run_command = None
    entry_points.append(_EntryPoint("fakesim", "fakepkg.plugin", lambda: broken))
    with pytest.raises(MissingBenchmark, match="not a usable benchmark"):
        simulators.load_plugins()


def test_a_plugin_calling_itself_something_else_is_refused(entry_points):
    """The entry point's name is what a skill's `simulator` carries; a mismatch would score a
    field on a benchmark nobody named."""
    other = _Plugin("elsewhere")
    entry_points.append(_EntryPoint("fakesim", "fakepkg.plugin", lambda: other))
    with pytest.raises(MissingBenchmark, match="calls itself 'elsewhere'"):
        simulators.load_plugins()


def test_the_hooks_that_cannot_work_in_this_process_refuse_rather_than_pretend(entry_points):
    """A hook that quietly did nothing would score a field on no episodes at all.

    `run_units` is deliberately not in this list: it is implemented for a plugged benchmark, by
    driving `run_command` and `read_result` in a subprocess. The rest genuinely cannot happen
    here - the orchestrator owns the weights and the architecture template, so a policy is
    *served* rather than loaded into the benchmark's process, and a pool is built from a
    benchmark's own sources on its own side.
    """
    entry_points.append(_EntryPoint("fakesim", "fakepkg.plugin", lambda: _Plugin()))
    simulators.load_plugins()
    sim = simulators.get("fakesim")
    for hook in (sim.make_policy, sim.build_stage, sim.make_unit):
        with pytest.raises(MissingBenchmark, match="out of process"):
            hook()


def test_a_plugged_benchmark_can_actually_run_units(entry_points):
    """The capability this repository lacked: an orchestration layer that can only run the
    benchmarks it ships is not one."""
    entry_points.append(_EntryPoint("fakesim", "fakepkg.plugin", lambda: _Plugin()))
    simulators.load_plugins()
    sim = simulators.get("fakesim")
    assert callable(sim.run_units)
    # It is the subprocess runner, not a refusal.
    with pytest.raises(TypeError):
        sim.run_units()
