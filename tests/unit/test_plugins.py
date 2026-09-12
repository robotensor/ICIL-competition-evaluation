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


class _EntryPoint:
    def __init__(self, name: str, value: str, on_load):
        self.name = name
        self.value = value
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


def test_a_plugin_that_registers_itself_is_found(entry_points):
    entry_points.append(
        _EntryPoint(
            "fakesim",
            "fakepkg.plugin",
            lambda: simulators.register(_sim("fakesim", distribution="fake-benchmark-icil")),
        )
    )
    assert simulators.load_plugins() == ("fakesim",)
    assert "fakesim" in simulators.names()
    assert simulators.get("fakesim").distribution == "fake-benchmark-icil"


def test_a_plugin_that_registers_nothing_is_refused(entry_points):
    entry_points.append(_EntryPoint("quiet", "quietpkg.plugin", lambda: None))
    with pytest.raises(MissingBenchmark, match="advertises the simulator 'quiet'"):
        simulators.load_plugins()


def test_a_plugin_that_registers_another_name_is_refused(entry_points):
    entry_points.append(
        _EntryPoint("declared", "wrongpkg.plugin", lambda: simulators.register(_sim("actual")))
    )
    with pytest.raises(MissingBenchmark, match="advertises the simulator 'declared'"):
        simulators.load_plugins()


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
        return simulators.register(_sim("countsim"))

    entry_points.append(_EntryPoint("countsim", "countpkg.plugin", once))
    simulators.names()
    simulators.names()
    simulators.get("libero")
    assert calls == [1]


# ------------------------------------------------------------------ require and audit


class _Spec:
    """Just enough of `Spec` for the registry: the skills and what each one runs on."""

    def __init__(self, mapping: dict[str, str], benchmarks: dict | None = None):
        self._mapping = mapping
        self.raw = {"benchmarks": benchmarks} if benchmarks is not None else {}

    @property
    def skills(self):
        return tuple(self._mapping)

    def simulator(self, skill: str) -> str:
        return self._mapping[skill]


def test_require_passes_when_every_benchmark_is_installed():
    simulators.require(_Spec({"pick_and_place": "libero", "draw_anything": "draw"}))


def test_require_names_the_distribution_to_install():
    spec = _Spec(
        {"rt_stacking": "robotwin"},
        benchmarks={"robotwin": {"distribution": "robotwin-icil-competition"}},
    )
    with pytest.raises(MissingBenchmark) as exc:
        simulators.require(spec)
    message = str(exc.value)
    assert "robotwin-icil-competition" in message
    assert "rt_stacking" in message
    assert simulators.ENTRY_POINT_GROUP in message


def test_require_says_so_even_when_the_spec_declares_nothing():
    with pytest.raises(MissingBenchmark, match="an unknown distribution"):
        simulators.require(_Spec({"rt_stacking": "robotwin"}))


def test_audit_reports_every_simulator_a_skill_names():
    spec = _Spec(
        {"pick_and_place": "libero", "rt_stacking": "robotwin"},
        benchmarks={"robotwin": {"distribution": "robotwin-icil-competition"}},
    )
    rows = {r["simulator"]: r for r in simulators.audit(spec)}
    assert rows["libero"]["installed"] and not rows["libero"]["problems"]
    assert rows["libero"]["skills"] == ["pick_and_place"]
    assert not rows["robotwin"]["installed"]
    assert "not installed" in rows["robotwin"]["problems"]
    assert rows["robotwin"]["declared"]


def test_audit_never_raises_so_an_operator_sees_everything_at_once():
    spec = _Spec({"a": "nosuch", "b": "alsomissing"})
    rows = simulators.audit(spec)
    assert {r["simulator"] for r in rows} == {"nosuch", "alsomissing"}
    assert all(r["problems"] for r in rows)
