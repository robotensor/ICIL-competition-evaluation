"""Deriving a duel's units from a benchmark in another repository.

Only the benchmark knows what one of its units is. What the orchestrator keeps is the identity of
a unit and the seed material it is derived from - both of which belong to the competition, because
both have to be reproducible from the published record by someone holding neither the pool nor the
simulator.
"""

from __future__ import annotations

import hashlib

import pytest

from icilval.benchmarks import units as plugin_units
from icilval.benchmarks.units import DerivationError


class Plugin:
    """A benchmark whose units are a pure function of its seed material."""

    id = "fakesim"
    api_version = 1

    def __init__(self, *, count: int | None = None, raises: bool = False, bad: bool = False):
        self.calls: list[dict] = []
        self._count = count
        self._raises = raises
        self._bad = bad

    def info(self):
        return {"id": self.id, "api_version": 1}

    def derive_units(self, *, seed_material: str, count: int, suite: str):
        self.calls.append({"seed_material": seed_material, "count": count, "suite": suite})
        if self._raises:
            raise RuntimeError("catalogue unreachable")
        n = count if self._count is None else self._count
        if self._bad:
            return ["not a dict"] * n
        # `hashlib`, not `hash()`: the builtin is salted per process, and a benchmark deriving
        # units from it would give a different list on every run - the exact failure the ABI's
        # "derive from a hash of seed_material, never a global RNG" sentence exists to prevent.
        return [
            {
                "task": f"t{i}",
                "scene_seed": int(
                    hashlib.sha256(f"{seed_material}|{i}".encode()).hexdigest()[:8], 16
                ),
                "unit_id": "mine",
            }
            for i in range(n)
        ]


class Sim:
    def __init__(self, benchmark, name="fakesim"):
        self.name = name
        self.benchmark = benchmark


class Spec:
    """The slice of `icilval.spec.Spec` derivation touches."""

    def __init__(self, skills=("rt_a", "rt_b"), per_skill=2, benchmark=None):
        self._skills = list(skills)
        self._per = per_skill
        self.sim = Sim(benchmark)

    def skills(self, track):
        return list(self._skills)

    def units_per_skill(self, track, size=None):
        return self._per

    def default_size(self, track):
        return "smoke"

    def skill(self, name):
        return {"code": name[-1] * 2, "tasks": {"suite": "v1"}}

    def skill_code(self, name):
        return name[-1] * 2


@pytest.fixture
def patched(monkeypatch):
    def use(spec):
        monkeypatch.setattr(plugin_units, "for_skill", lambda _spec, _skill: spec.sim)
        return spec

    return use


def test_units_come_from_the_benchmark_and_ids_from_the_competition(patched):
    plugin = Plugin()
    spec = patched(Spec(benchmark=plugin))
    units = plugin_units.plugin_units(spec, "sensorimotor", "duel123", "smoke")

    assert len(units) == 4
    # The id is the competition's shape, running across the whole field, not the benchmark's.
    assert [u["unit_id"] for u in units] == ["aa-000", "aa-001", "bb-002", "bb-003"]
    assert [u["skill"] for u in units] == ["rt_a", "rt_a", "rt_b", "rt_b"]
    # What the benchmark said about its own units is passed through untouched.
    assert all("scene_seed" in u and u["task"].startswith("t") for u in units)


def test_a_benchmark_cannot_choose_a_units_identity(patched):
    """The plugin returns `unit_id: "mine"` on every unit; ids are the competition's to set."""
    spec = patched(Spec(benchmark=Plugin()))
    units = plugin_units.plugin_units(spec, "sensorimotor", "duel123", "smoke")
    assert "mine" not in [u["unit_id"] for u in units]


def test_derivation_is_reproducible_from_the_published_record(patched):
    """Same spec, same duel id, same list - that is what lets a third party check a duel."""
    first = plugin_units.plugin_units(patched(Spec(benchmark=Plugin())), "t", "duelA", "smoke")
    again = plugin_units.plugin_units(patched(Spec(benchmark=Plugin())), "t", "duelA", "smoke")
    other = plugin_units.plugin_units(patched(Spec(benchmark=Plugin())), "t", "duelB", "smoke")

    assert first == again
    assert [u["seed"] for u in first] != [u["seed"] for u in other]


def test_the_seed_material_is_the_duel_and_the_skill_and_nothing_else(patched):
    plugin = Plugin()
    plugin_units.plugin_units(patched(Spec(benchmark=plugin)), "t", "duelA", "smoke")
    assert [c["seed_material"] for c in plugin.calls] == ["duelA|rt_a", "duelA|rt_b"]
    assert {c["suite"] for c in plugin.calls} == {"v1"}


def test_a_benchmark_returning_the_wrong_number_of_units_is_an_error(patched):
    spec = patched(Spec(benchmark=Plugin(count=1)))
    with pytest.raises(DerivationError, match="returned 1 units, not the 2"):
        plugin_units.plugin_units(spec, "sensorimotor", "duel123", "smoke")


def test_a_benchmark_that_raises_is_named_rather_than_traced(patched):
    spec = patched(Spec(benchmark=Plugin(raises=True)))
    with pytest.raises(DerivationError, match="could not derive its units: catalogue unreachable"):
        plugin_units.plugin_units(spec, "sensorimotor", "duel123", "smoke")


def test_a_unit_that_is_not_a_mapping_is_refused(patched):
    spec = patched(Spec(benchmark=Plugin(bad=True)))
    with pytest.raises(DerivationError, match="unit 0 is str"):
        plugin_units.plugin_units(spec, "sensorimotor", "duel123", "smoke")


def test_an_in_repo_simulator_is_told_to_use_its_pool(patched):
    spec = patched(Spec(benchmark=None))
    spec.sim.name = "libero"
    with pytest.raises(DerivationError, match="derives its units from a pool"):
        plugin_units.plugin_units(spec, "sensorimotor", "duel123", "smoke")
