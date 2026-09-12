"""The benchmark plugin ABI, exercised through a fake plugin.

Nothing here imports a benchmark: the point of the ABI is that this repository can describe and
check one without having it installed, so the fake below is the only plugin the suite has.
"""

from __future__ import annotations

from typing import Any

from icilval.benchmarks import (
    BENCHMARK_API_VERSION,
    COMMAND_METHODS,
    PURE_METHODS,
    validate_plugin,
)
from icilval.benchmarks.api import METHODS


class FakeBenchmark:
    """A minimal plugin that satisfies the ABI, and the base for every mutilation below."""

    id = "fake"
    api_version = BENCHMARK_API_VERSION

    def info(self) -> dict[str, Any]:
        return {"id": self.id, "api_version": self.api_version, "protocol": "fake_1demo"}

    def catalogue(self) -> dict[str, Any]:
        return {"suites": {"v1": ["t0", "t1"]}, "tasks": {"t0": {}, "t1": {}}}

    def derive_units(self, *, seed_material: str, count: int, suite: str) -> list[dict[str, Any]]:
        return [{"unit_index": i, "task": f"t{i % 2}", "suite": suite} for i in range(count)]

    def verify_prompt(self, *, path: str, unit: Any) -> dict[str, Any]:
        return {"ok": True, "sha256": "0" * 64, "problems": []}

    def read_result(self, *, out_dir: str) -> dict[str, Any]:
        return {"success": True, "void": False, "steps": 1, "error": None}

    def materialize_command(self, *, unit: Any, out_dir: str) -> list[str]:
        return ["fake-bench", "materialize", "--out", out_dir]

    def run_command(
        self, *, unit: Any, prompt: str, out_dir: str, policy_address: str, **extra: Any
    ) -> list[str]:
        return ["fake-bench", "run-unit", "--prompt", prompt, "--policy", policy_address]


def test_a_conforming_plugin_validates():
    assert validate_plugin(FakeBenchmark()) == []


def test_the_two_halves_together_are_the_whole_surface():
    assert set(PURE_METHODS) | set(COMMAND_METHODS) == set(METHODS)
    assert not set(PURE_METHODS) & set(COMMAND_METHODS)


def test_every_missing_method_is_reported_by_name():
    for name in METHODS:
        broken = type("Broken", (FakeBenchmark,), {name: None})()
        errors = validate_plugin(broken)
        assert any(e.startswith(f"{name}: ") for e in errors), (name, errors)


def test_a_method_that_is_not_callable_is_refused():
    broken = type("Broken", (FakeBenchmark,), {"catalogue": 3})()
    assert "catalogue: not callable" in validate_plugin(broken)


def test_a_method_missing_a_keyword_is_refused():
    class Broken(FakeBenchmark):
        def derive_units(self, *, seed_material: str, count: int) -> list[dict[str, Any]]:
            return []

    assert "derive_units: does not accept suite" in validate_plugin(Broken())


def test_a_method_taking_kwargs_is_accepted():
    """A plugin is free to absorb the call; only refusing a keyword is an error."""

    class Loose(FakeBenchmark):
        def run_command(self, **kw: Any) -> list[str]:
            return []

    assert validate_plugin(Loose()) == []


def test_an_id_must_be_a_non_empty_string():
    for value in ("", None, 7):
        broken = type("Broken", (FakeBenchmark,), {"id": value})()
        assert any(e.startswith("id: ") for e in validate_plugin(broken))


def test_a_plugin_speaking_another_abi_version_is_refused():
    broken = type("Broken", (FakeBenchmark,), {"api_version": BENCHMARK_API_VERSION + 1})()
    errors = validate_plugin(broken)
    assert any("api_version" in e and str(BENCHMARK_API_VERSION) in e for e in errors)


def test_a_boolean_is_not_an_api_version():
    """`True == 1` in Python, so a plugin with `api_version = True` would otherwise pass."""
    broken = type("Broken", (FakeBenchmark,), {"api_version": True})()
    assert "api_version: expected an integer" in validate_plugin(broken)


def test_errors_accumulate_rather_than_stopping_at_the_first():
    broken = type("Broken", (FakeBenchmark,), {"id": "", "api_version": "1", "info": None})()
    errors = validate_plugin(broken)
    assert len(errors) >= 3


def test_derive_units_is_a_pure_function_of_its_arguments():
    plugin = FakeBenchmark()
    first = plugin.derive_units(seed_material="abc", count=4, suite="v1")
    second = plugin.derive_units(seed_material="abc", count=4, suite="v1")
    assert first == second and len(first) == 4
