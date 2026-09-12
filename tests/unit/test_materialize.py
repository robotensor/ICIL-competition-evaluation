"""Fixing a duel's prompts before either side runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from icilval import materialize
from icilval.materialize import MaterializeFailed


class FakeBenchmark:
    """Writes a prompt file per unit and verifies it, with no simulator."""

    def __init__(self, fail_on: set[str] | None = None, wrong: set[str] | None = None):
        self.fail_on = fail_on or set()
        self.wrong = wrong or set()
        self.calls: list[str] = []

    def materialize_command(self, *, unit, out_dir):
        return ["fake", unit["unit_id"], out_dir]

    def verify_prompt(self, *, path, unit):
        uid = unit["unit_id"]
        if uid in self.wrong:
            return {"ok": False, "sha256": "0" * 64, "problems": ["task does not match"]}
        return {"ok": True, "sha256": f"{abs(hash(uid)) % (16**64):064x}", "problems": []}

    def run(self, argv, timeout_s):
        uid = argv[1]
        self.calls.append(uid)
        if uid in self.fail_on:
            return 1, "the expert never succeeded"
        (Path(argv[2]) / "prompt.npz").write_bytes(b"x")
        return 0, ""


def _units(n=2):
    return [
        {"unit_id": f"pp-{i}", "skill": "pick_and_place", "task": "t", "seed": i} for i in range(n)
    ]


def test_a_pool_field_materializes_nothing(spec):
    assert materialize.needed(spec, spec.sole_track) is False


def test_every_unit_gets_a_verified_prompt(spec, tmp_path):
    bench = FakeBenchmark()
    out = materialize.materialize(
        spec, spec.sole_track, _units(), tmp_path / "prompts", benchmark=bench, runner=bench.run
    )
    assert bench.calls == ["pp-0", "pp-1"]
    assert set(out.prompts) == {"pp-0", "pp-1"}
    assert all(len(p.sha256) == 64 for p in out.prompts.values())
    manifest = json.loads((tmp_path / "prompts" / "prompts.json").read_text())
    assert [e["unit_id"] for e in manifest] == ["pp-0", "pp-1"]


def test_a_prompt_that_could_not_be_produced_stops_the_duel(spec, tmp_path):
    bench = FakeBenchmark(fail_on={"pp-1"})
    with pytest.raises(MaterializeFailed, match="materialize exited 1"):
        materialize.materialize(
            spec, spec.sole_track, _units(), tmp_path / "p", benchmark=bench, runner=bench.run
        )


def test_a_prompt_that_is_not_the_one_asked_for_stops_the_duel(spec, tmp_path):
    """Both sides must see the prompt the unit named, so a mismatch fails here and not later."""
    bench = FakeBenchmark(wrong={"pp-0"})
    with pytest.raises(MaterializeFailed, match="not the one the unit asked for"):
        materialize.materialize(
            spec, spec.sole_track, _units(), tmp_path / "p", benchmark=bench, runner=bench.run
        )


def test_a_substitution_is_carried_onto_the_published_prompt(spec, tmp_path):
    """A unit whose expert never succeeded is replaced during materializing, and the record says
    which unit it stood in for."""
    bench = FakeBenchmark()
    units = _units(1)
    units[0]["substituted_from"] = "pp-9"
    out = materialize.materialize(
        spec, spec.sole_track, units, tmp_path / "p", benchmark=bench, runner=bench.run
    )
    assert out.prompts["pp-0"].substituted_from == "pp-9"
    assert out.manifest()[0]["substituted_from"] == "pp-9"


def test_the_published_manifest_can_be_rechecked_without_a_simulator(spec, tmp_path):
    bench = FakeBenchmark()
    out = materialize.materialize(
        spec, spec.sole_track, _units(), tmp_path / "p", benchmark=bench, runner=bench.run
    )
    assert materialize.verify_against(out.root, out.manifest()) == []
    (out.root / "pp-0" / "prompt.npz").unlink()
    assert materialize.verify_against(out.root, out.manifest()) == ["pp-0: empty"]


def test_a_materialized_field_is_recognised(spec, tmp_path):
    doc = json.loads(spec.path.read_text())
    doc["tracks"]["sensorimotor"]["prompts"] = "materialized"
    doc["tracks"]["sensorimotor"]["protocol"] = "same_initial_state"
    doc["tracks"]["sensorimotor"]["prompt_instance_disjoint"] = False
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(doc))
    from icilval.spec import load_spec_file

    assert materialize.needed(load_spec_file(path), "sensorimotor") is True
