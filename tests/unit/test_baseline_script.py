"""The organizer sweep derives units the way a duel does, with a forced change kind."""

import importlib.util
from pathlib import Path

from icilval.spec import _repo_root

from .test_units import DA, PP, make_pool


def _baseline():
    path = (_repo_root() or Path.cwd()) / "scripts" / "baseline.py"
    spec = importlib.util.spec_from_file_location("baseline_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sweep_units_forces_a_change_kind_and_spreads_over_tasks(spec):
    baseline = _baseline()
    pool = make_pool()
    units = baseline.sweep_units(pool, spec, PP, 2, None, None, change="camera")
    assert len(units) == 2 * len(pool.eligible(PP))
    assert all(u["change"]["kind"] == "camera" and "pos_delta" in u["change"] for u in units)
    assert [u["unit_id"] for u in units] == [f"pp-{i:04d}" for i in range(len(units))]
    assert all(u["demo"] == f"generated/{u['unit_id']}" for u in units)
    again = baseline.sweep_units(pool, spec, PP, 2, None, None, change="camera")
    assert again == units
    mixed = baseline.sweep_units(pool, spec, PP, 3, None, None)
    assert {u["change"]["kind"] for u in mixed} <= set(spec.changes(PP))
    glyph = baseline.sweep_units(pool, spec, DA, 1, None, None, family="glyph", change="pen_start")
    assert len(glyph) == 1 and glyph[0]["instance_params"]["family"] == "glyph"
    assert glyph[0]["change"] == {"kind": "pen_start"}
    assert baseline.sweep_units(pool, spec, PP, 1, None, 2)  # max_tasks spreads over the list
