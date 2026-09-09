"""Materializing a smoke duel's units with the real generators in worker processes."""

import pytest

from icilval.duel.materialize import GenerationContext, materialize
from icilval.ids import ModelRef, duel_id
from icilval.pools.demos import load_demo
from icilval.pools.units import derive_units

from .conftest import bpp_root, grasp_sources_root

pytestmark = [pytest.mark.sim, pytest.mark.slow]


def test_materialize_smoke_duel(spec, smoke_pool, tmp_path):
    root = grasp_sources_root()
    if not any(root.rglob("*_demo.hdf5")):
        pytest.skip(f"no grasp-source files under {root}")
    did = duel_id(spec.version, spec.track_id, ModelRef.make("a/b", "1" * 40), None)
    units = [u.as_dict() for u in derive_units(smoke_pool, spec, did, "smoke")]
    ctx = GenerationContext(bpp_root=bpp_root(), raw_root=root.parent, libero_datasets=root)
    out, report = materialize(units, smoke_pool, spec, did, tmp_path / "assets", ctx, workers=2)
    assert [u["unit_id"] for u in out] == [u["unit_id"] for u in units]
    scored = [u for u in out if not u.get("diagnostic")]
    assert scored and all(u["prompt_sha256"] for u in scored), report.notes()
    for u in scored:
        demo = load_demo(tmp_path / "assets" / f"{u['unit_id']}.npz")
        assert (
            demo["meta"]["demo_id"] == u["demo"]
            and demo["actions"].shape[0] == u["generation"]["steps"]
        )
        if u["skill"] == "draw_anything":
            p = u["instance_params"]
            assert p["demo_angle_rad"] == pytest.approx(float(demo["boundary_angle"]), abs=1e-5)
            if u["change"]["kind"] == "board_angle":
                assert (
                    u["change"]["delta_rad"]
                    >= spec.changes("draw_anything")["board_angle"]["min_delta_rad"]
                )
            else:
                assert p["angle_rad"] == pytest.approx(float(demo["boundary_angle"]), abs=1e-5)
        else:
            assert (
                demo["agentview"].shape[1:] == (128, 128, 3) and demo["meta"]["task"] == u["task"]
            )
    diag = [u for u in out if u.get("diagnostic")]
    assert diag and all(u["prompt_sha256"] is None for u in diag)
    summary = report.summary()
    assert summary["pick_and_place"]["failed"] == 0 and summary["draw_anything"]["failed"] == 0
