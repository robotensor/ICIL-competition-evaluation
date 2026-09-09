"""Generating a drawing of each family and replaying its demonstration on the board."""

import json

import numpy as np
import pytest

from icilval.pools.demos import load_demo
from icilval.simulators.draw.env import DrawBoard
from icilval.simulators.draw.generate import generate_drawing, glyph_strokes

pytestmark = pytest.mark.sim


def test_generate_each_family_and_replay(spec, tmp_path):
    board = DrawBoard(spec, "draw_anything")
    families = [
        k for k in spec.skill_generation("draw_anything")["families"] if not k.startswith("_")
    ]
    for n, family in enumerate(families):
        res = generate_drawing(
            spec,
            "draw_anything",
            family,
            100 + n,
            tmp_path / f"{family}.npz",
            f"generated/da-{n:03d}",
        )
        assert res.success, res.error
        demo = load_demo(res.npz)
        steps = demo["actions"].shape[0]
        assert steps == res.steps and demo["image"].shape == (steps, 224, 224, 3)
        assert demo["agent_pos"].shape == (steps, 2) and demo["pen_down"].shape == (steps, 1)
        assert demo["drawing"].shape == (512, 512) and demo["drawing"].sum() > 500
        meta = demo["meta"]
        assert (
            meta["family"] == family
            and meta["seed"] == 100 + n
            and meta["demo_id"] == f"generated/da-{n:03d}"
        )
        assert float(demo["boundary_angle"]) == pytest.approx(res.boundary_angle)
        lo, hi = spec.env("draw_anything")["board_angle_range_rad"]
        assert lo <= res.boundary_angle <= hi
        if family == "glyph":
            assert (
                meta["character"]
                in spec.skill_generation("draw_anything")["families"]["glyph"]["characters"]
            )
        # the demonstration's own actions on its own board redraw its strokes
        angle = float(demo["boundary_angle"])
        cursor = (int(demo["agent_pos"][0][0]), int(demo["agent_pos"][0][1]))
        board.reset(3, angle=angle, cursor=cursor, target=demo["drawing"], target_angle=angle)
        for a in demo["actions"]:
            _, chamfer = board.step(a)
        assert chamfer < 0.5, (family, chamfer)
    board.close()
    # determinism: the same seed gives the same actions and strokes
    a = generate_drawing(spec, "draw_anything", "polygon", 7, tmp_path / "a.npz", "generated/a")
    b = generate_drawing(spec, "draw_anything", "polygon", 7, tmp_path / "b.npz", "generated/a")
    da, db = load_demo(a.npz), load_demo(b.npz)
    assert np.array_equal(da["actions"], db["actions"]) and np.array_equal(
        da["drawing"], db["drawing"]
    )
    assert a.sha256 == b.sha256
    assert json.loads(json.dumps(da["meta"]))["parts"] == a.parts


def test_glyph_strokes_are_normalized_polylines():
    strokes = glyph_strokes("B", "DejaVuSans")
    assert strokes and all(s.ndim == 2 and s.shape[1] == 2 for s in strokes)
    allpts = np.concatenate(strokes)
    assert allpts.min() >= -1e-9 and allpts.max() <= 1.0 + 1e-9
