"""The drawing generator's pure parts: families -> parts -> actions, and rotation."""

import numpy as np
import pytest

from icilval.simulators.draw.generate import (
    CENTER,
    FINAL_HOLD_STEPS,
    MARGIN_PX,
    NOISE_BOUNDS_PX,
    _along,
    _length,
    actions_from_parts,
    bpp_parts,
    polygon_parts,
    rotate_actions,
)

BOARD = 350.0


def _cfg(spec, family):
    return {
        k: v
        for k, v in spec.skill_generation("draw_anything")["families"][family].items()
        if not k.startswith("_")
    }


def test_bpp_parts_follow_bpps_rules(spec):
    cfg = _cfg(spec, "bpp")
    for seed in range(30):
        parts = bpp_parts(np.random.default_rng(seed), cfg, BOARD, MARGIN_PX)
        assert cfg["min_parts"] <= len(parts) <= cfg["max_parts"]
        assert parts[0]["type"] != "movement"
        assert not any(
            a["type"] == "movement" and b["type"] == "movement"
            for a, b in zip(parts[:-1], parts[1:], strict=True)
        )
        for part in parts:
            assert part["pen"] == (0.0 if part["type"] == "movement" else 1.0)
            for key in ("start", "end"):
                assert np.all(part[key] >= CENTER - BOARD / 2) and np.all(
                    part[key] <= CENTER + BOARD / 2
                )
    a = bpp_parts(np.random.default_rng(3), cfg, BOARD, MARGIN_PX)
    b = bpp_parts(np.random.default_rng(3), cfg, BOARD, MARGIN_PX)
    assert len(a) == len(b) and all(
        np.allclose(x["end"], y["end"]) for x, y in zip(a, b, strict=True)
    )


def test_polygon_parts_close_the_loop(spec):
    cfg = _cfg(spec, "polygon")
    lo, hi = cfg["vertices"]
    for seed in range(20):
        parts = polygon_parts(np.random.default_rng(seed), cfg, BOARD, MARGIN_PX)
        assert lo <= len(parts) <= hi and all(p["type"] == "straight" for p in parts)
        assert np.allclose(parts[-1]["end"], parts[0]["start"])
        for a, b in zip(parts[:-1], parts[1:], strict=True):
            assert np.allclose(a["end"], b["start"])


def test_actions_from_parts_shape_pen_and_hold(spec):
    parts = bpp_parts(np.random.default_rng(5), _cfg(spec, "bpp"), BOARD, MARGIN_PX)
    actions = actions_from_parts(parts, np.random.default_rng(5), 10, 250.0, BOARD, MARGIN_PX)
    assert actions.ndim == 2 and actions.shape[1] == 3 and actions.dtype == np.float32
    assert set(np.unique(actions[:, 2])) <= {0.0, 1.0}
    tail = actions[-FINAL_HOLD_STEPS:]  # the final hold: the last end, noised like BPP does
    assert np.all(tail[:, 2] == tail[0, 2])
    assert np.all(np.abs(tail[:, :2] - parts[-1]["end"]) <= NOISE_BOUNDS_PX + 1e-3)
    again = actions_from_parts(parts, np.random.default_rng(5), 10, 250.0, BOARD, MARGIN_PX)
    assert np.array_equal(actions, again)
    faster = actions_from_parts(parts, np.random.default_rng(5), 10, 400.0, BOARD, MARGIN_PX)
    assert len(faster) <= len(actions)
    poly = [
        {
            "type": "polyline",
            "start": np.array([100.0, 100.0]),
            "end": np.array([200.0, 100.0]),
            "points": np.array([[100.0, 100.0], [150.0, 150.0], [200.0, 100.0]]),
            "pen": 1.0,
        }
    ]
    acts = actions_from_parts(poly, np.random.default_rng(1), 10, 100.0, BOARD, MARGIN_PX)
    assert len(acts) >= 8 + FINAL_HOLD_STEPS and np.allclose(acts[-1, :2], [200.0, 100.0])


def test_rotate_actions_about_the_centre_and_along():
    acts = np.array([[CENTER + 10, CENTER, 1.0], [CENTER, CENTER + 10, 0.0]], dtype=np.float32)
    rot = rotate_actions(acts, np.pi / 2)
    assert np.allclose(rot[0, :2], [CENTER, CENTER + 10], atol=1e-4) and rot[0, 2] == 1.0
    assert np.allclose(rotate_actions(acts, 0.0), acts)
    line = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    assert _length(line) == pytest.approx(10.0)
    pts = _along(line, 5)
    assert pts.shape == (5, 2) and np.allclose(pts[0], [0, 0]) and np.allclose(pts[-1], [6, 8])
    assert np.allclose(pts[2], [3, 4])
