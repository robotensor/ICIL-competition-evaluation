"""Sampling a LIBERO scene change is pure and stays within the published ranges."""

import numpy as np

from icilval.rng import HashRng
from icilval.simulators.libero.changes import (
    DISPLACE_CANDIDATES,
    ObservationChange,
    init_noise_of,
    sample_change,
    target_object,
)

PP = "pick_and_place"
STEPS = [["Grasp", "akita_black_bowl_1"], ["Place", "akita_black_bowl_1", "plate_1"]]


def test_sample_change_covers_the_menu_within_its_ranges(spec):
    menu = spec.changes(PP)
    seen = set()
    for i in range(200):
        c = sample_change(spec, PP, STEPS, HashRng("t", i))
        seen.add(c["kind"])
        cfg = menu[c["kind"]]
        if c["kind"] == "displace":
            assert c["target"] == "akita_black_bowl_1"
            assert len(c["candidates"]) == DISPLACE_CANDIDATES
            for cand in c["candidates"]:
                d = float(np.hypot(*cand["delta_xy"]))
                assert cfg["min_delta_m"] - 1e-3 <= d <= cfg["radius_m"] + 1e-3
                assert abs(cand["yaw"]) <= cfg["yaw_max_rad"]
            assert c["clearance_m"] == cfg["clearance_m"] and c["min_delta_m"] == cfg["min_delta_m"]
        elif c["kind"] == "camera":
            assert c["camera"] == spec.media["video"]["camera"]
            assert all(abs(x) <= cfg["pos_jitter_m"] for x in c["pos_delta"])
            assert all(abs(a) <= cfg["angle_jitter_rad"] for a in c["angles"])
        elif c["kind"] == "lighting":
            lo, hi = cfg["diffuse_scale"]
            assert len(c["lights"]) == 4 and any(light["active"] for light in c["lights"])
            assert all(lo <= light["diffuse_scale"] <= hi for light in c["lights"])
            assert cfg["headlight_scale"][0] <= c["headlight_scale"] <= cfg["headlight_scale"][1]
        elif c["kind"] == "observation":
            lo, hi = cfg["brightness_scale"]
            assert lo <= c["brightness_scale"] <= hi
            assert cfg["noise_sigma"][0] <= c["noise_sigma"] <= cfg["noise_sigma"][1]
            assert 0 <= c["noise_seed"] < (1 << 31)
        elif c["kind"] == "robot_pose":
            assert c["init_noise_magnitude"] == cfg["init_noise_magnitude"]
    assert seen == set(menu)


def test_sample_change_is_deterministic(spec):
    a = sample_change(spec, PP, STEPS, HashRng("d", 1))
    assert a == sample_change(spec, PP, STEPS, HashRng("d", 1))
    assert [sample_change(spec, PP, STEPS, HashRng("d", i)) for i in range(5)] != [a] * 5


def test_target_object_and_init_noise():
    assert target_object(STEPS) == "akita_black_bowl_1"
    assert target_object([["Open", "drawer"], ["Place", "x", "y"]]) is None
    assert init_noise_of({"kind": "robot_pose", "init_noise_magnitude": 0.05}) == 0.05
    assert init_noise_of({"kind": "camera"}) is None and init_noise_of(None) is None


def test_observation_change_is_seeded_clipped_and_leaves_the_rest_alone():
    obs = {
        "agentview": np.full((4, 4, 3), 200, np.uint8),
        "eye_in_hand": np.zeros((4, 4, 3), np.uint8),
        "ee_pos": np.zeros(3, np.float32),
    }
    change = {"kind": "observation", "brightness_scale": 1.5, "noise_sigma": 3.0, "noise_seed": 7}
    a = ObservationChange(change)(obs)
    b = ObservationChange(change)(obs)
    assert np.array_equal(a["agentview"], b["agentview"]) and a["agentview"].dtype == np.uint8
    assert a["agentview"].max() == 255 and a["eye_in_hand"].min() == 0
    assert a["ee_pos"] is obs["ee_pos"]
    assert ObservationChange({"kind": "camera"})(obs) is obs and ObservationChange(None)(obs) is obs
    dim = ObservationChange(
        {"kind": "observation", "brightness_scale": 0.5, "noise_sigma": 0.0, "noise_seed": 1}
    )(obs)
    assert dim["agentview"].max() == 100


def test_sample_change_can_be_forced_to_a_kind(spec):
    import pytest

    for kind in spec.changes(PP):
        assert sample_change(spec, PP, STEPS, HashRng("f", 1), kind=kind)["kind"] == kind
    with pytest.raises(ValueError, match="not in"):
        sample_change(spec, PP, STEPS, HashRng("f", 1), kind="teleport")
