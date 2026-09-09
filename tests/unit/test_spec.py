import json

import pytest

from icilval.spec import load_spec_file, validate_spec


def test_spec_loads_and_fingerprints(spec):
    assert spec.version == 4
    assert spec.track_id == "icil_1demo"
    assert spec.skills == ("pick_and_place", "draw_anything")
    assert (
        spec.skill_code("pick_and_place") == "pp" and spec.skill_for_code("da") == "draw_anything"
    )
    assert (
        spec.simulator("pick_and_place") == "libero" and spec.simulator("draw_anything") == "draw"
    )
    assert spec.tasks("pick_and_place")["views"] and spec.tasks("draw_anything")["files"]
    assert spec.tasks("pick_and_place")["grasp_sources"]["dataset"]
    assert "perturbations" not in spec.skill("pick_and_place")
    lo, hi = spec.env("draw_anything")["board_angle_range_rad"]
    assert lo < 0 < hi
    assert spec.units_per_duel("smoke") == len(spec.skills) * spec.units_per_skill("smoke")
    assert spec.size_of("bogus") == spec.default_size
    assert len(spec.fingerprint) == 64
    assert 0 <= spec.score_margin <= 100
    assert spec.success("draw_anything")["threshold"] == 2.5
    assert spec.success("pick_and_place") is None
    assert spec.env("pick_and_place")["prompt_actions_per_chunk"] == 20
    assert spec.env("draw_anything")["prompt_actions_per_chunk"] == 10


def test_spec_v4_generation_changes_and_diagnostics(spec):
    assert set(spec.changes("pick_and_place")) == {
        "displace",
        "camera",
        "lighting",
        "observation",
        "robot_pose",
    }
    assert set(spec.changes("draw_anything")) == {"board_angle", "pen_start"}
    assert all(isinstance(v, dict) for v in spec.changes("pick_and_place").values())
    assert spec.changes("pick_and_place")["displace"]["min_delta_m"] > 0
    assert spec.sub_score_key("pick_and_place") == "change.kind"
    assert spec.sub_score_key("draw_anything") == "instance_params.family"
    assert set(spec.skill_generation("draw_anything")["families"]) >= {"bpp", "polygon", "glyph"}
    assert spec.skill_generation("pick_and_place") == {}
    assert spec.generation["max_attempts"] >= 1 and spec.generation["workers"] >= 1
    assert spec.catalogue["repo"] and spec.pools is spec.catalogue
    diag = spec.diagnostics["handmade_drawings"]
    assert diag["skill"] in spec.skills and diag["units_per_duel"] >= 0
    assert not any(k.startswith("_") for k in spec.diagnostics)


def test_validate_rejects_bad_specs(spec, tmp_path):
    doc = json.loads(json.dumps(spec.raw))
    doc["duel"]["score_margin"] = 101
    assert any("score_margin" in e for e in validate_spec(doc))
    doc = json.loads(json.dumps(spec.raw))
    doc["skills"] = {}
    assert any("skills non-empty" in e for e in validate_spec(doc))
    doc = json.loads(json.dumps(spec.raw))
    doc["skills"]["draw_anything"]["code"] = "pp"
    assert any("code unique" in e for e in validate_spec(doc))
    doc = json.loads(json.dumps(spec.raw))
    doc["skills"]["draw_anything"]["simulator"] = "unity"
    assert any("simulator" in e for e in validate_spec(doc))
    doc = json.loads(json.dumps(spec.raw))
    del doc["skills"]["draw_anything"]["success"]
    assert any("threshold" in e for e in validate_spec(doc))
    doc = json.loads(json.dumps(spec.raw))
    doc["skills"]["pick_and_place"]["perturbations"] = {"spatial": {}}
    assert any("perturbations removed" in e for e in validate_spec(doc))
    doc = json.loads(json.dumps(spec.raw))
    doc["skills"]["draw_anything"]["environment"]["board_angle_range_rad"] = [0.5, -0.5]
    assert any("board_angle_range_rad" in e for e in validate_spec(doc))
    doc = json.loads(json.dumps(spec.raw))
    del doc["skills"]["pick_and_place"]["tasks"]["views"]
    assert any("tasks views|files" in e for e in validate_spec(doc))
    doc = json.loads(json.dumps(spec.raw))
    doc["skills"]["pick_and_place"]["changes"] = {"_comment": "nothing"}
    assert any("changes non-empty" in e for e in validate_spec(doc))
    doc = json.loads(json.dumps(spec.raw))
    del doc["skills"]["draw_anything"]["sub_scores"]
    assert any("sub_scores.by" in e for e in validate_spec(doc))
    doc = json.loads(json.dumps(spec.raw))
    doc["generation"]["max_attempts"] = 0
    assert any("max_attempts" in e for e in validate_spec(doc))
    doc = json.loads(json.dumps(spec.raw))
    doc["pools"] = doc["catalogue"]
    assert any("pools renamed" in e for e in validate_spec(doc))
    doc = json.loads(json.dumps(spec.raw))
    doc["diagnostics"]["handmade_drawings"]["skill"] = "goal_chain"
    assert any("diagnostics.handmade_drawings.skill" in e for e in validate_spec(doc))
    doc = json.loads(json.dumps(spec.raw))
    doc["duel"]["default_size"] = "gigantic"
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        load_spec_file(p)
