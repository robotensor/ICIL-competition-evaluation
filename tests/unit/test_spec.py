import json

import pytest

from icilval.spec import load_spec_file, validate_spec


def test_spec_loads_and_fingerprints(spec):
    assert spec.version == 3
    assert spec.track_id == "icil_1demo"
    assert spec.skills == ("pick_and_place", "draw_anything")
    assert (
        spec.skill_code("pick_and_place") == "pp" and spec.skill_for_code("da") == "draw_anything"
    )
    assert (
        spec.simulator("pick_and_place") == "libero" and spec.simulator("draw_anything") == "draw"
    )
    assert "perturbations" not in spec.skill("pick_and_place")
    lo, hi = spec.env("draw_anything")["board_angle_range_rad"]
    assert lo < 0 < hi
    assert spec.units_per_duel("smoke") == 2 * spec.units_per_skill("smoke")
    assert spec.size_of("bogus") == spec.default_size
    assert len(spec.fingerprint) == 64
    assert 0 <= spec.score_margin <= 100
    assert spec.success("draw_anything")["threshold"] > 0 and spec.success("pick_and_place") is None
    assert spec.env("pick_and_place")["prompt_actions_per_chunk"] == 20
    assert spec.env("draw_anything")["prompt_actions_per_chunk"] == 10


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
    doc["duel"]["default_size"] = "gigantic"
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        load_spec_file(p)
