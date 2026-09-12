import json

import pytest

from icilval.spec import load_spec_file, validate_spec


def test_spec_loads_and_fingerprints(spec):
    assert spec.version == 5
    assert spec.tracks == ("sensorimotor",) and spec.sole_track == "sensorimotor"
    assert spec.all_skills == ("pick_and_place", "goal_chain", "draw_anything")
    assert (
        spec.skill_code("pick_and_place") == "pp" and spec.skill_for_code("da") == "draw_anything"
    )
    assert (
        spec.simulator("pick_and_place") == "libero" and spec.simulator("draw_anything") == "draw"
    )
    assert spec.simulator("goal_chain") == "libero" and spec.skill_code("goal_chain") == "gc"
    assert spec.tasks("goal_chain")["views"] and spec.tasks("draw_anything")["files"]
    assert "perturbations" not in spec.skill("pick_and_place")
    lo, hi = spec.env("draw_anything")["board_angle_range_rad"]
    assert lo < 0 < hi
    track = spec.sole_track
    assert spec.units_per_side(track, "smoke") == len(spec.skills(track)) * spec.units_per_skill(
        track, "smoke"
    )
    assert spec.size_of(track, "bogus") == spec.default_size(track)
    assert len(spec.fingerprint) == 64
    assert 0 <= spec.score_margin(track) <= 100
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
    del doc["skills"]["goal_chain"]["tasks"]["views"]
    assert any("tasks views|files" in e for e in validate_spec(doc))
    doc = json.loads(json.dumps(spec.raw))
    doc["duel"]["default_size"] = "gigantic"
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        load_spec_file(p)


# ------------------------------------------------------------------ spec v5: the fields


def _save(tmp_path, doc):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(doc))
    return path


def _two_fields(spec):
    """The contract with a valid second field, `goal_chain` moved into it."""
    doc = _doc(spec)
    first = doc["tracks"]["sensorimotor"]
    first["skills"] = [s for s in first["skills"] if s != "goal_chain"]
    second = json.loads(json.dumps(first))
    second.update(id="second", code="xx", slug="second-field", skills=["goal_chain"])
    doc["tracks"]["second"] = second
    doc["baselines"]["second"] = None
    doc["pools"]["tracks"]["second"] = {"version": None, "pool_id": None}
    return doc


def _doc(spec):
    """A mutable copy of the real contract, so a rule is tested against the shipped spec."""
    return json.loads(spec.path.read_text())


def test_a_v3_spec_is_refused_rather_than_half_read(spec):
    """A singular `track` read by this code would render the wrong numbers in silence."""
    doc = _doc(spec)
    doc["track"] = {"id": "icil_1demo", "k_demos": 1, "language": "none"}
    assert any("track removed" in e for e in validate_spec(doc))


def test_every_skill_belongs_to_exactly_one_field(spec):
    doc = _doc(spec)
    assert validate_spec(doc) == []

    orphan = _doc(spec)
    orphan["tracks"]["sensorimotor"]["skills"] = ["pick_and_place"]
    assert any("partition" in e for e in validate_spec(orphan))

    twice = _doc(spec)
    twice["tracks"]["sensorimotor"]["skills"] = [
        *doc["tracks"]["sensorimotor"]["skills"],
        "pick_and_place",
    ]
    errors = validate_spec(twice)
    assert any("twice" in e for e in errors) or any("partition" in e for e in errors)


def test_a_field_naming_a_skill_that_does_not_exist_is_refused(spec):
    doc = _doc(spec)
    doc["tracks"]["sensorimotor"]["skills"] = [
        *doc["tracks"]["sensorimotor"]["skills"][:-1],
        "nope",
    ]
    assert any("skills.nope exists" in e for e in validate_spec(doc))


def test_same_scene_may_not_claim_a_disjoint_prompt(spec):
    """The two fields close the replay shortcut by different mechanisms, and confusing them is
    the single most consequential mistake the contract can encode."""
    doc = _doc(spec)
    doc["tracks"]["sensorimotor"]["protocol"] = "same_initial_state"
    assert any("prompt_instance_disjoint matches protocol" in e for e in validate_spec(doc))


def test_a_field_must_name_a_known_protocol_view_and_prompt_source(spec):
    for key, value in (("protocol", "whenever"), ("prompts", "magic")):
        doc = _doc(spec)
        doc["tracks"]["sensorimotor"][key] = value
        assert any(f"tracks.sensorimotor.{key}" in e for e in validate_spec(doc))
    doc = _doc(spec)
    doc["tracks"]["sensorimotor"]["demonstration"]["view"] = "telepathy"
    assert any("demonstration.view" in e for e in validate_spec(doc))


def test_a_demonstration_always_carries_video(spec):
    doc = _doc(spec)
    doc["tracks"]["sensorimotor"]["demonstration"]["modalities"] = ["actions"]
    assert any("modalities has video" in e for e in validate_spec(doc))


def test_a_field_overriding_sizes_must_keep_the_same_names(spec):
    doc = _doc(spec)
    doc["tracks"]["sensorimotor"]["sizes"] = {"tiny": {"units_per_skill": 1}}
    assert any("sizes names match" in e for e in validate_spec(doc))


def test_a_skill_whose_benchmark_is_undeclared_is_refused(spec):
    doc = _doc(spec)
    doc["benchmarks"].pop("draw")
    assert any("benchmarks.draw undeclared" in e for e in validate_spec(doc))


def test_two_fields_may_not_share_a_slug_or_a_code(spec):
    doc = _doc(spec)
    first = doc["tracks"]["sensorimotor"]
    second = json.loads(json.dumps(first))
    second["id"] = "second"
    second["skills"] = []
    doc["tracks"]["second"] = second
    doc["baselines"]["second"] = None
    doc["pools"]["tracks"]["second"] = {"version": None, "pool_id": None}
    errors = validate_spec(doc)
    assert any("slug unique" in e for e in errors)
    assert any("code unique" in e for e in errors)


def test_a_field_carries_its_own_duelling_constants(spec, tmp_path):
    """A RoboTwin unit and a LIBERO unit are not the same amount of work."""
    doc = _doc(spec)
    doc["tracks"]["sensorimotor"]["max_void_fraction"] = 0.25
    doc["tracks"]["sensorimotor"]["score_margin"] = 7.5
    assert validate_spec(doc) == []
    loaded = load_spec_file(_save(tmp_path, doc))
    assert loaded.max_void_fraction("sensorimotor") == 0.25
    assert loaded.score_margin("sensorimotor") == 7.5


def test_a_field_without_overrides_reads_the_competition_defaults(spec):
    assert spec.score_margin("sensorimotor") == spec.duel["score_margin"]
    assert spec.max_void_fraction("sensorimotor") == spec.duel["max_void_fraction"]


def test_sole_track_refuses_to_guess_once_a_second_field_exists(spec, tmp_path):
    doc = _two_fields(spec)
    loaded = load_spec_file(_save(tmp_path, doc))
    assert loaded.tracks == ("sensorimotor", "second")
    with pytest.raises(ValueError, match="still assumes one track"):
        _ = loaded.sole_track


def test_track_of_finds_the_field_a_skill_is_scored_in(spec):
    for skill in spec.all_skills:
        assert spec.track_of(skill) == "sensorimotor"
    with pytest.raises(KeyError):
        spec.track_of("no_such_skill")
