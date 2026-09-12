"""Two fields, side by side.

The contract ships one field today, so these build a two-field spec from the real one and check
the things that must hold once the second arrives: separate queues, separate heads, separate
lineages, and a benchmark missing from one field not stopping the other.
"""

from __future__ import annotations

import json

import pytest

from icilval import simulators
from icilval.queue import Queues
from icilval.spec import load_spec_file


@pytest.fixture
def two_fields(spec, tmp_path):
    """The real contract with `goal_chain` moved into a second field of its own."""
    doc = json.loads(spec.path.read_text())
    first = doc["tracks"]["sensorimotor"]
    first["skills"] = [s for s in first["skills"] if s != "goal_chain"]
    second = json.loads(json.dumps(first))
    second.update(
        id="video_only",
        code="vo",
        slug="video-only",
        short="Video-only",
        title="Video-only demonstration",
        skills=["goal_chain"],
        protocol="same_initial_state",
        prompt_instance_disjoint=False,
        prompts="materialized",
        default_size="smoke",
        sizes={k: {"units_per_skill": 1} for k in doc["duel"]["sizes"]},
        max_void_fraction=0.2,
    )
    second["demonstration"] = {
        "view": "video_only",
        "modalities": ["video"],
        "withheld": ["actions", "qpos", "endpose"],
    }
    doc["tracks"]["video_only"] = second
    doc["baselines"]["video_only"] = None
    doc["pools"]["tracks"]["video_only"] = {"version": None, "pool_id": None}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(doc))
    return load_spec_file(path)


def test_two_fields_load_and_keep_their_own_skills(two_fields):
    spec = two_fields
    assert spec.tracks == ("sensorimotor", "video_only")
    assert spec.skills("sensorimotor") == ("pick_and_place", "draw_anything")
    assert spec.skills("video_only") == ("goal_chain",)
    assert set(spec.all_skills) == set(spec.skills("sensorimotor")) | set(spec.skills("video_only"))


def test_each_field_carries_its_own_demonstration_view(two_fields):
    assert two_fields.demo_view("sensorimotor") == "sensorimotor"
    assert two_fields.withheld("sensorimotor") == ()
    assert two_fields.demo_view("video_only") == "video_only"
    assert "actions" in two_fields.withheld("video_only")


def test_the_two_fields_close_the_replay_shortcut_differently(two_fields):
    """Sensorimotor scores a state it did not demonstrate; video-only demonstrates the state it
    scores and withholds the actions instead. Neither leaves the shortcut open."""
    spec = two_fields
    assert spec.protocol("sensorimotor") == "different_initial_state"
    assert spec.prompt_instance_disjoint("sensorimotor") is True
    assert spec.withheld("sensorimotor") == ()

    assert spec.protocol("video_only") == "same_initial_state"
    assert spec.prompt_instance_disjoint("video_only") is False
    assert "actions" in spec.withheld("video_only")


def test_a_field_overrides_the_duelling_constants_it_needs_to(two_fields):
    spec = two_fields
    assert spec.units_per_skill("video_only", "heavy") == 1
    assert spec.units_per_skill("sensorimotor", "heavy") > 1
    assert spec.max_void_fraction("video_only") == 0.2
    assert spec.max_void_fraction("sensorimotor") == spec.duel["max_void_fraction"]
    # Both fields still show a submitter one vocabulary of sizes.
    assert set(spec.sizes("video_only")) == set(spec.sizes("sensorimotor"))


def test_units_per_side_counts_the_field_not_the_contract(two_fields):
    spec = two_fields
    assert spec.units_per_side("video_only", "smoke") == spec.units_per_skill("video_only", "smoke")
    assert spec.units_per_side("sensorimotor", "smoke") == 2 * spec.units_per_skill(
        "sensorimotor", "smoke"
    )


def test_sole_track_will_not_guess_between_them(two_fields):
    with pytest.raises(ValueError, match="still assumes one track"):
        _ = two_fields.sole_track


def test_each_field_has_its_own_queue(two_fields, tmp_path):
    queues = Queues(tmp_path / "queue", two_fields.tracks)
    assert set(queues.tracks) == {"sensorimotor", "video_only"}

    queues["sensorimotor"].add("org/one", "a" * 40)
    assert [e.repo for e in queues["sensorimotor"].entries()] == ["org/one"]
    assert queues["video_only"].entries() == []

    # A block counter belongs to a lineage, so advancing one leaves the other alone.
    assert queues["sensorimotor"].advance_block() == 1
    assert queues["video_only"].block == 0

    reopened = Queues(tmp_path / "queue", two_fields.tracks)
    assert [e.repo for e in reopened["sensorimotor"].entries()] == ["org/one"]


def test_an_unknown_field_names_the_ones_there_are(two_fields, tmp_path):
    queues = Queues(tmp_path / "queue", two_fields.tracks)
    with pytest.raises(KeyError, match="video_only"):
        _ = queues["nope"]


def test_the_old_single_queue_file_says_what_to_do(two_fields, tmp_path):
    legacy = tmp_path / "queue.json"
    legacy.write_text("{}")
    with pytest.raises(ValueError, match="one file per field"):
        Queues(legacy, two_fields.tracks)


def test_a_store_gives_every_field_a_head(two_fields, tmp_path):
    from icilval.canon import Signer
    from icilval.store.writer import Store

    store = Store(tmp_path / "store", two_fields, Signer.generate())
    manifest = store.init("k" * 64, pool_id=None)
    assert manifest["tracks"] == ["sensorimotor", "video_only"]
    for track in two_fields.tracks:
        assert store.head(track) is not None and store.head(track)["king"] is None


def test_a_field_whose_benchmark_is_absent_does_not_stop_the_other(two_fields):
    """`require` is per field, so a missing plugin skips that queue and no other."""
    simulators.require(two_fields, two_fields.skills("sensorimotor"))
    simulators.require(two_fields, two_fields.skills("video_only"))
