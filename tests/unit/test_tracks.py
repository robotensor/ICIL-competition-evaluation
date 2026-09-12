"""Two fields, side by side.

The contract ships one field today, so these build a two-field spec from the real one and check
the things that must hold once the second arrives: separate queues, separate heads, separate
lineages, and a benchmark missing from one field not stopping the other.
"""

from __future__ import annotations

import pytest

from icilval import simulators
from icilval.queue import Queues


@pytest.fixture
def two_fields(spec):
    """The shipped contract, which now declares both fields."""
    return spec


def test_two_fields_load_and_keep_their_own_skills(two_fields):
    spec = two_fields
    assert spec.tracks == ("sensorimotor", "video_only")
    assert spec.skills("sensorimotor") == (
        "rt_sm_pick_and_place",
        "rt_sm_stacking",
        "rt_sm_press_push",
    )
    assert spec.skills("video_only") == ("rt_pick_and_place", "rt_stacking", "rt_press_push")
    # The fields partition the skills: the same benchmark backs both, but no skill is in two.
    assert not set(spec.skills("sensorimotor")) & set(spec.skills("video_only"))
    assert set(spec.all_skills) == set(spec.skills("sensorimotor")) | set(spec.skills("video_only"))


def test_each_field_carries_its_own_demonstration_view(two_fields):
    assert two_fields.demo_view("sensorimotor") == "sensorimotor"
    assert two_fields.withheld("sensorimotor") == ()
    assert two_fields.demo_view("video_only") == "video_only"
    assert set(two_fields.withheld("video_only")) == {"actions", "proprio"}


def test_only_the_video_only_field_closes_the_replay_shortcut(two_fields):
    """This is not the arrangement the two fields were designed with, and the test says so.

    At v5 the fields closed the shortcut in opposite ways: sensorimotor scored a state it had not
    demonstrated, video-only demonstrated the state it scored and withheld the actions. At v6 the
    orchestrator ships no benchmark and RoboTwin - the one that is plugged - implements only
    `same_scene`. So the sensorimotor field is now Same Scene *with* the action trajectory, and
    that combination is degenerate: replaying the demonstration's own actions into the identical
    scene solves the episode, and the benchmark's replay oracle scores 18/18 doing exactly that.

    The test pins it rather than hiding it, so nobody reads a sensorimotor score as if the
    shortcut were still closed, and so restoring a real sensorimotor field is a visible change to
    this file rather than a silent one.
    """
    spec = two_fields
    assert spec.protocol("sensorimotor") == "same_initial_state"
    assert spec.prompt_instance_disjoint("sensorimotor") is False
    assert spec.withheld("sensorimotor") == (), "the actions are shown, which is the degeneracy"

    assert spec.protocol("video_only") == "same_initial_state"
    assert spec.prompt_instance_disjoint("video_only") is False
    assert "actions" in spec.withheld("video_only")
    assert "proprio" in spec.withheld("video_only")


def test_a_field_overrides_the_duelling_constants_it_needs_to(two_fields):
    spec = two_fields
    assert spec.units_per_skill("sensorimotor", "heavy") > 1
    assert spec.units_per_skill("video_only", "heavy") > 1
    assert spec.max_void_fraction("video_only") == 0.2
    assert spec.max_void_fraction("sensorimotor") == 0.2
    # Both fields still show a submitter one vocabulary of sizes.
    assert set(spec.sizes("video_only")) == set(spec.sizes("sensorimotor"))


def test_units_per_side_counts_the_field_not_the_contract(two_fields):
    spec = two_fields
    for track in spec.tracks:
        assert spec.units_per_side(track, "smoke") == len(
            spec.skills(track)
        ) * spec.units_per_skill(track, "smoke")


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


def test_a_field_whose_benchmark_is_absent_names_the_distribution_to_install(two_fields):
    """Every field depends on a plugin now: the orchestrator ships no benchmark of its own.

    So `require` failing is the ordinary state of a fresh checkout, and the message has to say
    what to install - "unknown simulator 'robotwin'" tells an operator nothing.
    """
    with pytest.raises(simulators.MissingBenchmark, match="robotwin-icil-competition"):
        simulators.require(two_fields, two_fields.skills("sensorimotor"))


def test_require_is_per_field_so_one_absence_does_not_stop_the_other():
    """A missing plugin skips that field's queue and no other.

    Both shipped fields happen to run on the same benchmark, so the property is checked against a
    spec with two - otherwise the test would pass for the wrong reason.
    """

    class _Spec:
        raw = {
            "benchmarks": {
                "presentsim": {"distribution": "icilval"},
                "absentsim": {"distribution": "some-benchmark-icil"},
            }
        }
        all_skills = ("here", "gone")

        def simulator(self, skill):
            return "presentsim" if skill == "here" else "absentsim"

    spec = _Spec()
    simulators.REGISTRY["presentsim"] = simulators.Simulator(
        name="presentsim",
        make_policy=lambda *a, **k: None,
        run_units=lambda *a, **k: None,
        build_stage=lambda *a, **k: None,
        make_unit=lambda *a, **k: None,
        demo_frames=lambda demo: [],
        validate_skill=lambda skill, doc: [],
    )
    try:
        simulators.require(spec, ("here",))
        with pytest.raises(simulators.MissingBenchmark, match="some-benchmark-icil"):
            simulators.require(spec, ("gone",))
    finally:
        simulators.REGISTRY.pop("presentsim", None)
