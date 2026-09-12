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
    assert spec.skills("sensorimotor") == ("pick_and_place", "goal_chain", "draw_anything")
    assert spec.skills("video_only") == ("rt_pick_and_place", "rt_stacking", "rt_press_push")
    assert set(spec.all_skills) == set(spec.skills("sensorimotor")) | set(spec.skills("video_only"))


def test_each_field_carries_its_own_demonstration_view(two_fields):
    assert two_fields.demo_view("sensorimotor") == "sensorimotor"
    assert two_fields.withheld("sensorimotor") == ()
    assert two_fields.demo_view("video_only") == "video_only"
    assert set(two_fields.withheld("video_only")) == {"actions", "proprio"}


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
    assert spec.units_per_skill("video_only", "heavy") < spec.units_per_skill(
        "sensorimotor", "heavy"
    )
    assert spec.units_per_skill("sensorimotor", "heavy") > 1
    assert spec.max_void_fraction("video_only") == 0.2
    assert spec.max_void_fraction("sensorimotor") == spec.duel["max_void_fraction"]
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


def test_require_is_per_field_so_one_absence_does_not_stop_the_other(two_fields):
    """A missing plugin skips that field's queue and no other.

    Whether the video-only field's benchmark is installed depends on the environment - it lives
    in another repository - so this asserts the *shape*: the sensorimotor field never needs it,
    and asking for a field whose benchmark is genuinely absent names the distribution to install.
    """
    simulators.require(two_fields, two_fields.skills("sensorimotor"))

    class _Absent:
        raw = {"benchmarks": {"nosuchsim": {"distribution": "some-benchmark-icil"}}}
        all_skills = ("phantom",)

        def simulator(self, skill):
            return "nosuchsim"

    with pytest.raises(simulators.MissingBenchmark, match="some-benchmark-icil"):
        simulators.require(_Absent(), ("phantom",))
