"""A duel for a field that makes its own prompts instead of drawing them from a pool.

Under `prompts: "materialized"` there is no pool at all: the orchestrator asks the benchmark for
its units, has it produce each prompt on the validator host before either side runs, and publishes
the prompts with the event. These are the three places that assumed a pool.
"""

from __future__ import annotations

from icilval.duel.orchestrate import _materialized_clip


def test_the_benchmarks_own_demonstration_clip_is_published(tmp_path):
    """`materialize_command` already wrote it; re-encoding would be a second chance to get the
    frame layout wrong, and would need arrays the orchestrator has no reason to hold."""
    prompt_dir = tmp_path / "pp-000"
    prompt_dir.mkdir()
    clip = prompt_dir / "demonstration.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"x" * 64)

    found = _materialized_clip({"prompt_dir": str(prompt_dir)}, "mp4")
    assert found == clip


def test_a_unit_with_no_materialized_prompt_falls_back_to_the_pool(tmp_path):
    """An in-repo benchmark's unit has no prompt directory, and must still render."""
    assert _materialized_clip({}, "mp4") is None
    assert _materialized_clip({"prompt_dir": str(tmp_path / "missing")}, "mp4") is None


def test_an_empty_clip_is_not_published_as_one(tmp_path):
    prompt_dir = tmp_path / "pp-000"
    prompt_dir.mkdir()
    (prompt_dir / "demonstration.mp4").write_bytes(b"")
    assert _materialized_clip({"prompt_dir": str(prompt_dir)}, "mp4") is None


def test_a_plugin_unit_carries_everything_the_published_record_asks_for():
    """The record's shape is the same for every field, so a plugin unit answers the same
    questions a pool unit does - and the answers have to be true, not placeholders."""
    from icilval.benchmarks import units as plugin_units
    from icilval.store.records import unit_verdict_from_unit

    class Plugin:
        id = "fakesim"
        api_version = 1

        def info(self):
            return {"id": self.id, "api_version": 1}

        def derive_units(self, *, seed_material, count, suite):
            return [{"task": "place_a2b_left", "scene_seed": 191664963} for _ in range(count)]

    class Sim:
        name = "fakesim"
        benchmark = Plugin()

    class Spec:
        def skills(self, track):
            return ["rt_sm_pick_and_place"]

        def units_per_skill(self, track, size=None):
            return 1

        def default_size(self, track):
            return "smoke"

        def skill(self, name):
            return {"code": "mp", "tasks": {"suite": "v1"}}

        def skill_code(self, name):
            return "mp"

    import icilval.benchmarks.units as mod

    original = mod.for_skill
    mod.for_skill = lambda _s, _k: Sim()
    try:
        units = plugin_units.plugin_units(Spec(), "sensorimotor", "duel1", "smoke")
    finally:
        mod.for_skill = original

    verdict = unit_verdict_from_unit(units[0], "sensorimotor")
    assert verdict["unit_id"] == "mp-000"
    assert verdict["task"] == "place_a2b_left"
    # One initial state, because this field scores from the state it demonstrated.
    assert verdict["instance"] == 0
    # The prompt made for this unit *is* its demonstration, so the unit names itself.
    assert verdict["prompt"]["demo_id"] == "mp-000"
    # Enough to rebuild the scene is carried, without the orchestrator knowing what it means.
    assert verdict["instance_params"]["scene_seed"] == 191664963
