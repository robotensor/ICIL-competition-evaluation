"""What a field's policies may see of a demonstration.

The test that matters is the last one: it drives the real side runner over a field declaring the
video-only view, with a fake benchmark that records whatever it is handed, and checks no action
array was in it. Everything above it checks the pieces that test depends on.
"""

from __future__ import annotations

import json

import numpy as np

from icilval import demoview
from icilval.demoview import DemoView

CHANNELS = {
    "video": ("agentview", "eye_in_hand"),
    "proprio": ("ee_pos", "gripper"),
    "actions": ("actions",),
}

SENSORIMOTOR = DemoView("sensorimotor", keep=("video", "proprio", "actions"), withheld=())
VIDEO_ONLY = DemoView("video_only", keep=("video",), withheld=("actions", "proprio"))


def _demo():
    return {
        "agentview": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        "eye_in_hand": np.ones((2, 2, 3), dtype=np.uint8),
        "ee_pos": np.arange(6, dtype=np.float32).reshape(2, 3),
        "gripper": np.zeros((2, 1), dtype=np.float32),
        "actions": np.arange(14, dtype=np.float32).reshape(2, 7),
        "init_state": np.arange(5, dtype=np.float64),
        "meta": {"demo_id": "t/demo_00", "steps": 2},
    }


def test_the_sensorimotor_view_keeps_every_channel():
    out = demoview.apply(_demo(), CHANNELS, SENSORIMOTOR)
    assert set(out) == {"agentview", "eye_in_hand", "ee_pos", "gripper", "actions", "meta"}


def test_the_video_only_view_keeps_the_frames_and_nothing_else():
    out = demoview.apply(_demo(), CHANNELS, VIDEO_ONLY)
    assert set(out) == {"agentview", "eye_in_hand", "meta"}
    assert "actions" not in out and "ee_pos" not in out


def test_an_array_belonging_to_no_channel_is_never_handed_over():
    """`init_state` is the scene the demonstration started from. It is in no channel, so it
    reaches no policy under any view - an allow-list, not a deny-list."""
    for view in (SENSORIMOTOR, VIDEO_ONLY):
        assert "init_state" not in demoview.apply(_demo(), CHANNELS, view)


def test_a_new_array_does_not_leak_into_a_restricted_view():
    """The property that makes this survive a benchmark changing under it."""
    demo = _demo()
    demo["tactile"] = np.ones((2, 4), dtype=np.float32)
    assert "tactile" not in demoview.apply(demo, CHANNELS, VIDEO_ONLY)
    assert "tactile" not in demoview.apply(demo, CHANNELS, SENSORIMOTOR)


def test_the_view_is_recorded_on_the_demonstration_it_produced():
    out = demoview.apply(_demo(), CHANNELS, VIDEO_ONLY)
    assert out["meta"]["view"] == "video_only"


def test_check_finds_anything_that_should_not_have_got_through():
    leaked = {**demoview.apply(_demo(), CHANNELS, VIDEO_ONLY), "actions": np.zeros(3)}
    problems = demoview.check(leaked, CHANNELS, VIDEO_ONLY)
    assert problems and "actions" in problems[0]
    assert demoview.check(demoview.apply(_demo(), CHANNELS, VIDEO_ONLY), CHANNELS, VIDEO_ONLY) == []


def test_the_digest_covers_exactly_what_was_handed_over():
    full = demoview.apply(_demo(), CHANNELS, SENSORIMOTOR)
    video = demoview.apply(_demo(), CHANNELS, VIDEO_ONLY)
    assert demoview.handed_sha256(full) != demoview.handed_sha256(video)
    assert len(demoview.handed_sha256(video)) == 64


def test_the_digest_is_stable_and_order_independent():
    video = demoview.apply(_demo(), CHANNELS, VIDEO_ONLY)
    shuffled = dict(reversed(list(video.items())))
    assert demoview.handed_sha256(video) == demoview.handed_sha256(shuffled)
    assert demoview.handed_sha256(video) == demoview.handed_sha256(
        demoview.apply(_demo(), CHANNELS, VIDEO_ONLY)
    )


def test_the_digest_changes_when_a_kept_array_changes():
    video = demoview.apply(_demo(), CHANNELS, VIDEO_ONLY)
    other = dict(video)
    other["agentview"] = other["agentview"] + 1
    assert demoview.handed_sha256(video) != demoview.handed_sha256(other)


def test_the_digest_ignores_metadata():
    video = demoview.apply(_demo(), CHANNELS, VIDEO_ONLY)
    assert demoview.handed_sha256(video) == demoview.handed_sha256({**video, "meta": {"x": 1}})


def test_a_view_is_read_off_the_field(spec):
    view = demoview.view_for(spec, "sensorimotor")
    assert view.name == "sensorimotor" and not view.is_restrictive
    assert set(view.keep) == {"video", "actions", "proprio"}


# ------------------------------------------------------------------ the enforcement test


def test_a_video_only_field_hands_its_policies_no_actions(spec, tmp_path, monkeypatch):
    """Drive the real side runner with a fake benchmark that records what it was handed."""
    from icilval import simulators
    from icilval.duel.side_runner import run_side
    from icilval.spec import load_spec_file

    # The real video-only field, with its benchmark swapped for a fake: the field's declaration
    # is the contract under test, and the benchmark it names lives in another repository.
    doc = json.loads(spec.path.read_text())
    doc["skills"]["rt_pick_and_place"]["simulator"] = "fakesim"
    doc["benchmarks"]["fakesim"] = {"distribution": "icilval", "in_repo": True}
    doc["tracks"]["video_only"]["skills"] = ["rt_pick_and_place"]
    for gone in ("rt_stacking", "rt_press_push"):
        doc["skills"].pop(gone)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(doc))

    handed: list[dict] = []

    def run_units(ctx, skill, policy, pool, units, spec_, media_dir, record_video):
        for unit in units:
            handed.append(ctx.demo(unit))
            ctx.finish(unit, ctx.record(unit, _Result(), media_dir / "x.mp4"), None)

    class _Result:
        void, success, progress, metric, first_step_done_at = False, True, None, None, None
        steps, model_errors, wall_s, error = 1, 0, 0.1, None
        prompt_steps, prompt_chunks, instance_applied = 2, 1, {}

    monkeypatch.setitem(
        simulators.REGISTRY,
        "fakesim",
        simulators.Simulator(
            name="fakesim",
            make_policy=lambda *a, **k: _Policy(),
            run_units=run_units,
            build_stage=lambda *a, **k: None,
            make_unit=lambda *a, **k: None,
            demo_frames=lambda demo: [],
            validate_skill=lambda skill, doc_: [],
            demo_channels=CHANNELS,
        ),
    )

    demo_dir = tmp_path / "pool" / "demos" / "task0"
    demo_dir.mkdir(parents=True)
    arrays = {k: v for k, v in _demo().items() if k != "meta"}
    np.savez_compressed(demo_dir / "demo_00.npz", meta=json.dumps({"demo_id": "d"}), **arrays)

    class _Pool:
        def path(self, rel):
            return tmp_path / "pool" / rel

    units = [
        {
            "unit_id": "rp-0",
            "skill": "rt_pick_and_place",
            "demo": "task0/demo_00",
            "index": 0,
            "task": "task0",
            "instance": 0,
            "seed": 1,
        }
    ]
    run_side(
        side="challenger",
        model_dir=tmp_path / "model",
        arch_dir=tmp_path / "arch",
        pool=_Pool(),
        units=units,
        spec=load_spec_file(path),
        track="video_only",
        out_dir=tmp_path / "out",
        record_video=False,
    )

    assert handed, "the benchmark was never handed a demonstration"
    got = handed[0]
    assert set(got) == {"agentview", "eye_in_hand", "meta"}
    for withheld in ("actions", "ee_pos", "gripper", "init_state"):
        assert withheld not in got
    # And what it was handed is published, so a third party can check the same thing.
    assert units[0]["handed_sha256"] == demoview.handed_sha256(got)


class _Policy:
    load_seconds = 0.0

    def load(self):
        pass

    def unload(self):
        pass


# A benchmark whose video channel is one array per camera, and which carries its own metadata.
# This is RoboTwin's shape, and neither half of it fits an exact-name allow-list.
PREFIXED = {
    "video": ("frames_*",),
    "proprio": ("qpos", "endpose", "gripper_joints"),
    "actions": ("actions",),
    "metadata": ("times",),
}

PREFIXED_DEMO = {
    "frames_head_camera": np.zeros((2, 4, 4, 3), dtype=np.uint8),
    "frames_far_side_camera": np.ones((2, 4, 4, 3), dtype=np.uint8),
    "qpos": np.zeros((2, 14)),
    "endpose": np.zeros((2, 16)),
    "gripper_joints": np.zeros((2, 4)),
    "actions": np.zeros((2, 16)),
    "times": np.array([0.0, 0.05]),
    "meta": {"task": "place_a2b_left"},
}


def test_a_prefixed_video_channel_survives_the_allow_list():
    """One array per camera cannot be enumerated by the orchestrator, which knows no camera list.

    Before prefixes, `frames_head_camera` matched no entry and was dropped - so a video-only
    prompt over this benchmark contained no video at all.
    """
    view = demoview.DemoView(name="video_only", keep=("video",), withheld=("actions", "proprio"))
    out = demoview.apply(PREFIXED_DEMO, PREFIXED, view)

    assert "frames_head_camera" in out and "frames_far_side_camera" in out
    assert out["frames_head_camera"].shape == (2, 4, 4, 3)


def test_the_benchmarks_own_metadata_is_kept_under_every_view():
    """`times` is not an observation and no field's modalities would ever claim it."""
    for name, keep in (
        ("video_only", ("video",)),
        ("sensorimotor", ("video", "actions", "proprio")),
    ):
        view = demoview.DemoView(name=name, keep=keep, withheld=())
        out = demoview.apply(PREFIXED_DEMO, PREFIXED, view)
        assert "times" in out, f"{name} lost the frame timestamps"
        assert "meta" in out


def test_video_only_still_withholds_every_action_and_proprio_array_of_a_prefixed_benchmark():
    """The prefix must widen the video channel only - not the allow-list in general."""
    view = demoview.DemoView(name="video_only", keep=("video",), withheld=("actions", "proprio"))
    out = demoview.apply(PREFIXED_DEMO, PREFIXED, view)

    for withheld in ("actions", "qpos", "endpose", "gripper_joints"):
        assert withheld not in out, f"{withheld} reached a video-only prompt"
    assert demoview.check(out, PREFIXED, view) == []
    # And the full view does hand them over, so the test above is about redaction, not absence.
    full = demoview.DemoView(name="sensorimotor", keep=("video", "actions", "proprio"), withheld=())
    assert set(demoview.apply(PREFIXED_DEMO, PREFIXED, full)) == set(PREFIXED_DEMO)


def test_a_prefix_admits_only_what_actually_carries_it():
    """`frames_*` must not become a licence for anything vaguely similar.

    A prefix widens the allow-list, which is the one direction that can leak, so it matches the
    declared string exactly and nothing near it.
    """
    view = demoview.DemoView(name="video_only", keep=("video",), withheld=("actions",))
    demo = {
        **PREFIXED_DEMO,
        "frame_actions": np.zeros((2, 16)),  # `frame_`, not `frames_`
        "framesactions": np.zeros((2, 16)),  # no separator at all
        "frames_": np.zeros((2, 16)),  # the prefix itself, carrying no camera
    }
    out = demoview.apply(demo, PREFIXED, view)

    assert "frame_actions" not in out, "a shorter prefix was admitted"
    assert "framesactions" not in out, "a name merely starting with the letters was admitted"
    assert "frames_" in out, "the declared prefix admits its own exact string"


def test_the_handed_digest_covers_the_prefixed_arrays():
    view = demoview.DemoView(name="video_only", keep=("video",), withheld=("actions", "proprio"))
    out = demoview.apply(PREFIXED_DEMO, PREFIXED, view)
    first = demoview.handed_sha256(out)

    changed = {**out, "frames_head_camera": np.full((2, 4, 4, 3), 7, dtype=np.uint8)}
    assert demoview.handed_sha256(changed) != first, "a camera's frames are not in the digest"
