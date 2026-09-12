"""The architectural half of a field's demonstration view.

`demoview` keeps the withheld arrays out of the mapping a policy is handed. That is a code path,
and a code path can be rewritten. This is the half that survives that: an architecture used by a
field that withholds a channel must have no input for it, and the fingerprint pins the template
so a submission cannot flip the switch back on.
"""

from __future__ import annotations

import json

from icilval import arch


def _cfg(**flags):
    return {
        "architecture": "x",
        "model": {"obs_encoder": {"prompt_tokenizer": {**flags}}},
    }


def test_the_switches_are_found_wherever_they_are_nested():
    cfg = _cfg(ignore_prompt_action=True, ignore_prompt_proprio=False)
    assert arch.switches(cfg) == {"ignore_prompt_action": True, "ignore_prompt_proprio": False}


def test_a_field_that_withholds_nothing_constrains_no_architecture():
    assert arch.check_view(_cfg(ignore_prompt_action=False), ()) == []


def test_an_architecture_that_would_still_read_the_actions_is_refused():
    problems = arch.check_view(_cfg(ignore_prompt_action=False), ("actions",))
    assert problems == ["actions: withheld by the field, but ignore_prompt_action is false"]


def test_an_architecture_that_ignores_the_withheld_channel_passes():
    cfg = _cfg(ignore_prompt_action=True, ignore_prompt_proprio=True)
    assert arch.check_view(cfg, ("actions", "proprio")) == []


def test_a_template_with_no_such_switch_is_refused_rather_than_assumed():
    """Absent is not the same as ignored: an architecture with no switch may read the channel."""
    assert arch.check_view({"model": {}}, ("actions",)) == [
        "actions: the template has no ignore_prompt_action"
    ]


def test_one_copy_of_a_switch_left_on_is_enough_to_refuse():
    """Any tokenizer that still reads the channel reads it, however many others do not."""
    cfg = {
        "model": {
            "a": {"ignore_prompt_action": True},
            "b": {"ignore_prompt_action": False},
        }
    }
    assert arch.check_view(cfg, ("actions",))


def test_the_shipped_sensorimotor_architectures_are_not_video_only_architectures(spec):
    """The real templates, against a hypothetical video-only field: both read the prompt actions,
    so neither could serve one. That is the check doing its job, not a defect."""
    for name in ("bpp_libero_v1", "bpp_draw_v1"):
        cfg = json.loads(arch.template_path("arch", name).read_text())
        assert arch.check_view(cfg, ("actions", "proprio"))


def test_the_sensorimotor_field_passes_its_own_check(spec):
    """Its architectures exist and it withholds nothing, so there is nothing to refuse."""
    problems = arch.check_spec(spec, "arch")
    assert not [p for p in problems if p.startswith("sensorimotor")]


def test_the_video_only_field_says_its_template_is_missing(spec):
    """Declared but not open: `uniskill_v1` cannot be emitted until a checkpoint exists, so the
    deploy check reports it rather than letting a submission fail later."""
    problems = arch.check_spec(spec, "arch")
    assert any("uniskill_v1 has no template" in p for p in problems)


def test_a_declared_architecture_with_no_template_is_reported(spec, tmp_path):
    assert any("no template" in p for p in arch.check_spec(spec, tmp_path))
