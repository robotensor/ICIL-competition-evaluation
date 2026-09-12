"""An exhibit published beside the competition, outside every ladder.

What matters about this format is what it refuses to let you say, so that is what these check.
"""

from __future__ import annotations

import json

import pytest

from icilval import reference
from icilval.canon import Signer

GOOD = dict(
    reference_id="bpp-robotwin-same-scene",
    headline="Benchmark reference evaluation - not a competition score.",
    not_a_competition_score=(
        "The policy was shown the demonstration's actions and proprioception. The video-only "
        "field withholds both."
    ),
    benchmark={"name": "robotwin-icil"},
    protocol={"evaluation_setting": "same_scene"},
    demonstration_shown={"view": "sensorimotor", "channels": ["video", "actions", "proprio"]},
    subject={"label": "bpp", "admissible": False},
    results={"scored": 1, "successes": 0},
    published_at="2026-09-12T00:00:00Z",
)


def test_an_exhibit_is_on_no_ladder_and_under_no_field():
    """Not parameters: making them settable would invite the one mistake this format prevents."""
    doc = reference.exhibit(**GOOD)
    assert doc["ladder"] is False
    assert doc["track"] is None
    assert doc["kind"] == reference.KIND


def test_it_must_say_what_the_policy_was_shown():
    bad = {**GOOD, "demonstration_shown": {"channels": ["video"]}}
    with pytest.raises(reference.ReferenceError, match="was shown"):
        reference.exhibit(**bad)


def test_it_must_carry_the_sentence_a_reader_sees_first():
    for field in ("headline", "not_a_competition_score"):
        with pytest.raises(reference.ReferenceError):
            reference.exhibit(**{**GOOD, field: "   "})


def test_the_id_is_a_slug_so_it_cannot_escape_its_directory():
    for bad_id in ("../../tracks/video_only/head", "Has Spaces", "a", ""):
        with pytest.raises(reference.ReferenceError, match="slug"):
            reference.exhibit(**{**GOOD, "reference_id": bad_id})


def test_it_is_signed_and_reads_back(tmp_path):
    signer = Signer.generate()
    doc = reference.exhibit(**GOOD)
    path = reference.write(tmp_path, doc, signer)
    assert path.parent.name == reference.ROOT
    line = path.read_text().strip()
    body, sig = line.split("\t", 1)
    assert len(sig) == 128
    assert json.loads(body) == doc
    assert reference.read(path) == doc


def test_it_lands_beside_the_ladders_never_inside_one(tmp_path):
    """`tracks/<field>/index-*.jsonl` is a claim that the validator ran it for that crown."""
    reference.write(tmp_path, reference.exhibit(**GOOD), Signer.generate())
    assert (tmp_path / "references").is_dir()
    assert not (tmp_path / "tracks").exists()
