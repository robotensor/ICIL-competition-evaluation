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


def test_the_listing_is_rebuilt_from_disk_and_repeats_only_what_is_signed(tmp_path):
    """Navigation, not provenance.

    A reader needs some way to find an exhibit at all, and a directory cannot be listed over
    HTTP. The listing carries no claim that is not also in the signed document it points at,
    and it is rebuilt rather than appended to, so it cannot drift from what the store holds.
    """
    signer = Signer.generate()
    first = {**GOOD, "benchmark": {"name": "robotwin-icil", "simulator": "robotwin"}}
    second = {
        **first,
        "reference_id": "later-run",
        "published_at": "2026-09-13T00:00:00Z",
    }
    reference.write(tmp_path, reference.exhibit(**first), signer)
    reference.write(tmp_path, reference.exhibit(**second), signer)
    path = reference.write_listing(tmp_path)

    doc = json.loads(path.read_text())
    assert [item["reference_id"] for item in doc["references"]] == [
        "later-run",
        "bpp-robotwin-same-scene",
    ], "newest first"
    assert doc["references"][0]["benchmark"]["simulator"] == "robotwin"
    assert all("track" not in item for item in doc["references"]), "an exhibit is under no field"

    # It is rebuilt, not appended: an exhibit removed from disk leaves the listing.
    (tmp_path / reference.ROOT / "later-run.json").unlink()
    reference.write_listing(tmp_path)
    doc = json.loads(path.read_text())
    assert [item["reference_id"] for item in doc["references"]] == ["bpp-robotwin-same-scene"]
