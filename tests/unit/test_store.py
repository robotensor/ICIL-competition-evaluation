import json

from icilval.canon import Signer
from icilval.ids import ModelRef
from icilval.spec import load_spec_file
from icilval.store.records import (
    duel_event,
    empty_skill_scores,
    index_record,
    now_iso,
    unit_verdict_from_unit,
)
from icilval.store.verify import verify_store
from icilval.store.writer import Store


def small_spec(spec, tmp_path, lines_per_part=2):
    doc = json.loads(json.dumps(spec.raw))
    doc["store"]["index_lines_per_part"] = lines_per_part
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(doc))
    return load_spec_file(p)


def make_record(
    spec, kind, block, king, challenger, dethroned=False, event=None, track="sensorimotor"
):
    eid = "%064x" % (block + 1)
    ks = (
        {**empty_skill_scores(spec.all_skills), "pick_and_place": 0.5, "average": 0.5}
        if kind == "duel"
        else None
    )
    cs = (
        {**empty_skill_scores(spec.all_skills), "pick_and_place": 0.9, "average": 0.9}
        if kind == "duel"
        else None
    )
    return index_record(
        schema=spec.store["schema"],
        event_id=eid,
        kind=kind,
        track=track,
        block=block,
        finished_at=now_iso(),
        king=king,
        challenger=challenger,
        king_scores=ks,
        challenger_scores=cs,
        score_margin=spec.score_margin(track),
        dethroned=dethroned,
        new_king=challenger if dethroned else None,
        duel_size="smoke",
    )


def publish(store, spec, record, units=None, media_count=0, track="sensorimotor"):
    event = duel_event(
        record,
        spec_version=spec.version,
        spec_fingerprint=spec.fingerprint,
        units=units or [],
        units_per_skill=spec.units_per_skill(track, "smoke"),
        started_at=now_iso(),
        wall_seconds=1.5,
    )
    store.write_event(track, event)
    return store.append(track, record)


def test_store_append_rotate_head_and_verify(spec, tmp_path):
    sp = small_spec(spec, tmp_path)
    signer = Signer.generate()
    store = Store(tmp_path / "store", sp, signer)
    store.init(signer.verify_key_hex, None)
    king = ModelRef.make("org/genesis", "a" * 40)
    ch = ModelRef.make("org/ch", "b" * 40)
    track = "sensorimotor"
    assert publish(store, sp, make_record(sp, "genesis", 0, king, None)) == 1
    assert store.head(track)["king"]["key"] == king.key
    assert publish(store, sp, make_record(sp, "duel", 1, king, ch, dethroned=False)) == 2
    assert store.head(track)["king"]["key"] == king.key
    assert publish(store, sp, make_record(sp, "duel", 2, king, ch, dethroned=True)) == 3
    assert store.head(track)["king"]["key"] == ch.key
    assert store.index_part_path(track, 0).exists() and store.index_part_path(track, 1).exists()
    assert len(store.index_part_path(track, 0).read_text().strip().split("\n")) == 2
    records = store.iter_index(track)
    assert [r["seq"] for r in records] == [1, 2, 3]
    report = verify_store(tmp_path / "store", sp)
    assert report.ok, report.errors
    assert report.records == 3 and report.events == 3

    # tamper: flip a byte in a record body -> signature fails
    p = store.index_part_path(track, 0)
    lines = p.read_text().split("\n")
    lines[0] = lines[0].replace('"dethroned":false', '"dethroned":true', 1)
    p.write_text("\n".join(lines))
    bad = verify_store(tmp_path / "store", sp)
    assert any("bad signature" in e for e in bad.errors)


def test_media_and_torn_line(spec, tmp_path):
    sp = small_spec(spec, tmp_path, lines_per_part=1000)
    signer = Signer.generate()
    store = Store(tmp_path / "store", sp, signer)
    store.init(signer.verify_key_hex, "f" * 64)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"x" * 100)
    sha = store.put_media(clip)
    assert store.has_media(sha, "mp4") and store.media_path(sha, "mp4").parent.name == sha[:2]
    assert store.put_media(clip) == sha
    unit = unit_verdict_from_unit(
        {
            "unit_id": "pp-000",
            "skill": "pick_and_place",
            "index": 0,
            "task": "t",
            "instance": 1,
            "seed": 5,
            "instance_params": {},
            "demo": "t/demo_00",
        }
    )
    unit.update(
        {
            "demo_video": sha,
            "king_video": sha,
            "challenger_video": None,
            "king_success": True,
            "challenger_success": False,
            "outcome": "king",
            "king_metric": 7.25,
        }
    )
    king = ModelRef.make("org/genesis", "a" * 40)
    ch = ModelRef.make("org/ch", "b" * 40)
    publish(store, sp, make_record(sp, "duel", 1, king, ch), units=[unit])
    assert verify_store(tmp_path / "store", sp).ok
    # torn final line is skipped by readers
    p = store.index_part_path("sensorimotor", 0)
    with open(p, "a") as fh:
        fh.write('{"seq":2,"event_id":"ab')
    assert [r["seq"] for r in store.iter_index("sensorimotor")] == [1]
    # missing media is an error
    store.media_path(sha, "mp4").unlink()
    assert any("media" in e for e in verify_store(tmp_path / "store", sp).errors)


def test_schema_rejects_axis_scores(spec, tmp_path):
    sp = small_spec(spec, tmp_path, lines_per_part=1000)
    signer = Signer.generate()
    store = Store(tmp_path / "store", sp, signer)
    store.init(signer.verify_key_hex, None)
    king = ModelRef.make("org/genesis", "a" * 40)
    ch = ModelRef.make("org/ch", "b" * 40)
    rec = make_record(sp, "duel", 1, king, ch)
    rec["king_scores"] = {"spatial": 0.5, "average": 1.5}
    publish(store, sp, rec)
    report = verify_store(tmp_path / "store", sp)
    assert any("schema" in e for e in report.errors)


def test_store_lock_is_exclusive(tmp_path):
    import pytest

    from icilval.store.writer import store_lock

    with store_lock(tmp_path / "s"):
        with pytest.raises(RuntimeError):
            with store_lock(tmp_path / "s"):
                pass
    with store_lock(tmp_path / "s"):
        pass


def test_mirror_lists_only_the_store_files(spec, tmp_path):
    """The lock and anything else dotted is the store's own bookkeeping, not the record."""
    from icilval.store.mirror import store_files
    from icilval.store.writer import store_lock

    signer = Signer.generate()
    store = Store(tmp_path / "store", spec, signer)
    store.init(signer.verify_key_hex, None)
    with store_lock(tmp_path / "store"):
        pass
    assert (tmp_path / "store" / ".validator.lock").exists()
    files = store_files(tmp_path / "store")
    assert "manifest.json" in files
    assert "tracks/sensorimotor/head.json" in files
    assert not any(f.startswith(".") or "/." in f for f in files)


def test_prune_keeps_the_repository_own_files():
    """A rebuilt store replaces the records; it does not take the dataset card with it."""
    from icilval.store.mirror import REPO_OWNED

    remote = {"README.md", ".gitattributes", "manifest.json", "events/t/old.json", "media/aa/x.mp4"}
    local = {"manifest.json", "events/t/new.json"}
    assert sorted(remote - local - REPO_OWNED) == ["events/t/old.json", "media/aa/x.mp4"]


def test_two_fields_are_verified_independently(spec, tmp_path):
    """Each field has its own index, head and sequence, and `store verify` walks every one the
    manifest names. A duel in one leaves the other's lineage untouched."""
    sp = small_spec(spec, tmp_path)
    signer = Signer.generate()
    store = Store(tmp_path / "store", sp, signer)
    manifest = store.init(signer.verify_key_hex, None)
    assert manifest["tracks"] == list(sp.tracks)

    king = ModelRef.make("org/genesis", "a" * 40)
    other = ModelRef.make("org/vo", "c" * 40)
    assert publish(store, sp, make_record(sp, "genesis", 0, king, None)) == 1
    assert (
        publish(
            store,
            sp,
            make_record(sp, "genesis", 0, other, None, track="video_only"),
            track="video_only",
        )
        == 1
    )

    # Separate sequences, separate kings.
    assert store.head("sensorimotor")["king"]["key"] == king.key
    assert store.head("video_only")["king"]["key"] == other.key
    assert [r["seq"] for r in store.iter_index("sensorimotor")] == [1]
    assert [r["seq"] for r in store.iter_index("video_only")] == [1]

    report = verify_store(tmp_path / "store", sp)
    assert report.ok, report.errors
    assert report.records == 2 and report.events == 2


def test_a_foreign_kind_carrying_a_king_does_not_take_the_crown(spec, tmp_path):
    """Crowning is an allow-list.

    The head is what every reader treats as "who holds this field's crown". A record of some
    other kind that happens to carry `king`/`new_king` - a kind added later, or a tool reusing
    the record shape - must not move it. Before the allow-list, `new_king` alone was enough.
    """
    sp = small_spec(spec, tmp_path, lines_per_part=1000)
    signer = Signer.generate()
    store = Store(tmp_path / "store", sp, signer)
    store.init(signer.verify_key_hex, None)
    king = ModelRef.make("org/genesis", "a" * 40)
    usurper = ModelRef.make("org/usurper", "c" * 40)
    track = "sensorimotor"
    publish(store, sp, make_record(sp, "genesis", 0, king, None))
    assert store.head(track)["king"]["key"] == king.key

    foreign = make_record(sp, "duel", 1, king, usurper, dethroned=True)
    foreign["kind"] = "something_else"
    store.append(track, foreign)

    head = store.head(track)
    assert head["king"]["key"] == king.key, "a non-crowning kind moved the crown"
    assert head["seq"] == 2, "it is still recorded; only the crown is left alone"
