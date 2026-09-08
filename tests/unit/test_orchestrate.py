import json

from icilval.canon import Signer
from icilval.duel.orchestrate import Orchestrator, Runtime
from icilval.ids import ModelRef
from icilval.live import LiveReporter
from icilval.pools.schema import Pool
from icilval.store.records import unit_verdict_from_unit
from icilval.store.writer import Store


def make_rt(spec, tmp_path):
    signer = Signer.generate()
    store = Store(tmp_path / "store", spec, signer)
    store.init(signer.verify_key_hex, None)
    pool = Pool(
        schema=3,
        pool_version="t",
        spec_version=3,
        sources={},
        tasks={},
        skills={s: {"eligible": []} for s in spec.skills},
        root=tmp_path,
    )
    return Runtime(
        spec=spec,
        pool=pool,
        store=store,
        signer=signer,
        arch_dir=tmp_path,
        run_root=tmp_path / "runs",
        live=LiveReporter(spec, None, None),
    )


def unit(skill, i):
    code = {"pick_and_place": "pp", "draw_anything": "da"}[skill]
    return unit_verdict_from_unit(
        {
            "unit_id": f"{code}-{i:03d}",
            "skill": skill,
            "index": i,
            "task": "t",
            "instance": i,
            "seed": i,
            "instance_params": {},
            "demo": "t/demo_00",
        }
    )


def test_merge_and_media_flush(spec, tmp_path):
    rt = make_rt(spec, tmp_path)
    orch = Orchestrator(rt)
    state = {
        "units": [unit("pick_and_place", 0), unit("draw_anything", 0)],
        "media_done": {},
        "recent_media": None,
        "event_id": "e" * 64,
        "kind": "duel",
        "size": "smoke",
        "king": None,
        "challenger": ModelRef.make("a/b", "1" * 40),
        "phase": "evaluating",
        "started_at": "2026-01-01T00:00:00Z",
    }
    orch._merge(
        state,
        "king",
        {
            "unit_id": "pp-000",
            "success": True,
            "progress": 1.0,
            "steps": 50,
            "prompt_steps": 100,
            "prompt_chunks": 5,
        },
    )
    orch._merge(
        state, "challenger", {"unit_id": "pp-000", "success": False, "progress": 0.0, "steps": 400}
    )
    u = state["units"][0]
    assert (
        u["king_success"] is True
        and u["challenger_success"] is False
        and u["outcome"] == "king"
        and u["prompt"]["chunks"] == 5
    )
    orch._merge(state, "king", {"unit_id": "da-000", "success": True, "metric": 6.5, "steps": 120})
    assert state["units"][1]["king_metric"] == 6.5
    orch._merge(state, "king", {"unit_id": "da-000", "void": True, "error": "boom"})
    assert state["units"][1]["void"] and state["units"][1]["king_error"] == "boom"

    side_dir = tmp_path / "runs" / "king"
    (side_dir / "media").mkdir(parents=True)
    clip = side_dir / "media" / "pp-000.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"y" * 50)
    from icilval.canon import sha256_file

    (side_dir / "units.jsonl").write_text(
        json.dumps(
            {
                "unit_id": "pp-000",
                "success": True,
                "video": "media/pp-000.mp4",
                "video_sha256": sha256_file(clip),
            }
        )
        + "\n"
    )
    orch._flush_media(state, "king", side_dir)  # below the flush threshold: nothing copied yet
    assert not state["media_done"]
    orch._flush_media(state, "king", side_dir, force=True)
    sha = state["media_done"][("king", "pp-000")]
    assert rt.store.has_media(sha, "mp4") and state["units"][0]["king_video"] == sha
    assert state["recent_media"]["side"] == "king" and state["recent_media"]["video"] == sha
    assert state["recent_media"]["unit"]["skill"] == "pick_and_place"
    frame = orch._frame(state)
    assert frame["units"][0]["king_video"] == sha
    assert frame["skill_progress"]["king"]["pick_and_place"] == {"done": 1, "total": 1}
    assert frame["skill_progress"]["king"]["draw_anything"] == {
        "done": 1,
        "total": 1,
    }  # void counts
