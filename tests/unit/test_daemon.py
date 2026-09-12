from icilval.daemon import Daemon, DaemonConfig
from icilval.duel.orchestrate import DuelFailed
from icilval.queue import Queues
from tests.unit.test_orchestrate import make_rt


class FakeOrchestrator:
    def __init__(self, rt, outcome):
        self.rt = rt
        self.outcome = outcome
        self.calls = []

    def run(self, req, block):
        self.calls.append((req.challenger.repo, req.king.repo if req.king else None, block))
        if self.outcome == "fail":
            raise DuelFailed("nope")
        from icilval.store.records import duel_event, index_record, now_iso

        rec = index_record(
            schema=1,
            event_id=f"{block:064x}",
            kind="duel",
            track="sensorimotor",
            block=block,
            finished_at=now_iso(),
            king=req.king,
            challenger=req.challenger,
            king_scores=None,
            challenger_scores=None,
            score_margin=3.0,
            dethroned=self.outcome == "win",
            new_king=req.challenger if self.outcome == "win" else None,
        )
        self.rt.store.write_event(
            "sensorimotor",
            duel_event(
                rec,
                spec_version=1,
                spec_fingerprint="0" * 64,
                units=[],
                units_per_skill=2,
                started_at=now_iso(),
                wall_seconds=1,
            ),
        )
        self.rt.store.append("sensorimotor", rec)
        return rec


def test_daemon_genesis_then_duels(spec, tmp_path, monkeypatch):
    rt = make_rt(spec, tmp_path)
    track = "sensorimotor"
    queues = Queues(tmp_path / "queue", spec.tracks)
    q = queues[track]
    cfg = DaemonConfig(
        store_root=tmp_path / "store",
        queue_path=tmp_path / "queue",
        run_root=tmp_path / "runs",
        docker_image=None,
        local_models={},
        once=True,
    )
    d = Daemon(rt, queues, cfg)
    monkeypatch.setattr(
        "icilval.duel.orchestrate.publish_genesis",
        lambda rt_, track_, ref, block, **kw: (
            FakeOrchestrator(rt_, "win").run(
                type("R", (), {"challenger": ref, "king": None})(), block
            )
            if False
            else _genesis(rt_, ref, block)
        ),
    )
    assert d.step(track) is False
    q.add("org/first", "a" * 40)
    q.add("org/second", "b" * 40)
    assert d.step(track) is True  # genesis
    assert d.current_king(track).repo == "org/first"
    fake = FakeOrchestrator(rt, "win")
    d.orchestrator = fake
    assert d.step(track) is True  # duel: second beats first
    assert fake.calls == [("org/second", "org/first", 2)]
    assert d.current_king(track).repo == "org/second"
    assert q.state.in_progress is None and not q.entries()
    q.add("org/second", "b" * 40)  # the king resubmits: dropped
    assert d.step(track) is True and not q.entries()
    q.add("org/third", "c" * 40)
    d.orchestrator = FakeOrchestrator(rt, "fail")
    assert (
        d.step(track) is True
        and d.current_king(track).repo == "org/second"
        and q.state.in_progress is None
    )
    snap = rt.store.queue_path("sensorimotor")
    assert snap.exists()


def _genesis(rt, ref, block):
    from icilval.store.records import duel_event, index_record, now_iso

    rec = index_record(
        schema=1,
        event_id=f"{1000 + block:064x}",
        kind="genesis",
        track="sensorimotor",
        block=block,
        finished_at=now_iso(),
        king=ref,
        challenger=None,
        king_scores=None,
        challenger_scores=None,
        score_margin=3.0,
        dethroned=False,
        new_king=None,
    )
    rt.store.write_event(
        "sensorimotor",
        duel_event(
            rec,
            spec_version=1,
            spec_fingerprint="0" * 64,
            units=[],
            units_per_skill=0,
            started_at=now_iso(),
            wall_seconds=0,
        ),
    )
    rt.store.append("sensorimotor", rec)
    return rec
