"""`icilval` command line. argparse only: it must import inside the minimal container."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _spec(args):
    from .spec import load_spec

    return load_spec(args.spec)


def cmd_spec(args) -> int:
    from .spec import load_spec, spec_path, validate_spec

    if args.spec_cmd == "fingerprint":
        print(load_spec(args.spec).fingerprint)
        return 0
    path = Path(args.spec) if args.spec else spec_path()
    errors = validate_spec(json.loads(path.read_text()))
    if errors:
        print("invalid:", ", ".join(errors))
        return 1
    print(f"{path}: ok")
    return 0


def cmd_keys(args) -> int:
    from .canon import Signer

    out = Path(args.out)
    secret = out / "validator.ed25519"
    if secret.exists() and not args.force:
        print(f"{secret} exists; pass --force to overwrite", file=sys.stderr)
        return 1
    signer = Signer.generate()
    signer.save(secret)
    (out / "validator.pub").write_text(signer.verify_key_hex + "\n")
    print(signer.verify_key_hex)
    return 0


def cmd_store(args) -> int:
    from .canon import Signer
    from .store.verify import verify_store
    from .store.writer import Store

    spec = _spec(args)
    if args.store_cmd == "init":
        signer = Signer.from_file(args.key)
        store = Store(args.root, spec, signer)
        manifest = store.init(signer.verify_key_hex, args.pool_id)
        print(json.dumps(manifest, indent=2))
        return 0
    if args.store_cmd == "verify":
        report = verify_store(args.root, spec)
        for w in report.warnings:
            print("warning:", w)
        for e in report.errors:
            print("error:", e)
        print(
            f"records={report.records} events={report.events} media={report.media} {'OK' if report.ok else 'FAILED'}"
        )
        return 0 if report.ok else 1
    if args.store_cmd == "mirror":
        from .store.mirror import mirror_store

        n = mirror_store(
            args.root, args.repo, spec, message=args.message, all_files=args.all, prune=args.prune
        )
        print(f"mirrored {n} files to {args.repo}")
        return 0
    return 2


def cmd_queue(args) -> int:
    from .queue import Queue

    q = Queue(args.queue)
    if args.queue_cmd == "add":
        entry, pos = q.add(args.repo, args.revision, duel_size=args.duel_size, source="cli")
        print(f"{entry.ref.entry} key={entry.key} position={pos}")
        return 0
    if args.queue_cmd == "list":
        for i, e in enumerate(q.entries(), start=1):
            print(
                f"{i:3d} {e.key} {e.repo}@{e.revision} size={e.duel_size or '-'} accepted={e.accepted_at}"
            )
        ip = q.state.in_progress
        print(f"in_progress={ip.event_id if ip else '-'} block={q.block}")
        return 0
    if args.queue_cmd == "remove":
        print("removed" if q.remove(args.key) else "not found")
        return 0
    return 2


def cmd_admin(args) -> int:
    from .admin import AdminServer
    from .queue import Queue

    spec = _spec(args)
    key = Path(args.pub).read_text().strip() if args.pub else ""
    server = AdminServer(spec, Queue(args.queue), args.token, key, bind=args.bind, port=args.port)
    print(f"admin listening on http://{server.bind}:{server.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def cmd_units(args) -> int:
    from .ids import ModelRef, duel_id
    from .pools.schema import Pool
    from .pools.units import derive_units

    spec = _spec(args)
    pool = Pool.load(args.pool)
    ch_repo, ch_rev = args.challenger.split("@", 1)
    challenger = ModelRef.make(ch_repo, ch_rev)
    king = None
    if args.king:
        k_repo, k_rev = args.king.split("@", 1)
        king = ModelRef.make(k_repo, k_rev)
    did = duel_id(spec.version, spec.track_id, challenger, king)
    units = [u.as_dict() for u in derive_units(pool, spec, did, args.size)]
    doc = {"duel_id": did, "pool_id": pool.pool_id, "size": spec.size_of(args.size), "units": units}
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
        print(f"{len(units)} units -> {args.out}")
    else:
        print(json.dumps(doc, indent=2))
    return 0


def cmd_pools(args) -> int:
    import logging

    from .pools.build import finalize, open_pool, stage_skill, summary, verify_pool
    from .pools.schema import Pool
    from .pools.sources import Sources
    from .spec import _repo_root

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    spec = _spec(args)

    def sources() -> Sources:
        src = Sources.default(_repo_root() or Path.cwd())
        if getattr(args, "raw", None):
            src.raw = Path(args.raw)
        return src

    if args.pools_cmd == "build":
        out = Path(args.out)
        src = sources()
        stages = args.stage or [*spec.skills, "finalize"]
        unknown = [st for st in stages if st != "finalize" and st not in spec.skills]
        if unknown:
            print("unknown stage(s):", *unknown, "- stages are the skill ids and finalize")
            return 2
        datasets = tuple(
            dict.fromkeys(str(spec.tasks(st)["dataset"]) for st in stages if st != "finalize")
        )
        if args.fetch:
            for d in datasets:
                src.dataset_root(d).mkdir(parents=True, exist_ok=True)
        missing = src.missing(datasets)
        if missing:
            print("missing sources:", *missing, sep="\n  ")
            return 1
        pool = open_pool(out, spec, args.version or str(spec.pools["version"]))
        for stage in stages:
            if stage == "finalize":
                print("eligible:", json.dumps(finalize(pool, spec)))
                continue
            stage_skill(
                pool,
                spec,
                src,
                stage,
                limit=args.limit,
                validate=not args.no_validate,
                fetch_missing=args.fetch,
                evict_demos=args.evict,
            )
        pool.save()
        print(json.dumps(summary(pool), indent=1))
        return 0
    if args.pools_cmd == "verify":
        errors = verify_pool(Path(args.pool), spec)
        for e in errors:
            print("error:", e)
        print(json.dumps(summary(Pool.load(args.pool)), indent=1))
        return 0 if not errors else 1
    if args.pools_cmd == "push":
        from .pools.hub import push_pool

        pool = Pool.load(args.pool)
        print(push_pool(Path(args.pool), args.repo or str(spec.pools["repo"]), pool.pool_version))
        return 0
    if args.pools_cmd == "pull":
        from .pools.hub import pull_pool

        print(
            pull_pool(
                args.repo or str(spec.pools["repo"]),
                args.version or str(spec.pools["version"]),
                Path(args.dest),
                revision=args.revision,
            )
        )
        return 0
    if args.pools_cmd == "generate-draw":
        from .simulators.draw.pool import generate_draw, import_generated_draw
        from .spec import _repo_root as rr

        root = rr() or Path.cwd()
        bpp = Path(args.bpp_root) if args.bpp_root else root / "vendor" / "behavior_prompting"
        run_dir = Path(args.run_dir)
        if not args.import_only:
            generate_draw(
                bpp,
                run_dir,
                n_tasks=args.n_tasks,
                demos_per_task=args.demos_per_task,
                base_seed=args.base_seed,
                workers=args.workers,
                python=args.python,
                dry_run=args.dry_run,
            )
        if args.pool and not args.dry_run:
            pool = Pool.load(args.pool)
            pool.pool_id = None
            got = import_generated_draw(pool, spec, run_dir, skill=args.skill, limit=args.limit)
            print("imported", got)
            print("eligible:", json.dumps(finalize(pool, spec)))
        return 0
    return 2


def _parse_ref(text: str):
    from .ids import ModelRef

    if "@" not in text:
        raise SystemExit(f"model reference must be repo@revision, got {text!r}")
    repo, rev = text.split("@", 1)
    return ModelRef.make(repo, rev)


def _local_models(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        repo, _, path = item.partition("=")
        out[repo] = path
    return out


def _runtime(args, spec, *, enforce_pool_id: bool = True):
    from .daemon import make_runtime
    from .spec import _repo_root

    root = _repo_root() or Path.cwd()
    return make_runtime(
        spec,
        store_root=Path(args.store),
        key_file=Path(args.key),
        pool_dir=Path(args.pool),
        arch_dir=Path(args.arch) if args.arch else root / "arch",
        run_root=Path(args.runs),
        live_url=args.live,
        live_token=args.live_token,
        mirror_repo=args.mirror,
        enforce_pool_id=enforce_pool_id,
    )


def cmd_convert(args) -> int:
    from .model.convert import convert_checkpoint

    out = convert_checkpoint(
        args.ckpt, args.out, arch_name=args.arch_name, emit_arch=args.emit_arch
    )
    print(json.dumps(out, indent=1))
    return 0


def cmd_fetch(args) -> int:
    from .submission import fetch_model

    spec = _spec(args)
    got = fetch_model(_parse_ref(args.ref), Path(args.dest), spec)
    print(f"{got.path} files={len(got.files)} bytes={got.bytes} skipped={got.skipped}")
    return 0


def cmd_check(args) -> int:
    from .model.fingerprint import check_submission
    from .spec import _repo_root

    spec = _spec(args)
    arch = Path(args.arch) if args.arch else (_repo_root() or Path.cwd()) / "arch"
    rep = check_submission(
        Path(args.model), spec, arch, skills=tuple(args.skill) if args.skill else None
    )
    for e in rep.errors:
        print("error:", e)
    for w in rep.warnings:
        print("warning:", w)
    for skill, r in rep.skills.items():
        print(
            f"{skill}: {r.architecture} params={r.param_count} model_sha256={r.model_sha256[:16]} {'ok' if r.ok else 'rejected'}"
        )
    print(f"bytes={rep.repo_bytes} {'OK' if rep.ok else 'REJECTED'}")
    return 0 if rep.ok else 1


def cmd_render_demo(args) -> int:
    from .pools.demos import render_demo
    from .pools.schema import Pool

    spec = _spec(args)
    pool = Pool.load(args.pool)
    task_id = args.demo.rsplit("/", 1)[0]
    skill = pool.tasks[task_id].skill
    sha = render_demo(pool.path("demos") / f"{args.demo}.npz", args.out, spec, skill)
    print(f"{args.out} ({skill}) sha256={sha}")
    return 0


def cmd_run_side(args) -> int:
    import logging

    from .duel.side_runner import run_side
    from .pools.schema import Pool

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    spec = _spec(args)
    doc = json.loads(Path(args.units).read_text())
    units = doc["units"] if isinstance(doc, dict) else doc
    summary = run_side(
        side=args.side,
        model_dir=Path(args.model),
        arch_dir=Path(args.arch),
        pool=Pool.load(args.pool),
        units=units,
        spec=spec,
        out_dir=Path(args.out),
        device=args.device,
        record_video=not args.no_video,
    )
    print(json.dumps(summary, indent=1))
    return 0


def cmd_duel(args) -> int:
    import logging

    from .duel.orchestrate import DuelFailed, DuelRequest, Orchestrator

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    spec = _spec(args)
    rt = _runtime(args, spec)
    req = DuelRequest(
        challenger=_parse_ref(args.challenger),
        king=_parse_ref(args.king) if args.king else None,
        size=args.size,
        skip_model_check=args.skip_model_check,
        local_models=_local_models(args.local_model),
        docker_image=args.docker_image,
        gpus=args.gpus,
        device=args.device,
        record_video=not args.no_video,
    )
    from .store.writer import store_lock

    try:
        with store_lock(args.store):
            record = Orchestrator(rt).run(req, args.block)
    except DuelFailed as exc:
        print("duel failed:", exc)
        return 1
    if rt.mirror is not None:
        rt.mirror.push(rt.store.drain_touched())
    print(json.dumps(record, indent=1))
    return 0


def cmd_genesis(args) -> int:
    from .duel.orchestrate import DuelFailed, publish_genesis

    spec = _spec(args)
    rt = _runtime(args, spec)
    try:
        record = publish_genesis(
            rt,
            _parse_ref(args.king),
            args.block,
            local_models=_local_models(args.local_model),
            check=not args.skip_model_check,
        )
    except DuelFailed as exc:
        print("genesis refused:", exc)
        return 1
    if rt.mirror is not None:
        rt.mirror.push(rt.store.drain_touched())
    print(json.dumps(record, indent=1))
    return 0


def cmd_daemon(args) -> int:
    import logging
    import threading

    from .daemon import Daemon, DaemonConfig
    from .queue import Queue

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    spec = _spec(args)
    rt = _runtime(args, spec)
    queue = Queue(args.queue)
    if args.admin_token:
        from .admin import AdminServer

        server = AdminServer(
            spec, queue, args.admin_token, rt.signer.verify_key_hex, lock=threading.Lock()
        )
        server.start_background()
        print(f"admin listening on http://{server.bind}:{server.port}")
    cfg = DaemonConfig(
        store_root=Path(args.store),
        queue_path=Path(args.queue),
        run_root=Path(args.runs),
        docker_image=args.docker_image,
        local_models=_local_models(args.local_model),
        once=args.once,
    )
    Daemon(rt, queue, cfg).run()
    return 0


def cmd_smoke(args) -> int:
    """Genesis + a genesis-vs-genesis duel into a fresh store, then verify it. In-process."""
    import logging
    import tempfile

    from .canon import Signer
    from .duel.orchestrate import DuelFailed, DuelRequest, Orchestrator, publish_genesis
    from .ids import ModelRef
    from .store.verify import verify_store

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    spec = _spec(args)
    store_root = Path(args.store)
    key = (
        Path(args.key)
        if args.key
        else Path(tempfile.mkdtemp(prefix="icilval-smoke-")) / "validator.ed25519"
    )
    if not key.exists():
        Signer.generate().save(key)
    args.key = str(key)
    args.mirror = None
    rt = _runtime(args, spec, enforce_pool_id=False)  # a smoke pool is never the pinned one
    king = ModelRef.make(args.repo, args.revision)
    challenger = ModelRef.make(args.repo, args.revision[::-1] if args.same_model else args.revision)
    local = {args.repo: args.model_dir}
    block = 1
    if rt.store.head(spec.track_id) is None or not rt.store.head(spec.track_id).get("king"):
        publish_genesis(rt, king, block, local_models=local, check=True)
        block += 1
    req = DuelRequest(
        challenger=challenger,
        king=king,
        size=args.size or "smoke",
        local_models=local,
        device=args.device,
        record_video=not args.no_video,
    )
    try:
        record = Orchestrator(rt).run(req, block)
    except DuelFailed as exc:
        print("smoke duel failed:", exc)
        return 1
    report = verify_store(store_root, spec)
    for e in report.errors:
        print("error:", e)
    print(
        json.dumps(
            {
                "event_id": record["event_id"],
                "dethroned": record["dethroned"],
                "king_scores": record["king_scores"],
                "challenger_scores": record["challenger_scores"],
                "wins": record["wins"],
                "losses": record["losses"],
                "ties": record["ties"],
                "void": record["void"],
                "media": record["media_count"],
                "store_ok": report.ok,
            },
            indent=1,
        )
    )
    return 0 if report.ok else 1


def _add_runtime_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--store", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--key", default="keys/validator.ed25519", help="validator.ed25519 secret file")
    p.add_argument("--arch", default=None)
    p.add_argument("--runs", default="runs")
    p.add_argument("--live", default=None, help="dashboard base url for live frames")
    p.add_argument("--live-token", default=None)
    p.add_argument(
        "--mirror", default=None, help="Hugging Face dataset repo to mirror the store to"
    )
    p.add_argument(
        "--local-model", action="append", default=None, help="repo=/local/dir (offline models)"
    )
    p.add_argument(
        "--docker-image", default=None, help="run model sides in this image with --network none"
    )
    p.add_argument("--device", default="cuda")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="icilval", description="ICIL competition validator")
    p.add_argument("--spec", help="path to spec.json (default: packaged / repo root)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("spec", help="inspect the contract")
    s.add_argument("spec_cmd", choices=["fingerprint", "validate"])
    s.set_defaults(func=cmd_spec)

    k = sub.add_parser("keys", help="generate the validator signing key")
    k.add_argument("keys_cmd", choices=["generate"])
    k.add_argument("--out", default="keys")
    k.add_argument("--force", action="store_true")
    k.set_defaults(func=cmd_keys)

    st = sub.add_parser("store", help="result store operations")
    st_sub = st.add_subparsers(dest="store_cmd", required=True)
    st_init = st_sub.add_parser("init")
    st_init.add_argument("root")
    st_init.add_argument("--key", required=True, help="validator.ed25519 secret file")
    st_init.add_argument("--pool-id", default=None)
    st_ver = st_sub.add_parser("verify")
    st_ver.add_argument("root")
    st_mir = st_sub.add_parser("mirror")
    st_mir.add_argument("root")
    st_mir.add_argument("--repo", required=True, help="Hugging Face dataset repo owner/name")
    st_mir.add_argument("--message", default="publish")
    st_mir.add_argument(
        "--all", action="store_true", help="upload every file, not only changed ones"
    )
    st_mir.add_argument(
        "--prune",
        action="store_true",
        help="make the repo exactly this store in one commit: upload every file and delete every "
        "path the store no longer has (for a rebuilt store, e.g. after a schema change)",
    )
    st.set_defaults(func=cmd_store)

    q = sub.add_parser("queue", help="challenger queue")
    q.add_argument("--queue", default="queue/queue.json")
    q_sub = q.add_subparsers(dest="queue_cmd", required=True)
    q_add = q_sub.add_parser("add")
    q_add.add_argument("repo")
    q_add.add_argument("revision")
    q_add.add_argument("--duel-size", default=None)
    q_sub.add_parser("list")
    q_rm = q_sub.add_parser("remove")
    q_rm.add_argument("key")
    q.set_defaults(func=cmd_queue)

    a = sub.add_parser("admin", help="submission intake server")
    a.add_argument("admin_cmd", choices=["serve"])
    a.add_argument("--queue", default="queue/queue.json")
    a.add_argument("--token", required=True)
    a.add_argument("--pub", default=None, help="validator.pub, echoed by /admin/health")
    a.add_argument("--bind", default=None)
    a.add_argument("--port", type=int, default=None)
    a.set_defaults(func=cmd_admin)

    po = sub.add_parser("pools", help="build, verify and distribute evaluation pools")
    po_sub = po.add_subparsers(dest="pools_cmd", required=True)
    po_b = po_sub.add_parser("build")
    po_b.add_argument("--out", required=True)
    po_b.add_argument(
        "--stage",
        nargs="*",
        default=None,
        help="skill ids (spec order) and finalize; default: all",
    )
    po_b.add_argument("--limit", type=int, default=None, help="tasks per view (smoke builds)")
    po_b.add_argument(
        "--no-validate", action="store_true", help="skip simulator validation (no instance lists)"
    )
    po_b.add_argument("--raw", default=None, help="raw cache dir (default ~/.cache/icilval/raw)")
    po_b.add_argument(
        "--fetch", action="store_true", help="download missing LIBERO-Gen files from the hub"
    )
    po_b.add_argument(
        "--evict",
        action="store_true",
        help="delete each demonstration hdf5 after its demos are imported (they total ~160 GB)",
    )
    po_b.add_argument("--version", default=None)
    po_v = po_sub.add_parser("verify")
    po_v.add_argument("pool")
    po_p = po_sub.add_parser("push")
    po_p.add_argument("pool")
    po_p.add_argument("--repo", default=None)
    po_l = po_sub.add_parser("pull")
    po_l.add_argument("--dest", required=True)
    po_l.add_argument("--repo", default=None)
    po_l.add_argument("--version", default=None)
    po_l.add_argument("--revision", default=None)
    po_gd = po_sub.add_parser(
        "generate-draw", help="run BPP's procedural drawing generator, then import"
    )
    po_gd.add_argument("--run-dir", required=True)
    po_gd.add_argument("--pool", default=None)
    po_gd.add_argument("--skill", default="draw_anything", help="the drawing skill the run feeds")
    po_gd.add_argument("--n-tasks", type=int, default=50)
    po_gd.add_argument("--demos-per-task", type=int, default=10)
    po_gd.add_argument("--base-seed", type=int, required=True, help="the organizer's secret seed")
    po_gd.add_argument("--workers", type=int, default=8)
    po_gd.add_argument("--python", default="python")
    po_gd.add_argument("--bpp-root", default=None)
    po_gd.add_argument("--limit", type=int, default=None)
    po_gd.add_argument("--dry-run", action="store_true")
    po_gd.add_argument("--import-only", action="store_true")
    po.set_defaults(func=cmd_pools)

    u = sub.add_parser("units", help="derive a duel's unit list")
    u.add_argument("units_cmd", choices=["derive"])
    u.add_argument("--pool", required=True)
    u.add_argument("--challenger", required=True, help="repo@revision")
    u.add_argument("--king", default=None, help="repo@revision")
    u.add_argument("--size", default=None)
    u.add_argument("--out", default=None)
    u.set_defaults(func=cmd_units)

    c = sub.add_parser("convert-ckpt", help="BPP checkpoint -> model.safetensors + config.yaml")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--arch-name", required=True, help="bpp_libero_v1 or bpp_draw_v1")
    c.add_argument("--emit-arch", default=None, help="also write the arch templates into this dir")
    c.set_defaults(func=cmd_convert)

    f = sub.add_parser("fetch-model", help="download a submission's allow-listed files")
    f.add_argument("ref", help="repo@revision")
    f.add_argument("--dest", required=True)
    f.set_defaults(func=cmd_fetch)

    ck = sub.add_parser(
        "check-model", help="fingerprint a submission directory (one dir per skill)"
    )
    ck.add_argument("model")
    ck.add_argument("--arch", default=None)
    ck.add_argument("--skill", action="append", default=None, help="check only these skills")
    ck.set_defaults(func=cmd_check)

    rd = sub.add_parser("render-demo", help="render a pool demonstration to mp4")
    rd.add_argument("--pool", required=True)
    rd.add_argument(
        "--demo",
        required=True,
        help="demo id, e.g. libero_spatial/<task>/demo_00 or drawanything_handmade/<task>/demo_00",
    )
    rd.add_argument("--out", required=True)
    rd.set_defaults(func=cmd_render_demo)

    rs = sub.add_parser(
        "run-side", help="run one side over a unit list (inside the model container)"
    )
    rs.add_argument("--model", required=True)
    rs.add_argument("--pool", required=True)
    rs.add_argument("--arch", required=True)
    rs.add_argument("--units", required=True)
    rs.add_argument("--side", required=True, choices=["challenger", "king"])
    rs.add_argument("--out", required=True)
    rs.add_argument("--device", default="cuda")
    rs.add_argument("--no-video", action="store_true")
    rs.set_defaults(func=cmd_run_side)

    d = sub.add_parser("duel", help="run and publish one duel")
    _add_runtime_args(d)
    d.add_argument("--challenger", required=True, help="repo@revision")
    d.add_argument("--king", default=None, help="repo@revision (default: none)")
    d.add_argument("--size", default=None)
    d.add_argument("--block", type=int, default=1)
    d.add_argument("--skip-model-check", action="store_true")
    d.add_argument("--gpus", default="all")
    d.add_argument("--no-video", action="store_true")
    d.set_defaults(func=cmd_duel)

    g = sub.add_parser("genesis", help="crown the opening entrant without a duel")
    _add_runtime_args(g)
    g.add_argument("--king", required=True, help="repo@revision")
    g.add_argument("--block", type=int, default=1)
    g.add_argument("--skip-model-check", action="store_true")
    g.set_defaults(func=cmd_genesis)

    dm = sub.add_parser("daemon", help="the validator loop")
    _add_runtime_args(dm)
    dm.add_argument("--queue", default="queue/queue.json")
    dm.add_argument("--admin-token", default=None, help="also serve the submission intake")
    dm.add_argument("--once", action="store_true")
    dm.set_defaults(func=cmd_daemon)

    sm = sub.add_parser(
        "smoke", help="genesis + genesis-vs-genesis duel into a fresh store (in-process)"
    )
    _add_runtime_args(sm)
    sm.set_defaults(key=None)
    sm.add_argument(
        "--model-dir",
        required=True,
        help="local submission directory: <skill>/model.safetensors per skill",
    )
    sm.add_argument("--repo", default="local/bpp-genesis")
    sm.add_argument("--revision", default="0000000000000000000000000000000000000001")
    sm.add_argument(
        "--same-model",
        action="store_true",
        help="challenger = a distinct ref to the same weights (copy-of-king case)",
    )
    sm.add_argument("--size", default="smoke")
    sm.add_argument("--no-video", action="store_true")
    sm.set_defaults(func=cmd_smoke)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
