"""A duel from queue entry to published record.

fetching -> checking -> evaluating(challenger) -> evaluating(king) -> publishing -> done|failed

Each side runs either in-process (this interpreter has the simulators) or in an isolated
container with no network. A side runs every skill's units with that skill's checkpoint from
the submission. Progress is posted as live frames; finished clips are copied into the store
every few units so the dashboard can show them while the duel runs.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import simulators
from ..canon import Signer
from ..ids import ModelRef, duel_id, event_id
from ..live import LiveReporter, build_frame
from ..model.fingerprint import check_submission
from ..pools.demos import render_demo
from ..pools.schema import Pool
from ..pools.units import derive_units
from ..spec import Spec
from ..store.records import duel_event, index_record, media_shas, now_iso, unit_verdict_from_unit
from ..store.writer import Store
from ..submission import fetch_model
from . import score
from .side_runner import read_results, run_side

log = logging.getLogger(__name__)


class DuelFailed(Exception):
    pass


@dataclass
class DuelRequest:
    challenger: ModelRef
    king: ModelRef | None
    size: str | None = None
    skip_model_check: bool = False
    local_models: dict[str, str] = field(default_factory=dict)
    docker_image: str | None = None
    gpus: str = "all"
    device: str = "cuda"
    record_video: bool = True
    kind: str = "duel"


@dataclass
class Runtime:
    spec: Spec
    pool: Pool
    store: Store
    signer: Signer
    arch_dir: Path
    run_root: Path
    live: LiveReporter
    mirror: Any | None = None  # object with .push(files: list[str]) -> None


class Orchestrator:
    def __init__(self, rt: Runtime):
        self.rt = rt
        self.spec = rt.spec

    # ---------------------------------------------------------------- helpers
    def _frame(self, state: dict[str, Any], **kw: Any) -> dict[str, Any]:
        return build_frame(
            self.spec,
            validator_key=self.rt.signer.verify_key_hex,
            event_id=state["event_id"],
            kind=state["kind"],
            duel_size=state["size"],
            king=state["king"].as_dict() if state["king"] else None,
            challenger=state["challenger"].as_dict(),
            phase=kw.get("phase", state["phase"]),
            side=kw.get("side", state.get("side")),
            units=state["units"],
            current=kw.get("current", state.get("current")),
            recent_media=kw.get("recent_media", state.get("recent_media")),
            message=kw.get("message", state.get("message", "")),
            started_at=state["started_at"],
        )

    def _post(self, state: dict[str, Any], force: bool = False, **kw: Any) -> None:
        for k, v in kw.items():
            state[k] = v
        self.rt.live.post(self._frame(state), force=force)

    def _flush_media(
        self, state: dict[str, Any], side: str, side_dir: Path, force: bool = False
    ) -> None:
        """Copy finished clips into the store, mirror them, then let frames reference the shas."""
        pending = state.setdefault("pending_media", {})
        results = read_results(side_dir / "units.jsonl")
        for unit_id, rec in results.items():
            if (
                rec.get("video")
                and rec.get("video_sha256")
                and (side, unit_id) not in state["media_done"]
            ):
                pending[(side, unit_id)] = side_dir / rec["video"]
        if not pending:
            return
        if not force and len(pending) < int(self.spec.live["media_flush_units"]):
            return
        ext = str(self.spec.media["video"]["format"])
        touched: list[str] = []
        for (s, unit_id), path in list(pending.items()):
            sha = self.rt.store.put_media(path, ext)
            touched.append(str(self.rt.store.media_path(sha, ext).relative_to(self.rt.store.root)))
            state["media_done"][(s, unit_id)] = sha
            del pending[(s, unit_id)]
        if self.rt.mirror is not None:
            try:
                self.rt.mirror.push(touched)
            except Exception as exc:  # noqa: BLE001 - mirroring is retried at publish
                log.warning("media mirror failed: %s", exc)
                return
        for (s, unit_id), sha in state["media_done"].items():
            for u in state["units"]:
                if u["unit_id"] == unit_id:
                    u[f"{s}_video"] = sha
                    state["recent_media"] = {
                        "unit": {
                            "skill": u["skill"],
                            "task": u["task"],
                            "task_label": u.get("task_label"),
                            "instance": u["instance"],
                        },
                        "side": s,
                        "demo_video": u.get("demo_video"),
                        "video": sha,
                        "success": u.get(f"{s}_success"),
                    }

    def _merge(self, state: dict[str, Any], side: str, rec: dict[str, Any]) -> None:
        for u in state["units"]:
            if u["unit_id"] != rec["unit_id"]:
                continue
            if rec.get("void"):
                u["void"] = True
                u[f"{side}_error"] = rec.get("error")
                u[f"{side}_success"] = None
            else:
                u[f"{side}_success"] = bool(rec.get("success"))
                u[f"{side}_progress"] = rec.get("progress")
                u[f"{side}_metric"] = rec.get("metric")
                u[f"{side}_steps"] = rec.get("steps")
                u[f"{side}_error"] = rec.get("error")
                if rec.get("prompt_chunks"):
                    u["prompt"] = {
                        "demo_id": u["prompt"]["demo_id"],
                        "steps": rec.get("prompt_steps", 0),
                        "chunks": rec.get("prompt_chunks", 0),
                    }
            u["outcome"] = score.paired_outcome(u.get("king_success"), u.get("challenger_success"))
            return

    # ---------------------------------------------------------------- sides
    def _run_side_inprocess(
        self, state: dict[str, Any], side: str, model_dir: Path, side_dir: Path, req: DuelRequest
    ) -> None:
        def on_unit(rec: dict[str, Any]) -> None:
            self._merge(state, side, rec)
            self._flush_media(state, side, side_dir)
            self._post(
                state,
                side=side,
                current=None,
                message=f"{side}: {rec['unit_id']} {'ok' if rec.get('success') else 'fail' if rec.get('success') is False else 'void'}",
            )

        run_side(
            side=side,
            model_dir=model_dir,
            arch_dir=self.rt.arch_dir,
            pool=self.rt.pool,
            units=state["unit_defs"],
            spec=self.spec,
            out_dir=side_dir,
            device=req.device,
            on_unit=on_unit,
            record_video=req.record_video,
        )

    def _run_side_docker(
        self, state: dict[str, Any], side: str, model_dir: Path, side_dir: Path, req: DuelRequest
    ) -> None:
        side_dir.mkdir(parents=True, exist_ok=True)
        (side_dir / "units.json").write_text(json.dumps(state["unit_defs"]))
        cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--gpus",
            req.gpus,
            "--shm-size",
            "2g",
            # run as the host user so the side directory stays writable and its files stay ours
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            "HOME=/tmp",
            "-v",
            f"{model_dir.resolve()}:/model:ro",
            "-v",
            f"{self.rt.pool.root.resolve()}:/pool:ro",
            "-v",
            f"{self.rt.arch_dir.resolve()}:/arch:ro",
            "-v",
            f"{side_dir.resolve()}:/work:rw",
            req.docker_image,
            "icilval",
            "run-side",
            "--model",
            "/model",
            "--pool",
            "/pool",
            "--arch",
            "/arch",
            "--units",
            "/work/units.json",
            "--side",
            side,
            "--out",
            "/work",
        ]
        log.info("docker: %s", " ".join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        seen: set[str] = set()
        try:
            while proc.poll() is None:
                self._poll_side(state, side, side_dir, seen)
                time.sleep(2.0)
            self._poll_side(state, side, side_dir, seen)
        finally:
            if proc.poll() is None:
                proc.kill()
        if proc.returncode != 0:
            err = proc.stderr.read().decode(errors="replace")[-2000:] if proc.stderr else ""
            raise DuelFailed(f"{side} container exited {proc.returncode}: {err}")

    def _poll_side(self, state: dict[str, Any], side: str, side_dir: Path, seen: set[str]) -> None:
        for unit_id, rec in read_results(side_dir / "units.jsonl").items():
            if unit_id in seen:
                continue
            seen.add(unit_id)
            self._merge(state, side, rec)
        self._flush_media(state, side, side_dir)
        self._post(state, side=side, message=f"{side}: {len(seen)}/{len(state['units'])} units")

    # ---------------------------------------------------------------- the duel
    def run(self, req: DuelRequest, block: int) -> dict[str, Any]:
        spec = self.spec
        # Before anything is fetched: a duel that would score a skill whose benchmark is not
        # installed must stop here, not halfway through with half a score.
        simulators.require(spec)
        size = spec.size_of(req.size)
        did = duel_id(spec.version, spec.track_id, req.challenger, req.king)
        eid = event_id(req.kind, spec.track_id, block, did)
        run_dir = self.rt.run_root / eid[:16]
        run_dir.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "event_id": eid,
            "kind": req.kind,
            "size": size,
            "king": req.king,
            "challenger": req.challenger,
            "phase": "fetching",
            "side": None,
            "units": [],
            "unit_defs": [],
            "started_at": now_iso(),
            "message": "",
            "media_done": {},
            "recent_media": None,
        }
        t0 = time.monotonic()
        sides_meta: dict[str, Any] = {}
        try:
            # ---- fetching
            self._post(state, force=True, message="fetching models")
            dirs: dict[str, Path] = {}
            for side, ref in (("challenger", req.challenger), ("king", req.king)):
                if ref is None:
                    continue
                dest = self.rt.run_root / "models" / ref.key
                got = fetch_model(ref, dest, spec, local_models=req.local_models)
                dirs[side] = got.path
            # ---- checking
            self._post(
                state, force=True, phase="checking", message="checking architectures and weights"
            )
            for side, d in dirs.items():
                rep = check_submission(d, spec, self.rt.arch_dir)
                sides_meta[side] = {
                    "repo_bytes": rep.repo_bytes,
                    "skills": {
                        s: {
                            "model_sha256": r.model_sha256,
                            "config_sha256": r.config_sha256,
                            "param_count": r.param_count,
                        }
                        for s, r in rep.skills.items()
                    },
                }
                if not rep.ok and not (req.skip_model_check and side == "challenger"):
                    raise DuelFailed(f"{side} failed the model check: " + "; ".join(rep.errors[:5]))
            # ---- units
            units = derive_units(self.rt.pool, spec, did, size)
            state["unit_defs"] = [u.as_dict() for u in units]
            state["units"] = [unit_verdict_from_unit(u.as_dict()) for u in units]
            self._render_demos(state, run_dir)
            # ---- evaluating
            for side in ("challenger", "king"):
                if side not in dirs:
                    continue
                side_dir = run_dir / side
                self._post(
                    state, force=True, phase="evaluating", side=side, message=f"evaluating {side}"
                )
                t_side = time.monotonic()
                if req.docker_image:
                    self._run_side_docker(state, side, dirs[side], side_dir, req)
                else:
                    self._run_side_inprocess(state, side, dirs[side], side_dir, req)
                self._flush_media(state, side, side_dir, force=True)
                sides_meta.setdefault(side, {})["wall_seconds"] = round(
                    time.monotonic() - t_side, 1
                )
                summary = read_summary(side_dir)
                sides_meta[side]["errors"] = int(summary.get("errors", 0))
                if time.monotonic() - t0 > float(spec.budgets["duel_wall_seconds"]):
                    raise DuelFailed("duel wall time exceeded")
            # ---- scoring
            v = score.verdict(state["units"], spec.score_margin, spec.all_skills)
            if score.void_fraction(state["units"]) > spec.max_void_fraction:
                raise DuelFailed(f"{v.tally.void} of {len(state['units'])} units void")
            # ---- publishing
            self._post(
                state, force=True, phase="publishing", side=None, message="publishing the record"
            )
            record = index_record(
                schema=int(spec.store["schema"]),
                event_id=eid,
                kind=req.kind,
                track=spec.track_id,
                block=block,
                finished_at=now_iso(),
                king=req.king,
                challenger=req.challenger,
                king_scores=v.king_scores if req.king else None,
                challenger_scores=v.challenger_scores,
                score_margin=spec.score_margin,
                dethroned=bool(v.dethroned),
                new_king=req.challenger if v.dethroned else None,
                tally=v.tally.as_dict(),
                media_count=len(media_shas(state["units"])),
                duel_size=size,
                duel_id=did,
                pool_id=self.rt.pool.pool_id,
            )
            event = duel_event(
                record,
                spec_version=spec.version,
                spec_fingerprint=spec.fingerprint,
                units=state["units"],
                units_per_skill=spec.units_per_skill(size),
                started_at=state["started_at"],
                wall_seconds=time.monotonic() - t0,
                sides=sides_meta,
                notes=[],
            )
            self.rt.store.write_event(spec.track_id, event)
            record["seq"] = self.rt.store.append(spec.track_id, record)
            (run_dir / "record.json").write_text(json.dumps(record, indent=2))
            self._post(
                state,
                force=True,
                phase="done",
                message=f"crown {'moves' if v.dethroned else 'stays'}: {v.reason}",
            )
            return record
        except DuelFailed as exc:
            log.error("duel failed: %s", exc)
            (run_dir / "failed.txt").write_text(str(exc))
            self._post(state, force=True, phase="failed", message=str(exc)[:280])
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("duel crashed")
            (run_dir / "failed.txt").write_text(repr(exc))
            self._post(
                state, force=True, phase="failed", message=f"{type(exc).__name__}: {exc}"[:280]
            )
            raise DuelFailed(str(exc)) from exc

    def _render_demos(self, state: dict[str, Any], run_dir: Path) -> None:
        if not self.spec.media.get("demo_video", True):
            return
        ext = str(self.spec.media["video"]["format"])
        demo_dir = run_dir / "demos"
        demo_dir.mkdir(exist_ok=True)
        cache: dict[str, str] = {}
        touched: list[str] = []
        for u, d in zip(state["units"], state["unit_defs"], strict=True):
            demo = d["demo"]
            if demo not in cache:
                out = demo_dir / (demo.replace("/", "__") + f".{ext}")
                try:
                    render_demo(
                        self.rt.pool.path("demos") / f"{demo}.npz",
                        out,
                        self.spec,
                        d["skill"],
                    )
                    sha = self.rt.store.put_media(out, ext)
                    touched.append(
                        str(self.rt.store.media_path(sha, ext).relative_to(self.rt.store.root))
                    )
                    cache[demo] = sha
                except Exception as exc:  # noqa: BLE001
                    log.warning("demo clip failed for %s: %s", demo, exc)
                    cache[demo] = ""
            u["demo_video"] = cache[demo] or None
        if self.rt.mirror is not None and touched:
            try:
                self.rt.mirror.push(touched)
            except Exception as exc:  # noqa: BLE001
                log.warning("demo mirror failed: %s", exc)


def read_summary(side_dir: Path) -> dict[str, Any]:
    p = side_dir / "summary.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def publish_genesis(
    rt: Runtime,
    king: ModelRef,
    block: int,
    *,
    local_models: dict[str, str] | None = None,
    check: bool = True,
) -> dict[str, Any]:
    spec = rt.spec
    simulators.require(spec)
    if check:
        got = fetch_model(king, rt.run_root / "models" / king.key, spec, local_models=local_models)
        rep = check_submission(got.path, spec, rt.arch_dir)
        if not rep.ok:
            raise DuelFailed("genesis model failed the check: " + "; ".join(rep.errors[:5]))
    eid = event_id("genesis", spec.track_id, block, king.key)
    record = index_record(
        schema=int(spec.store["schema"]),
        event_id=eid,
        kind="genesis",
        track=spec.track_id,
        block=block,
        finished_at=now_iso(),
        king=king,
        challenger=None,
        king_scores=None,
        challenger_scores=None,
        score_margin=spec.score_margin,
        dethroned=False,
        new_king=None,
        media_count=0,
        duel_size=None,
        duel_id=None,
        pool_id=rt.pool.pool_id,
    )
    event = duel_event(
        record,
        spec_version=spec.version,
        spec_fingerprint=spec.fingerprint,
        units=[],
        units_per_skill=0,
        started_at=now_iso(),
        wall_seconds=0.0,
        notes=["The opening entrant took an empty throne."],
    )
    rt.store.write_event(spec.track_id, event)
    record["seq"] = rt.store.append(spec.track_id, record)
    return record


def cleanup_run(run_root: Path, keep: int = 5) -> None:
    runs = sorted(
        (p for p in run_root.iterdir() if p.is_dir() and p.name != "models"),
        key=lambda p: p.stat().st_mtime,
    )
    for p in runs[:-keep]:
        shutil.rmtree(p, ignore_errors=True)


__all__ = [
    "Orchestrator",
    "DuelRequest",
    "Runtime",
    "DuelFailed",
    "publish_genesis",
    "cleanup_run",
    "threading",
]
