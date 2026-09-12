"""The validator loop: queue -> duel -> publish -> mirror, forever. One instance per store."""

from __future__ import annotations

import fcntl
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import simulators
from .canon import Signer
from .duel.orchestrate import DuelFailed, DuelRequest, Orchestrator, Runtime
from .ids import ModelRef
from .queue import Queues
from .spec import Spec
from .store.writer import Store

log = logging.getLogger(__name__)


@dataclass
class DaemonConfig:
    store_root: Path
    queue_path: Path
    run_root: Path
    docker_image: str | None
    local_models: dict[str, str]
    idle_sleep_s: float = 15.0
    once: bool = False


class Daemon:
    def __init__(self, rt: Runtime, queues: Queues, cfg: DaemonConfig):
        self.rt = rt
        self.queues = queues
        self.cfg = cfg
        self.orchestrator = Orchestrator(rt)
        self._lock_fd: int | None = None

    def lock(self) -> None:
        path = self.cfg.store_root / ".validator.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                f"another validator is publishing to {self.cfg.store_root} (holds {path})"
            ) from exc

    def current_king(self, track: str) -> ModelRef | None:
        head = self.rt.store.head(track)
        return ModelRef.from_dict(head.get("king")) if head else None

    def publish_queue(self, track: str) -> None:
        snap = self.queues[track].snapshot(
            track, self.current_king(track), int(self.rt.spec.store["schema"])
        )
        self.rt.store.write_queue(track, snap)
        self._mirror(self.rt.store.drain_touched())

    def _mirror(self, files: list[str]) -> None:
        if self.rt.mirror is None or not files:
            return
        try:
            self.rt.mirror.push(files)
        except Exception as exc:  # noqa: BLE001
            log.warning("mirror failed (will retry next publish): %s", exc)

    def step(self, track: str) -> bool:
        """Run one entry of one field's queue. Returns False when that queue is empty."""
        queue = self.queues[track]
        entry = queue.peek()
        if entry is None:
            return False
        king = self.current_king(track)
        if king is None:
            log.info("empty throne: genesis for %s", entry.ref.entry)
            from .duel.orchestrate import publish_genesis

            block = queue.advance_block()
            queue.pop()
            try:
                publish_genesis(
                    self.rt,
                    track,
                    entry.ref,
                    block,
                    local_models=self.cfg.local_models,
                    check=not entry.skip_model_config_check,
                )
            except DuelFailed as exc:
                log.error("genesis refused: %s", exc)
            self.publish_queue(track)
            self._mirror(self.rt.store.drain_touched())
            return True
        if king.key == entry.key:
            log.info("%s already holds the crown; dropping the entry", entry.ref.entry)
            queue.pop()
            self.publish_queue(track)
            return True
        block = queue.advance_block()
        req = DuelRequest(
            track=track,
            challenger=entry.ref,
            king=king,
            size=entry.duel_size,
            skip_model_check=entry.skip_model_config_check,
            local_models=self.cfg.local_models,
            docker_image=self.cfg.docker_image,
        )
        from .ids import duel_id, event_id

        eid = event_id(
            "duel",
            track,
            block,
            duel_id(self.rt.spec.version, track, entry.ref, king),
        )
        queue.pop()
        queue.start(eid, entry.ref)
        self.publish_queue(track)
        try:
            self.orchestrator.run(req, block)
        except DuelFailed:
            pass
        finally:
            queue.finish()
            self.publish_queue(track)
            self._mirror(self.rt.store.drain_touched())
        return True

    def step_all(self) -> bool:
        """One entry from each field, in turn. Returns False when every queue is empty.

        Round-robin rather than a worker per field: the store has one writer, and a lock fine
        enough to let two fields publish at once would risk a torn index. A field whose benchmark
        is not installed is logged and skipped, so an absent plugin never stops another's queue.
        """
        ran = False
        for track in self.queues.tracks:
            try:
                simulators.require(self.rt.spec, self.rt.spec.skills(track))
            except simulators.MissingBenchmark as exc:
                log.warning("%s: skipped, %s", track, exc)
                continue
            ran |= self.step(track)
        return ran

    def run(self) -> None:
        self.lock()
        log.info(
            "validator %s watching %s", self.rt.signer.verify_key_hex[:12], self.cfg.queue_path
        )
        for track in self.queues.tracks:
            self.publish_queue(track)
        while True:
            try:
                ran = self.step_all()
            except Exception:  # noqa: BLE001
                log.exception("daemon step crashed; continuing")
                ran = True
            if self.cfg.once:
                return
            if not ran:
                time.sleep(self.cfg.idle_sleep_s)


def make_runtime(
    spec: Spec,
    *,
    store_root: Path,
    key_file: Path,
    pool_dir: Path,
    arch_dir: Path,
    run_root: Path,
    live_url: str | None,
    live_token: str | None,
    mirror_repo: str | None,
    hf_token: str | None = None,
    enforce_pool_id: bool = True,
) -> Runtime:
    """`enforce_pool_id`: the daemon and the publishing commands refuse a pool other than the one
    pinned in spec.json; the smoke test runs a small local pool and turns the check off."""
    from .live import LiveReporter
    from .pools.schema import Pool

    signer = Signer.from_file(key_file)
    pool = Pool.load(pool_dir)
    if enforce_pool_id and spec.pools.get("pool_id") and spec.pools["pool_id"] != pool.pool_id:
        raise RuntimeError(
            f"pool {pool.pool_id} does not match spec.pools.pool_id {spec.pools['pool_id']}"
        )
    store = Store(store_root, spec, signer)
    if store.manifest() is None:
        store.init(signer.verify_key_hex, pool.pool_id)
    mirror: Any | None = None
    if mirror_repo:
        from .store.mirror import Mirror

        mirror = Mirror(store_root, mirror_repo, token=hf_token)
    return Runtime(
        spec=spec,
        pool=pool,
        store=store,
        signer=signer,
        arch_dir=arch_dir,
        run_root=run_root,
        live=LiveReporter(spec, live_url, live_token),
        mirror=mirror,
    )
