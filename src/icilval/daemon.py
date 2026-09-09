"""The validator loop: queue -> duel -> publish -> mirror, forever. One instance per store."""

from __future__ import annotations

import fcntl
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canon import Signer
from .duel.orchestrate import DuelFailed, DuelRequest, Orchestrator, Runtime
from .ids import ModelRef
from .queue import Queue
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
    def __init__(self, rt: Runtime, queue: Queue, cfg: DaemonConfig):
        self.rt = rt
        self.queue = queue
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

    def current_king(self) -> ModelRef | None:
        head = self.rt.store.head(self.rt.spec.track_id)
        return ModelRef.from_dict(head.get("king")) if head else None

    def publish_queue(self) -> None:
        snap = self.queue.snapshot(
            self.rt.spec.track_id, self.current_king(), int(self.rt.spec.store["schema"])
        )
        self.rt.store.write_queue(self.rt.spec.track_id, snap)
        self._mirror(self.rt.store.drain_touched())

    def _mirror(self, files: list[str]) -> None:
        if self.rt.mirror is None or not files:
            return
        try:
            self.rt.mirror.push(files)
        except Exception as exc:  # noqa: BLE001
            log.warning("mirror failed (will retry next publish): %s", exc)

    def step(self) -> bool:
        """Run one queue entry. Returns False when the queue is empty."""
        entry = self.queue.peek()
        if entry is None:
            return False
        king = self.current_king()
        if king is None:
            log.info("empty throne: genesis for %s", entry.ref.entry)
            from .duel.orchestrate import publish_genesis

            block = self.queue.advance_block()
            self.queue.pop()
            try:
                publish_genesis(
                    self.rt,
                    entry.ref,
                    block,
                    local_models=self.cfg.local_models,
                    check=not entry.skip_model_config_check,
                )
            except DuelFailed as exc:
                log.error("genesis refused: %s", exc)
            self.publish_queue()
            self._mirror(self.rt.store.drain_touched())
            return True
        if king.key == entry.key:
            log.info("%s already holds the crown; dropping the entry", entry.ref.entry)
            self.queue.pop()
            self.publish_queue()
            return True
        block = self.queue.advance_block()
        req = DuelRequest(
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
            self.rt.spec.track_id,
            block,
            duel_id(self.rt.spec.version, self.rt.spec.track_id, entry.ref, king),
        )
        self.queue.pop()
        self.queue.start(eid, entry.ref)
        self.publish_queue()
        try:
            self.orchestrator.run(req, block)
        except DuelFailed:
            pass
        finally:
            self.queue.finish()
            self.publish_queue()
            self._mirror(self.rt.store.drain_touched())
        return True

    def run(self) -> None:
        self.lock()
        log.info(
            "validator %s watching %s", self.rt.signer.verify_key_hex[:12], self.cfg.queue_path
        )
        self.publish_queue()
        while True:
            try:
                ran = self.step()
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
    bpp_root: Path | None = None,
    raw_root: Path | None = None,
    workers: int | None = None,
) -> Runtime:
    """`enforce_pool_id`: the daemon and the publishing commands refuse a catalogue other than the
    one pinned in spec.json; the smoke test runs a small local catalogue and turns the check off.
    `bpp_root` (the vendored BPP checkout) and `raw_root` (the raw cache with the grasp-source
    files) are what prompt generation needs; without `bpp_root` a duel cannot materialize."""
    from .duel.materialize import GenerationContext
    from .live import LiveReporter
    from .pools.schema import Pool
    from .pools.sources import DEFAULT_CACHE, Sources

    signer = Signer.from_file(key_file)
    pool = Pool.load(pool_dir)
    pinned = spec.catalogue.get("pool_id")
    if enforce_pool_id and pinned and pinned != pool.pool_id:
        raise RuntimeError(
            f"catalogue {pool.pool_id} does not match spec catalogue.pool_id {pinned}"
        )
    store = Store(store_root, spec, signer)
    if store.manifest() is None:
        store.init(signer.verify_key_hex, pool.pool_id)
    mirror: Any | None = None
    if mirror_repo:
        from .store.mirror import Mirror

        mirror = Mirror(store_root, mirror_repo, token=hf_token)
    generation: GenerationContext | None = None
    if bpp_root is not None:
        raw = raw_root or DEFAULT_CACHE / "raw"
        datasets = raw
        for skill in spec.skills:
            grasp = spec.tasks(skill).get("grasp_sources")
            if grasp:
                datasets = Sources(raw=raw, bpp_root=bpp_root).dataset_root(str(grasp["dataset"]))
                break
        generation = GenerationContext(bpp_root=bpp_root, raw_root=raw, libero_datasets=datasets)
    return Runtime(
        spec=spec,
        pool=pool,
        store=store,
        signer=signer,
        arch_dir=arch_dir,
        run_root=run_root,
        live=LiveReporter(spec, live_url, live_token),
        mirror=mirror,
        generation=generation,
        workers=int(workers or spec.generation["workers"]),
    )
