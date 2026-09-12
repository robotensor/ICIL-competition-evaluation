"""The signed, append-only result store.

Layout (spec.store):
  manifest.json
  tracks/{track}/head.json
  tracks/{track}/queue.json
  tracks/{track}/index-NNNN.jsonl     one '<canonical_json>\t<signature_hex>' per line
  events/{track}/{event_id}.json
  media/{sha[:k]}/{sha}.{ext}

Every write is atomic (tmp + rename) except the index append, which is a single
O_APPEND write followed by fsync so a concurrent reader sees at most one torn
final line, which the dashboard tolerates.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..canon import Signer, canonical_json, sha256_file
from ..spec import Spec

SHA_LEN = 64


@contextmanager
def store_lock(root: str | Path):
    """One writer per store. Publishing commands and the daemon take this; a second writer fails fast."""
    path = Path(root) / ".validator.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"another validator is publishing to {root} (holds {path})") from exc
        yield
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any, pretty: bool = True) -> None:
    text = json.dumps(obj, indent=2 if pretty else None, sort_keys=True) + "\n"
    atomic_write_text(path, text)


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class Store:
    def __init__(self, root: str | Path, spec: Spec, signer: Signer | None = None):
        self.root = Path(root)
        self.spec = spec
        self.signer = signer
        self.touched: set[str] = set()

    # ---------------------------------------------------------------- paths
    def track_dir(self, track: str) -> Path:
        return self.root / "tracks" / track

    def index_part_path(self, track: str, part: int) -> Path:
        return self.track_dir(track) / f"index-{part:04d}.jsonl"

    def head_path(self, track: str) -> Path:
        return self.track_dir(track) / "head.json"

    def queue_path(self, track: str) -> Path:
        return self.track_dir(track) / "queue.json"

    def event_path(self, track: str, event_id: str) -> Path:
        return self.root / "events" / track / f"{event_id}.json"

    def media_path(self, sha: str, ext: str) -> Path:
        bucket = sha[: int(self.spec.store["media_bucket_hex"])]
        return self.root / "media" / bucket / f"{sha}.{ext}"

    def _touch(self, path: Path) -> None:
        self.touched.add(str(path.relative_to(self.root)))

    # ---------------------------------------------------------------- manifest
    def init(self, validator_key: str, pool_id: str | None) -> dict[str, Any]:
        manifest = {
            "schema": int(self.spec.store["schema"]),
            "validator_key": validator_key,
            "spec_version": self.spec.version,
            "spec_fingerprint": self.spec.fingerprint,
            "pool_id": pool_id,
            "tracks": [self.spec.sole_track],
        }
        atomic_write_json(self.root / "manifest.json", manifest)
        self._touch(self.root / "manifest.json")
        track = self.spec.sole_track
        if not self.head_path(track).exists():
            self.write_head(track, seq=0, event_id="", block=0, finished_at="", king=None)
        return manifest

    def manifest(self) -> dict[str, Any] | None:
        return read_json(self.root / "manifest.json")

    # ---------------------------------------------------------------- head / index
    def head(self, track: str) -> dict[str, Any] | None:
        return read_json(self.head_path(track))

    def write_head(
        self,
        track: str,
        *,
        seq: int,
        event_id: str,
        block: int,
        finished_at: str,
        king: dict | None,
    ) -> None:
        head = {
            "schema": int(self.spec.store["schema"]),
            "seq": seq,
            "event_id": event_id,
            "block": block,
            "finished_at": finished_at,
            "king": king,
        }
        atomic_write_json(self.head_path(track), head)
        self._touch(self.head_path(track))

    def next_seq(self, track: str) -> int:
        head = self.head(track)
        return int(head["seq"]) + 1 if head and isinstance(head.get("seq"), int) else 1

    def part_of(self, seq: int) -> int:
        per = max(1, int(self.spec.store["index_lines_per_part"]))
        return (seq - 1) // per

    def append(self, track: str, record: dict[str, Any]) -> int:
        """Assign the next seq, sign, append to the index and advance the head."""
        if self.signer is None:
            raise RuntimeError("store has no signer")
        seq = self.next_seq(track)
        record = dict(record)
        record["seq"] = seq
        canonical = canonical_json(record)
        line = canonical + "\t" + self.signer.sign(canonical) + "\n"
        path = self.index_part_path(track, self.part_of(seq))
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        self._touch(path)
        king = (
            record.get("new_king")
            if record.get("new_king")
            else (
                record.get("king")
                if record.get("kind") in ("genesis", "succession")
                else self.current_king(track, record)
            )
        )
        self.write_head(
            track,
            seq=seq,
            event_id=str(record["event_id"]),
            block=int(record["block"]),
            finished_at=str(record["finished_at"]),
            king=king,
        )
        return seq

    def current_king(self, track: str, record: dict[str, Any]) -> dict | None:
        if record.get("kind") == "vacancy":
            return None
        if record.get("kind") == "duel":
            return record.get("new_king") if record.get("dethroned") else record.get("king")
        head = self.head(track)
        return head.get("king") if head else None

    def iter_index(self, track: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        part = 0
        while True:
            path = self.index_part_path(track, part)
            if not path.exists():
                break
            for raw in path.read_text(encoding="utf-8").split("\n"):
                if not raw.strip():
                    continue
                body = raw.split("\t", 1)[0]
                try:
                    out.append(json.loads(body))
                except ValueError:
                    continue
            part += 1
        out.sort(key=lambda r: r.get("seq", 0))
        return out

    # ---------------------------------------------------------------- events / queue / media
    def write_event(self, track: str, event: dict[str, Any]) -> Path:
        path = self.event_path(track, str(event["event_id"]))
        atomic_write_json(path, event)
        self._touch(path)
        return path

    def event(self, track: str, event_id: str) -> dict[str, Any] | None:
        return read_json(self.event_path(track, event_id))

    def write_queue(self, track: str, snapshot: dict[str, Any]) -> None:
        atomic_write_json(self.queue_path(track), snapshot)
        self._touch(self.queue_path(track))

    def put_media(self, src: str | Path, ext: str | None = None) -> str:
        src = Path(src)
        ext = ext or src.suffix.lstrip(".")
        sha = sha256_file(src)
        dst = self.media_path(sha, ext)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".tmp")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
        self._touch(dst)
        return sha

    def has_media(self, sha: str, ext: str) -> bool:
        return self.media_path(sha, ext).exists()

    def drain_touched(self) -> list[str]:
        out = sorted(self.touched)
        self.touched.clear()
        return out
