"""The challenger queues: one per field, one entry per submission key, rewritten atomically."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .ids import ModelRef
from .store.records import now_iso
from .store.writer import atomic_write_json, read_json


@dataclass
class QueueEntry:
    key: str
    repo: str
    revision: str
    commit_block: int
    duel_size: str | None
    skip_model_config_check: bool
    accepted_at: str
    source: str = ""

    @property
    def ref(self) -> ModelRef:
        return ModelRef(key=self.key, repo=self.repo, revision=self.revision)


@dataclass
class InProgress:
    event_id: str
    challenger: dict[str, str]
    started_at: str


@dataclass
class QueueState:
    entries: list[QueueEntry] = field(default_factory=list)
    in_progress: InProgress | None = None
    block: int = 0


class Queue:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.state = self._load()

    def _load(self) -> QueueState:
        doc = read_json(self.path) or {}
        entries = [
            QueueEntry(**{k: v for k, v in e.items() if k in QueueEntry.__dataclass_fields__})
            for e in doc.get("entries", [])
        ]
        ip = doc.get("in_progress")
        in_progress = InProgress(**ip) if ip else None
        return QueueState(entries=entries, in_progress=in_progress, block=int(doc.get("block", 0)))

    def save(self) -> None:
        atomic_write_json(
            self.path,
            {
                "entries": [asdict(e) for e in self.state.entries],
                "in_progress": asdict(self.state.in_progress) if self.state.in_progress else None,
                "block": self.state.block,
            },
        )

    # ---------------------------------------------------------------- mutations
    def add(
        self,
        repo: str,
        revision: str,
        *,
        duel_size: str | None = None,
        skip_model_config_check: bool = False,
        source: str = "",
    ) -> tuple[QueueEntry, int]:
        ref = ModelRef.make(repo, revision)
        self.state.entries = [e for e in self.state.entries if e.key != ref.key]
        entry = QueueEntry(
            key=ref.key,
            repo=repo,
            revision=revision,
            commit_block=self.state.block,
            duel_size=duel_size,
            skip_model_config_check=skip_model_config_check,
            accepted_at=now_iso(),
            source=source,
        )
        self.state.entries.append(entry)
        self.save()
        return entry, len(self.state.entries)

    def remove(self, key: str) -> bool:
        before = len(self.state.entries)
        self.state.entries = [e for e in self.state.entries if e.key != key]
        self.save()
        return len(self.state.entries) != before

    def peek(self) -> QueueEntry | None:
        return self.state.entries[0] if self.state.entries else None

    def pop(self) -> QueueEntry | None:
        if not self.state.entries:
            return None
        entry = self.state.entries.pop(0)
        self.save()
        return entry

    def start(self, event_id: str, challenger: ModelRef) -> None:
        self.state.in_progress = InProgress(
            event_id=event_id, challenger=challenger.as_dict(), started_at=now_iso()
        )
        self.save()

    def finish(self) -> None:
        self.state.in_progress = None
        self.save()

    def advance_block(self) -> int:
        self.state.block += 1
        self.save()
        return self.state.block

    @property
    def block(self) -> int:
        return self.state.block

    def entries(self) -> list[QueueEntry]:
        return list(self.state.entries)

    # ---------------------------------------------------------------- published view
    def snapshot(self, track: str, king: ModelRef | None, schema: int) -> dict[str, Any]:
        return {
            "schema": schema,
            "track": track,
            "block": self.state.block,
            "written_at": now_iso(),
            "king": king.as_dict() if king else None,
            "in_progress": asdict(self.state.in_progress) if self.state.in_progress else None,
            "entries": [
                {
                    "position": i + 1,
                    "key": e.key,
                    "repo": e.repo,
                    "revision": e.revision,
                    "commit_block": e.commit_block,
                    "duel_size": e.duel_size,
                    "skip_model_config_check": e.skip_model_config_check,
                    "accepted_at": e.accepted_at,
                }
                for i, e in enumerate(self.state.entries)
            ],
        }


class Queues:
    """One queue per field, as one file each in a directory.

    A field's queue is its own file because its lineage is its own: the block counter advances
    with that field's events, and a busy field must not hold up another's entries. The v3 layout
    was a single `queue.json`; an operator moves it to `<dir>/<track>.json` once (see
    `docs/operations.md`), and pointing at the old file says so rather than starting empty.
    """

    def __init__(self, root: str | Path, tracks: Sequence[str]):
        self.root = Path(root)
        if self.root.is_file():
            raise ValueError(
                f"{self.root} is a file: the queue is a directory with one file per field. "
                f"Move it to {self.root.with_suffix('')}/{tracks[0]}.json and pass that directory."
            )
        self.root.mkdir(parents=True, exist_ok=True)
        self._queues = {t: Queue(self.root / f"{t}.json") for t in tracks}

    @property
    def tracks(self) -> tuple[str, ...]:
        return tuple(self._queues)

    def __getitem__(self, track: str) -> Queue:
        try:
            return self._queues[track]
        except KeyError:
            raise KeyError(
                f"unknown track {track!r}; the fields are {', '.join(self._queues)}"
            ) from None

    def __contains__(self, track: object) -> bool:
        return track in self._queues

    def items(self):
        return self._queues.items()
