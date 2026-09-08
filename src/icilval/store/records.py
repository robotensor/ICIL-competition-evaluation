"""Constructors for published records. Shapes are formalised in store-schema.json."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from ..ids import ModelRef


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def model_ref_dict(ref: ModelRef | None) -> dict[str, str] | None:
    return ref.as_dict() if ref else None


def index_record(
    *,
    schema: int,
    event_id: str,
    kind: str,
    track: str,
    block: int,
    finished_at: str,
    king: ModelRef | None,
    challenger: ModelRef | None,
    king_scores: dict[str, float | None] | None,
    challenger_scores: dict[str, float | None] | None,
    score_margin: float,
    dethroned: bool,
    new_king: ModelRef | None,
    tally: dict[str, int] | None = None,
    media_count: int = 0,
    duel_size: str | None = None,
    duel_id: str | None = None,
    pool_id: str | None = None,
    sub_scores: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    t = tally or {"wins": 0, "losses": 0, "ties": 0, "decided": 0, "void": 0}
    return {
        "schema": schema,
        "seq": 0,  # assigned by the store on append
        "event_id": event_id,
        "kind": kind,
        "track": track,
        "block": block,
        "finished_at": finished_at,
        "duel_size": duel_size,
        "duel_id": duel_id,
        "pool_id": pool_id,
        "king": model_ref_dict(king),
        "challenger": model_ref_dict(challenger),
        "king_scores": king_scores,
        "challenger_scores": challenger_scores,
        "score_margin": score_margin,
        "dethroned": dethroned,
        "new_king": model_ref_dict(new_king),
        "wins": t["wins"],
        "losses": t["losses"],
        "ties": t["ties"],
        "decided": t["decided"],
        "void": t["void"],
        "media_count": media_count,
        "sub_scores": sub_scores,
        "diagnostics": diagnostics,
    }


def duel_event(
    record: dict[str, Any],
    *,
    spec_version: int,
    spec_fingerprint: str,
    units: list[dict[str, Any]],
    units_per_skill: int,
    started_at: str,
    wall_seconds: float,
    sides: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    event = dict(record)
    event.pop("seq", None)
    event.update(
        {
            "spec_version": spec_version,
            "spec_fingerprint": spec_fingerprint,
            "units_per_skill": units_per_skill,
            "started_at": started_at,
            "wall_seconds": round(float(wall_seconds), 3),
            "sides": sides or {},
            "units": units,
            "notes": notes or [],
        }
    )
    return event


def unit_verdict_from_unit(unit: dict[str, Any]) -> dict[str, Any]:
    """The published shape of a unit before either side has run."""
    return {
        "unit_id": unit["unit_id"],
        "skill": unit["skill"],
        "index": unit["index"],
        "task": unit["task"],
        "task_label": unit.get("task_label", unit["task"]),
        "instance": unit["instance"],
        "seed": unit["seed"],
        "instance_params": dict(unit.get("instance_params") or {}),
        "change": dict(unit.get("change") or {"kind": "none"}),
        "substituted_from": unit.get("substituted_from"),
        "diagnostic": bool(unit.get("diagnostic", False)),
        "prompt": {
            "demo_id": unit["demo"],
            "steps": 0,
            "chunks": 0,
            "sha256": unit.get("prompt_sha256"),
        },
        "demo_video": None,
        "king_video": None,
        "challenger_video": None,
        "king_success": None,
        "challenger_success": None,
        "outcome": "tie",
        "king_progress": None,
        "challenger_progress": None,
        "king_metric": None,
        "challenger_metric": None,
        "king_steps": None,
        "challenger_steps": None,
        "king_error": None,
        "challenger_error": None,
        "void": False,
    }


def media_shas(units: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for u in units:
        for k in ("demo_video", "king_video", "challenger_video"):
            v = u.get(k)
            if isinstance(v, str) and v:
                out.append(v)
    return sorted(set(out))


def empty_skill_scores(skills: Sequence[str]) -> dict[str, float | None]:
    return {**{s: None for s in skills}, "average": None}
