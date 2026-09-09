"""Live progress frames: ephemeral, best-effort, never part of the record."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from .spec import Spec
from .store.records import now_iso

log = logging.getLogger(__name__)

PHASES = ("fetching", "checking", "materializing", "evaluating", "publishing", "done", "failed")


def build_frame(
    spec: Spec,
    *,
    validator_key: str,
    event_id: str,
    kind: str,
    duel_size: str | None,
    king: dict[str, str] | None,
    challenger: dict[str, str] | None,
    phase: str,
    side: str | None,
    units: list[dict[str, Any]],
    current: dict[str, Any] | None,
    recent_media: dict[str, Any] | None,
    message: str,
    started_at: str,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    per_skill: dict[str, dict[str, dict[str, int]]] = {s: {} for s in ("challenger", "king")}
    for s in per_skill:
        for skill in spec.skills:
            skill_units = [u for u in units if u.get("skill") == skill]
            done = sum(
                1 for u in skill_units if isinstance(u.get(f"{s}_success"), bool) or u.get("void")
            )
            per_skill[s][skill] = {"done": done, "total": len(skill_units)}
    done = sum(v["done"] for v in per_skill[side].values()) if side in per_skill else 0
    live_units = [
        {
            "unit_id": u.get("unit_id"),
            "skill": u.get("skill"),
            "task": u.get("task"),
            "task_label": u.get("task_label"),
            "instance": u.get("instance"),
            "king_success": u.get("king_success"),
            "challenger_success": u.get("challenger_success"),
            "outcome": u.get("outcome")
            if isinstance(u.get("king_success"), bool)
            and isinstance(u.get("challenger_success"), bool)
            else None,
            "demo_video": u.get("demo_video"),
            "king_video": u.get("king_video"),
            "challenger_video": u.get("challenger_video"),
        }
        for u in units
    ]
    return {
        "schema": int(spec.live["schema"]),
        "validator_key": validator_key,
        "track": spec.track_id,
        "event_id": event_id,
        "kind": kind,
        "duel_size": duel_size,
        "king": king,
        "challenger": challenger,
        "phase": phase,
        "side": side,
        "done": done,
        "total": len(units),
        "skill_progress": per_skill,
        "current": current,
        "recent_media": recent_media,
        "units": live_units,
        "message": message[:300],
        "started_at": started_at,
        "sent_at": now_iso(),
    }


class LiveReporter:
    """POSTs frames to the dashboard. Failures are logged and otherwise ignored."""

    def __init__(self, spec: Spec, url: str | None, token: str | None, *, timeout_s: float = 5.0):
        self.spec = spec
        self.url = (url.rstrip("/") + str(spec.live["path"])) if url else None
        self.token = token
        self.timeout_s = timeout_s
        self.min_interval_s = float(spec.live.get("min_interval_s", 1.0))
        self.max_bytes = int(spec.live["max_frame_bytes"])
        self._last_sent = 0.0
        self.sent = 0
        self.failed = 0

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.token)

    def post(self, frame: dict[str, Any], *, force: bool = False) -> bool:
        if not self.enabled:
            return False
        now = time.monotonic()
        if not force and now - self._last_sent < self.min_interval_s:
            return False
        data = json.dumps(frame).encode("utf-8")
        if len(data) > self.max_bytes:
            slim = dict(frame)
            slim["units"] = []
            data = json.dumps(slim).encode("utf-8")
        req = urllib.request.Request(
            self.url,  # type: ignore[arg-type]
            data=data,
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310 - fixed https url from config
                resp.read()
            self._last_sent = now
            self.sent += 1
            return True
        except (urllib.error.URLError, OSError, ValueError) as exc:
            self.failed += 1
            log.warning("live frame not delivered: %s", exc)
            return False
