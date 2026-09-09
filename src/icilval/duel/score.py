"""Scoring and the crown rule. Pure functions over unit verdict dicts.

Scores are fractions in [0, 1]: one success rate per skill and their mean.
`score_margin` arrives in percentage points and is divided by 100 here and
nowhere else. The skill list comes from the spec so nothing here names one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

SIDES = ("challenger", "king")
SCORE_EPSILON = 1e-9


def paired_outcome(king_success: bool | None, challenger_success: bool | None) -> str:
    if king_success is None or challenger_success is None:
        return "tie"
    if king_success == challenger_success:
        return "tie"
    return "challenger" if challenger_success else "king"


def side_success(unit: dict[str, Any], side: str) -> bool | None:
    value = unit.get(f"{side}_success")
    return value if isinstance(value, bool) else None


def skill_rate(units: Iterable[dict[str, Any]], side: str, skill: str) -> float | None:
    scored = 0
    successes = 0
    for u in units:
        if u.get("skill") != skill or u.get("void") or u.get("diagnostic"):
            continue
        s = side_success(u, side)
        if s is None:
            continue
        scored += 1
        successes += int(s)
    return successes / scored if scored else None


def lookup(unit: dict[str, Any], path: str) -> Any:
    """A dotted path into a unit verdict (`change.kind`, `instance_params.family`)."""
    cur: Any = unit
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def sub_scores(
    units: Iterable[dict[str, Any]], side: str, keys: dict[str, str]
) -> dict[str, dict[str, float]]:
    """Per skill, the success rate per value of the skill's sub-score key (`Spec.sub_score_key`):
    a change kind, a drawing family. Published on the record, never part of a score."""
    units = list(units)
    out: dict[str, dict[str, float]] = {}
    for skill, key in keys.items():
        groups: dict[str, list[int]] = {}
        for u in units:
            if u.get("skill") != skill or u.get("void") or u.get("diagnostic"):
                continue
            s = side_success(u, side)
            if s is None:
                continue
            value = lookup(u, key)
            groups.setdefault("none" if value is None else str(value), []).append(int(s))
        out[skill] = {g: sum(v) / len(v) for g, v in sorted(groups.items())}
    return out


def diagnostic_rates(
    units: Iterable[dict[str, Any]], side: str, diagnostics: dict[str, dict[str, Any]]
) -> dict[str, float | None]:
    """The success rate of each unscored diagnostic's units (`Spec.diagnostics`), found by the
    task group the diagnostic names."""
    units = list(units)
    out: dict[str, float | None] = {}
    for name, diag in diagnostics.items():
        prefix = f"{diag['group']}/"
        scored = [
            side_success(u, side)
            for u in units
            if u.get("diagnostic")
            and str(u.get("task", "")).startswith(prefix)
            and not u.get("void")
            and side_success(u, side) is not None
        ]
        out[name] = sum(int(bool(s)) for s in scored) / len(scored) if scored else None
    return out


def average(per_skill: dict[str, float | None], skills: Sequence[str]) -> float | None:
    present = [per_skill[s] for s in skills if per_skill.get(s) is not None]
    return sum(present) / len(present) if present else None


def skill_scores(
    units: Iterable[dict[str, Any]], side: str, skills: Sequence[str]
) -> dict[str, float | None]:
    units = list(units)
    per = {s: skill_rate(units, side, s) for s in skills}
    per["average"] = average(per, skills)
    return per


def empty_scores(skills: Sequence[str]) -> dict[str, float | None]:
    return {**{s: None for s in skills}, "average": None}


def crown_moves(
    king_average: float | None, challenger_average: float | None, margin_points: float
) -> bool:
    if king_average is None or challenger_average is None:
        return False
    return challenger_average >= king_average + margin_points / 100.0 - SCORE_EPSILON


@dataclass
class Tally:
    units: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    decided: int = 0
    void: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "decided": self.decided,
            "void": self.void,
        }


def tally(units: Iterable[dict[str, Any]]) -> Tally:
    t = Tally()
    for u in units:
        t.units += 1
        if u.get("void"):
            t.void += 1
            continue
        outcome = u.get("outcome") or paired_outcome(
            side_success(u, "king"), side_success(u, "challenger")
        )
        if outcome == "challenger":
            t.wins += 1
            t.decided += 1
        elif outcome == "king":
            t.losses += 1
            t.decided += 1
        else:
            t.ties += 1
    return t


@dataclass
class Verdict:
    king_scores: dict[str, float | None]
    challenger_scores: dict[str, float | None]
    score_margin: float
    dethroned: bool
    reason: str
    tally: Tally = field(default_factory=Tally)

    @property
    def delta_points(self) -> float | None:
        k, c = self.king_scores["average"], self.challenger_scores["average"]
        return None if k is None or c is None else (c - k) * 100.0


def verdict(units: Iterable[dict[str, Any]], score_margin: float, skills: Sequence[str]) -> Verdict:
    units = list(units)
    king = skill_scores(units, "king", skills)
    challenger = skill_scores(units, "challenger", skills)
    moves = crown_moves(king["average"], challenger["average"], score_margin)
    if not units:
        reason = "no-units"
    elif king["average"] is None or challenger["average"] is None:
        reason = "unscored"
    else:
        reason = "margin-met" if moves else "short-of-margin"
    return Verdict(king, challenger, score_margin, moves, reason, tally(units))


def void_fraction(units: Iterable[dict[str, Any]]) -> float:
    units = list(units)
    return sum(1 for u in units if u.get("void")) / len(units) if units else 0.0
