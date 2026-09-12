"""The contract: `spec.json`, read once, validated, never duplicated as literals."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .canon import canonical_sha256

SPEC_ENV = "ICILVAL_SPEC"
SCHEMA_ENV = "ICILVAL_STORE_SCHEMA"
SKILL_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
SKILL_CODE_RE = re.compile(r"^[a-z]{2}$")


def _repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "spec.json").exists():
            return parent
    return None


def _resolve(name: str, env: str) -> Path:
    explicit = os.environ.get(env)
    if explicit:
        return Path(explicit)
    packaged = Path(__file__).resolve().parent / "data" / name
    if packaged.exists():
        return packaged
    root = _repo_root()
    if root and (root / name).exists():
        return root / name
    raise FileNotFoundError(f"{name} not found; set {env}")


def spec_path() -> Path:
    return _resolve("spec.json", SPEC_ENV)


def schema_path() -> Path:
    return _resolve("store-schema.json", SCHEMA_ENV)


def validate_spec(doc: dict[str, Any]) -> list[str]:
    from .simulators import get as simulator
    from .simulators import names as simulator_names

    errors: list[str] = []

    def need(path: str, cond: bool) -> None:
        if not cond:
            errors.append(path)

    need("spec_version:int", isinstance(doc.get("spec_version"), int))
    track = doc.get("track") or {}
    need("track.id", isinstance(track.get("id"), str) and bool(track.get("id")))
    need("track.k_demos==1", track.get("k_demos") == 1)
    need("track.language==none", track.get("language") == "none")
    skills = doc.get("skills") or {}
    need("skills non-empty", isinstance(skills, dict) and bool(skills))
    codes: set[str] = set()
    for sid, s in skills.items():
        need(f"skills.{sid} id", bool(SKILL_ID_RE.match(sid)))
        if not isinstance(s, dict):
            errors.append(f"skills.{sid}: mapping")
            continue
        code = s.get("code")
        need(f"skills.{sid}.code", isinstance(code, str) and bool(SKILL_CODE_RE.match(code)))
        need(f"skills.{sid}.code unique", code not in codes)
        codes.add(str(code))
        need(f"skills.{sid}.title", isinstance(s.get("title"), str) and bool(s.get("title")))
        need(f"skills.{sid}.architecture", isinstance(s.get("architecture"), str))
        need(f"skills.{sid}.simulator", s.get("simulator") in simulator_names())
        need(f"skills.{sid}.max_steps", isinstance(s.get("max_steps"), int) and s["max_steps"] > 0)
        need(f"skills.{sid}.environment", isinstance(s.get("environment"), dict))
        env = s.get("environment") or {}
        for key in ("obs_history", "action_horizon", "exec_horizon", "prompt_actions_per_chunk"):
            need(f"skills.{sid}.environment.{key}", isinstance(env.get(key), int) and env[key] > 0)
        need(f"skills.{sid}.perturbations removed", "perturbations" not in s)
        if s.get("simulator") in simulator_names():
            errors.extend(simulator(str(s["simulator"])).validate_skill(sid, s))
        tasks = s.get("tasks") or {}
        need(
            f"skills.{sid}.tasks.dataset",
            isinstance(tasks.get("dataset"), str) and "/" in tasks["dataset"],
        )
        need(
            f"skills.{sid}.tasks.kind",
            isinstance(tasks.get("kind"), str) and bool(tasks.get("kind")),
        )
        need(
            f"skills.{sid}.tasks views|files",
            (isinstance(tasks.get("views"), list) and bool(tasks["views"]))
            or (isinstance(tasks.get("files"), dict) and bool(tasks["files"])),
        )
    duel = doc.get("duel") or {}
    sizes = duel.get("sizes") or {}
    need("duel.default_size in sizes", duel.get("default_size") in sizes)
    for name, s in sizes.items():
        n = s.get("units_per_skill")
        need(f"duel.sizes.{name}.units_per_skill>=1", isinstance(n, int) and n >= 1)
    margin = duel.get("score_margin")
    need("duel.score_margin in [0,100]", isinstance(margin, (int, float)) and 0 <= margin <= 100)
    void = duel.get("max_void_fraction")
    need("duel.max_void_fraction in [0,1]", isinstance(void, (int, float)) and 0 <= void <= 1)
    store = doc.get("store") or {}
    need(
        "store.index_lines_per_part>=1",
        isinstance(store.get("index_lines_per_part"), int) and store["index_lines_per_part"] >= 1,
    )
    need("store.media_bucket_hex in 1..4", store.get("media_bucket_hex") in (1, 2, 3, 4))
    model = doc.get("model") or {}
    need(
        "model.required_files",
        isinstance(model.get("required_files"), list) and model["required_files"],
    )
    need("model.max_repo_bytes", isinstance(model.get("max_repo_bytes"), int))
    return errors


@dataclass(frozen=True)
class Spec:
    raw: dict[str, Any]
    path: Path
    fingerprint: str

    # -- track
    @property
    def version(self) -> int:
        return int(self.raw["spec_version"])

    @property
    def track_id(self) -> str:
        return str(self.raw["track"]["id"])

    @property
    def track(self) -> dict[str, Any]:
        return self.raw["track"]

    # -- skills
    @property
    def all_skills(self) -> tuple[str, ...]:
        """Every skill in the contract, in declaration order.

        Named `all_` because with more than one field "the skills" is ambiguous: a duel scores
        one field's skills, not the contract's. `Spec.skills(track)` is that other reading.
        """
        return tuple(self.raw["skills"].keys())

    def skill(self, name: str) -> dict[str, Any]:
        return self.raw["skills"][name]

    def skill_code(self, name: str) -> str:
        return str(self.skill(name)["code"])

    def skill_for_code(self, code: str) -> str:
        for s in self.all_skills:
            if self.skill_code(s) == code:
                return s
        raise KeyError(code)

    def skill_title(self, name: str) -> str:
        return str(self.skill(name)["title"])

    def architecture(self, name: str) -> str:
        return str(self.skill(name)["architecture"])

    def simulator(self, name: str) -> str:
        return str(self.skill(name)["simulator"])

    def max_steps(self, name: str) -> int:
        return int(self.skill(name)["max_steps"])

    def env(self, name: str) -> dict[str, Any]:
        return self.skill(name)["environment"]

    def tasks(self, name: str) -> dict[str, Any]:
        """Where a skill's tasks come from: a Hugging Face dataset and its views or files."""
        return self.skill(name)["tasks"]

    def success(self, name: str) -> dict[str, Any] | None:
        return self.skill(name).get("success")

    # -- duel
    @property
    def duel(self) -> dict[str, Any]:
        return self.raw["duel"]

    @property
    def default_size(self) -> str:
        return str(self.raw["duel"]["default_size"])

    @property
    def sizes(self) -> tuple[str, ...]:
        return tuple(self.raw["duel"]["sizes"].keys())

    def size_of(self, size: str | None) -> str:
        return size if size in self.raw["duel"]["sizes"] else self.default_size

    def units_per_skill(self, size: str | None = None) -> int:
        return int(self.raw["duel"]["sizes"][self.size_of(size)]["units_per_skill"])

    def units_per_duel(self, size: str | None = None) -> int:
        return self.units_per_skill(size) * len(self.all_skills)

    @property
    def score_margin(self) -> float:
        return float(self.raw["duel"]["score_margin"])

    @property
    def max_void_fraction(self) -> float:
        return float(self.raw["duel"]["max_void_fraction"])

    # -- the rest
    @property
    def budgets(self) -> dict[str, Any]:
        return self.raw["budgets"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def media(self) -> dict[str, Any]:
        return self.raw["media"]

    @property
    def store(self) -> dict[str, Any]:
        return self.raw["store"]

    @property
    def live(self) -> dict[str, Any]:
        return self.raw["live"]

    @property
    def admin(self) -> dict[str, Any]:
        return self.raw["admin"]

    @property
    def pools(self) -> dict[str, Any]:
        return self.raw["pools"]

    @property
    def baseline(self) -> dict[str, Any]:
        return self.raw["baseline"]


def load_spec_file(path: str | Path) -> Spec:
    p = Path(path)
    doc = json.loads(p.read_text())
    errors = validate_spec(doc)
    if errors:
        raise ValueError(f"{p}: invalid spec: " + ", ".join(errors))
    return Spec(raw=doc, path=p, fingerprint=canonical_sha256(doc))


@lru_cache(maxsize=4)
def _cached(path: str) -> Spec:
    return load_spec_file(path)


def load_spec(path: str | Path | None = None) -> Spec:
    return _cached(str(Path(path) if path else spec_path()))


def load_schema(path: str | Path | None = None) -> dict[str, Any]:
    return json.loads(Path(path or schema_path()).read_text())
