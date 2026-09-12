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
TRACK_CODE_RE = re.compile(r"^[a-z]{2}$")
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

#: How a field stops a model from simply replaying the demonstration it was shown.
PROTOCOLS = ("different_initial_state", "same_initial_state")

#: What a field's policies may see of a demonstration.
DEMO_VIEWS = ("sensorimotor", "video_only")

#: Where a field's prompts come from. "pool" is published up front; "materialized" is
#: produced per duel and published with the event, which is what a field must use when the
#: demonstration is the answer for the very scene it is scored on.
PROMPT_SOURCES = ("pool", "materialized")


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
    # A v3 spec read by v5 code would render the wrong numbers silently, so say so loudly.
    need("track removed (v3); use tracks", "track" not in doc)
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
    tracks = doc.get("tracks") or {}
    need("tracks non-empty", isinstance(tracks, dict) and bool(tracks))
    benchmarks = doc.get("benchmarks") or {}
    claimed: list[str] = []
    slugs: set[str] = set()
    track_codes: set[str] = set()
    for tid, t in tracks.items():
        if not isinstance(t, dict):
            errors.append(f"tracks.{tid}: mapping")
            continue
        need(f"tracks.{tid}.id == key", t.get("id") == tid)
        need(
            f"tracks.{tid}.code",
            isinstance(t.get("code"), str) and bool(TRACK_CODE_RE.match(str(t.get("code")))),
        )
        need(f"tracks.{tid}.code unique", t.get("code") not in track_codes)
        track_codes.add(str(t.get("code")))
        need(f"tracks.{tid}.slug", bool(SLUG_RE.match(str(t.get("slug", "")))))
        need(f"tracks.{tid}.slug unique", t.get("slug") not in slugs)
        slugs.add(str(t.get("slug")))
        for key in ("short", "title", "blurb"):
            need(f"tracks.{tid}.{key}", isinstance(t.get(key), str) and bool(t.get(key)))
        need(f"tracks.{tid}.k_demos==1", t.get("k_demos") == 1)
        need(f"tracks.{tid}.language==none", t.get("language") == "none")
        need(f"tracks.{tid}.protocol", t.get("protocol") in PROTOCOLS)
        need(f"tracks.{tid}.prompts", t.get("prompts") in PROMPT_SOURCES)
        need(
            f"tracks.{tid}.prompt_instance_disjoint",
            isinstance(t.get("prompt_instance_disjoint"), bool),
        )
        # Same Scene shows the demonstration of the very state it scores; a field that does not
        # is the one that needs the disjointness. Getting this pair backwards is the mistake the
        # whole two-field design exists to make impossible.
        need(
            f"tracks.{tid}.prompt_instance_disjoint matches protocol",
            t.get("prompt_instance_disjoint") == (t.get("protocol") == "different_initial_state"),
        )
        demo = t.get("demonstration") or {}
        need(f"tracks.{tid}.demonstration.view", demo.get("view") in DEMO_VIEWS)
        need(
            f"tracks.{tid}.demonstration.modalities has video",
            isinstance(demo.get("modalities"), list) and "video" in (demo.get("modalities") or []),
        )
        need(f"tracks.{tid}.demonstration.withheld", isinstance(demo.get("withheld"), list))
        track_skills = t.get("skills")
        need(
            f"tracks.{tid}.skills non-empty", isinstance(track_skills, list) and bool(track_skills)
        )
        for sid in track_skills or []:
            need(f"tracks.{tid}.skills.{sid} exists", sid in skills)
            claimed.append(sid)
        own_sizes = t.get("sizes")
        if own_sizes is not None:
            need(
                f"tracks.{tid}.sizes names match duel.sizes",
                isinstance(own_sizes, dict) and set(own_sizes) == set(sizes),
            )
            for name, entry in (own_sizes or {}).items():
                n = (entry or {}).get("units_per_skill")
                need(f"tracks.{tid}.sizes.{name}.units_per_skill>=1", isinstance(n, int) and n >= 1)
        need(
            f"tracks.{tid}.default_size in sizes",
            t.get("default_size") in (own_sizes if own_sizes is not None else sizes),
        )
        for sid in track_skills or []:
            name = (skills.get(sid) or {}).get("simulator")
            if isinstance(name, str) and name not in benchmarks:
                errors.append(f"benchmarks.{name} undeclared (skills.{sid}.simulator)")
    # Every skill belongs to exactly one field: one scored twice would need two architectures,
    # and one scored nowhere would sit in the contract affecting nothing.
    need("tracks partition the skills", sorted(claimed) == sorted(skills))
    need("tracks claim no skill twice", len(claimed) == len(set(claimed)))
    for name, entry in benchmarks.items():
        need(f"benchmarks.{name} mapping", isinstance(entry, dict))
        need(
            f"benchmarks.{name}.distribution",
            isinstance((entry or {}).get("distribution"), str)
            and bool((entry or {}).get("distribution")),
        )
    need("baselines", isinstance(doc.get("baselines"), dict))
    for tid in tracks:
        need(f"baselines.{tid}", tid in (doc.get("baselines") or {}))
        need(f"pools.tracks.{tid}", tid in ((doc.get("pools") or {}).get("tracks") or {}))
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

    # -- tracks
    @property
    def version(self) -> int:
        return int(self.raw["spec_version"])

    @property
    def tracks(self) -> tuple[str, ...]:
        """The fields of the competition, in declaration order."""
        return tuple(self.raw["tracks"].keys())

    def track(self, track: str) -> dict[str, Any]:
        return self.raw["tracks"][track]

    @property
    def sole_track(self) -> str:
        """The only field, for a caller that has not been given one yet.

        Transitional. Every remaining use is a place the track still has to be threaded through
        (#42), and this raises rather than guessing the moment a second field is declared - so
        the second field cannot quietly be scored as the first.
        """
        tracks = self.tracks
        if len(tracks) != 1:
            raise ValueError(
                f"this call site still assumes one track, but the spec declares {len(tracks)}: "
                f"{', '.join(tracks)}. Thread the track through instead."
            )
        return tracks[0]

    def track_title(self, track: str) -> str:
        return str(self.track(track)["title"])

    def track_of(self, skill: str) -> str:
        """The field a skill is scored in. A skill belongs to exactly one, which `validate_spec`
        holds to, so this is total for any skill in the contract."""
        for tid in self.tracks:
            if skill in self.track(tid)["skills"]:
                return tid
        raise KeyError(f"skill {skill!r} belongs to no track")

    def demo_view(self, track: str) -> str:
        """What this field's policies may see of a demonstration."""
        return str(self.track(track)["demonstration"]["view"])

    def withheld(self, track: str) -> tuple[str, ...]:
        """The demonstration arrays this field keeps from its policies."""
        return tuple(self.track(track)["demonstration"].get("withheld", ()))

    def protocol(self, track: str) -> str:
        return str(self.track(track)["protocol"])

    def prompts(self, track: str) -> str:
        """Where this field's prompts come from: a published pool, or materialized per duel."""
        return str(self.track(track)["prompts"])

    def prompt_instance_disjoint(self, track: str) -> bool:
        """Whether the demonstration must start somewhere other than the scored state."""
        return bool(self.track(track)["prompt_instance_disjoint"])

    def architectures(self, track: str) -> tuple[str, ...]:
        """Derived, never stored: no contract value is duplicated."""
        return tuple(dict.fromkeys(self.architecture(s) for s in self.skills(track)))

    def simulators(self, track: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.simulator(s) for s in self.skills(track)))

    # -- skills
    @property
    def all_skills(self) -> tuple[str, ...]:
        """Every skill in the contract, in declaration order.

        Named `all_` because with more than one field "the skills" is ambiguous: a duel scores
        one field's skills, not the contract's. `Spec.skills(track)` is that other reading.
        """
        return tuple(self.raw["skills"].keys())

    def skills(self, track: str) -> tuple[str, ...]:
        """The skills one field scores, in the order they are run, rendered and averaged."""
        return tuple(self.track(track)["skills"])

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

    def _duel_of(self, track: str, key: str) -> Any:
        """A field's own value for a duelling constant, or the competition-wide default.

        A RoboTwin unit and a LIBERO unit are not the same amount of work, so sizes and
        tolerances have to be per field; the margin is per field for the same reason a crown is.
        """
        own = self.track(track).get(key)
        return self.raw["duel"][key] if own is None else own

    def sizes(self, track: str) -> tuple[str, ...]:
        return tuple(self._duel_of(track, "sizes").keys())

    def default_size(self, track: str) -> str:
        return str(self.track(track)["default_size"])

    def size_of(self, track: str, size: str | None) -> str:
        return size if size in self._duel_of(track, "sizes") else self.default_size(track)

    def units_per_skill(self, track: str, size: str | None = None) -> int:
        sizes = self._duel_of(track, "sizes")
        return int(sizes[self.size_of(track, size)]["units_per_skill"])

    def units_per_side(self, track: str, size: str | None = None) -> int:
        """Units one side of a duel runs: this field's skills, not the contract's."""
        return self.units_per_skill(track, size) * len(self.skills(track))

    def score_margin(self, track: str) -> float:
        return float(self._duel_of(track, "score_margin"))

    def max_void_fraction(self, track: str) -> float:
        return float(self._duel_of(track, "max_void_fraction"))

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

    def pool_pin(self, track: str) -> dict[str, Any]:
        """The pool version and id this field is pinned to. Empty for a field whose prompts are
        materialized per duel rather than drawn from a published pool."""
        return dict((self.raw["pools"].get("tracks") or {}).get(track) or {})

    def baseline(self, track: str) -> dict[str, Any] | None:
        """This field's genesis king, or None where it has none yet and opens on an empty
        throne - which the daemon already handles by crowning its first entrant."""
        entry = (self.raw.get("baselines") or {}).get(track)
        return dict(entry) if isinstance(entry, dict) else None


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
