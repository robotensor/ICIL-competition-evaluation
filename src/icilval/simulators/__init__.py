"""The simulator registry.

A skill names its simulator in `spec.json` (`skills.<skill>.simulator`). Everything that is
specific to a simulator - the policy class, the episode loop, the pool stage, how a unit's
instance is set up, how a demonstration is rendered, what its spec entry must carry - lives
behind `Simulator`, one package per simulator under this one. Generic code (the side runner,
unit derivation, the pool builder, the spec validator) looks the simulator up here and never
names one. Adding a simulator is a new package that calls `register` and an import below.

A benchmark that lives in another repository registers the same way, from its own distribution,
found through the `icilval.benchmarks` entry point group (`load_plugins`). The registry is the
one place that knows the difference, and it knows it only as `Simulator.distribution`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

#: Where a benchmark in another repository advertises itself. The entry point's name is the
#: simulator name a skill will carry; its value is a module that registers exactly that name.
ENTRY_POINT_GROUP = "icilval.benchmarks"


@dataclass(frozen=True)
class Simulator:
    name: str
    #: (model_dir, arch_dir, spec, skill, device) -> a loaded-on-demand policy (`PolicyBase`)
    make_policy: Callable[..., Any]
    #: (ctx, skill, policy, pool, units, spec, media_dir, record_video) -> None; runs each unit
    #: and hands `ctx.finish` a record (see `duel.side_runner.SideContext`)
    run_units: Callable[..., None]
    #: (pool, spec, src, skill, *, limit, validate, fetch_missing, evict_demos) -> None
    build_stage: Callable[..., None]
    #: (spec, skill, index, task, seed, rng) -> `pools.units.Unit`
    make_unit: Callable[..., Any]
    #: demo (npz arrays) -> frames for the demonstration clip
    demo_frames: Callable[[dict[str, Any]], list[Any]]
    #: (skill id, its spec.json entry) -> validation errors for what this simulator needs there
    validate_skill: Callable[[str, dict[str, Any]], list[str]]
    #: (pool, `pools.schema.PoolTask`) -> errors `pools.build.verify_pool` cannot check
    #: generically, because reading them needs this simulator's own file formats
    verify_pool_task: Callable[[Any, Any], list[str]] = field(default=lambda pool, task: [])
    #: The distribution providing it, named in the error when it turns out not to be installed
    distribution: str = "icilval"


REGISTRY: dict[str, Simulator] = {}


def register(sim: Simulator) -> Simulator:
    if sim.name in REGISTRY:
        raise ValueError(f"simulator {sim.name!r} registered twice")
    REGISTRY[sim.name] = sim
    return sim


class MissingBenchmark(RuntimeError):
    """A skill names a simulator that no installed distribution provides."""


_LOADED = False


def load_plugins() -> tuple[str, ...]:
    """Import every installed distribution advertising `ENTRY_POINT_GROUP`; return what it added.

    An entry point's name is the simulator name and its value is a module that must register
    exactly that name. A distribution that registers nothing, or something else, raises rather
    than being skipped: a benchmark that half-installs must fail where it is installed, not
    later as an empty score.
    """
    global _LOADED
    from importlib.metadata import entry_points

    added: list[str] = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name in REGISTRY:
            continue
        try:
            ep.load()  # importing it is what registers
        except Exception as exc:  # noqa: BLE001
            raise MissingBenchmark(
                f"the benchmark {ep.name!r} ({ep.value}) is installed but did not import: {exc}"
            ) from exc
        if ep.name not in REGISTRY:
            raise MissingBenchmark(
                f"{ep.value} advertises the simulator {ep.name!r} but registered {sorted(REGISTRY)}"
            )
        added.append(ep.name)
    _LOADED = True
    return tuple(added)


def _ensure_loaded() -> None:
    if not _LOADED:
        load_plugins()


def names() -> tuple[str, ...]:
    _ensure_loaded()
    return tuple(REGISTRY)


def installed(name: str) -> bool:
    _ensure_loaded()
    return name in REGISTRY


def get(name: str) -> Simulator:
    _ensure_loaded()
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown simulator {name!r}; registered: {', '.join(REGISTRY)}") from None


def for_skill(spec: Any, skill: str) -> Simulator:
    return get(spec.simulator(skill))


def require(spec: Any, skills: Any = None) -> None:
    """Raise before any work starts if a skill's benchmark is not installed.

    Called by the orchestrator, genesis and the pool builder. The message names the distribution
    to install, which the spec declares, because "unknown simulator 'robotwin'" tells an operator
    nothing about what to do next.
    """
    _ensure_loaded()
    missing: dict[str, list[str]] = {}
    for skill in skills if skills is not None else spec.skills:
        name = spec.simulator(skill)
        if name not in REGISTRY:
            missing.setdefault(name, []).append(skill)
    if not missing:
        return
    pins = benchmark_pins(spec)
    parts = []
    for name, affected in sorted(missing.items()):
        dist = (pins.get(name) or {}).get("distribution") or "an unknown distribution"
        parts.append(f"{name} (for {', '.join(affected)}): install {dist}")
    raise MissingBenchmark(
        "a benchmark is not installed; each advertises itself through the "
        f"{ENTRY_POINT_GROUP!r} entry point group. " + "; ".join(parts)
    )


def benchmark_pins(spec: Any) -> dict[str, dict[str, Any]]:
    """`spec.json`'s `benchmarks` block: what each simulator's distribution must be.

    Absent until the spec declares it, so this returns an empty mapping rather than raising -
    the in-repo simulators need no declaration to work.
    """
    raw = getattr(spec, "raw", None) or {}
    block = raw.get("benchmarks") or {}
    return {k: dict(v) for k, v in block.items() if isinstance(v, dict)}


def audit(spec: Any) -> list[dict[str, Any]]:
    """One row per simulator a skill names: what is declared, and what is actually installed.

    The deploy check behind `icilval benchmarks list|verify`. Never raises: an operator asking
    what is wrong should be told everything at once.
    """
    _ensure_loaded()
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as dist_version

    pins = benchmark_pins(spec)
    wanted = {spec.simulator(skill) for skill in spec.skills} | set(pins)
    rows: list[dict[str, Any]] = []
    for name in sorted(wanted):
        pin = pins.get(name) or {}
        sim = REGISTRY.get(name)
        distribution = pin.get("distribution") or (sim.distribution if sim else None)
        row: dict[str, Any] = {
            "simulator": name,
            "declared": name in pins,
            "distribution": distribution,
            "installed": sim is not None,
            "version": None,
            "problems": [],
        }
        if sim is None:
            row["problems"].append("not installed")
        if distribution and distribution != "icilval":
            try:
                row["version"] = dist_version(distribution)
            except PackageNotFoundError:
                row["problems"].append(f"distribution {distribution} not found")
        pinned = pin.get("version")
        if pinned and row["version"] and pinned != row["version"]:
            row["problems"].append(f"version {row['version']} is not the pinned {pinned}")
        row["skills"] = [s for s in spec.skills if spec.simulator(s) == name]
        rows.append(row)
    return rows


def make_policy(model_dir: Any, arch_dir: Any, spec: Any, skill: str, device: str = "cuda") -> Any:
    """The policy for a skill, from its simulator."""
    return for_skill(spec, skill).make_policy(model_dir, arch_dir, spec, skill, device=device)


# The simulators this validator ships. Each import registers one.
from . import draw as _draw  # noqa: E402, F401
from . import libero as _libero  # noqa: E402, F401
