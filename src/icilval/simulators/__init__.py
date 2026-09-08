"""The simulator registry.

A skill names its simulator in `spec.json` (`skills.<skill>.simulator`). Everything that is
specific to a simulator - the policy class, the episode loop, the pool stage, how a unit's
instance is set up, how a demonstration is rendered, what its spec entry must carry - lives
behind `Simulator`, one package per simulator under this one. Generic code (the side runner,
unit derivation, the pool builder, the spec validator) looks the simulator up here and never
names one. Adding a simulator is a new package that calls `register` and an import below.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


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


REGISTRY: dict[str, Simulator] = {}


def register(sim: Simulator) -> Simulator:
    if sim.name in REGISTRY:
        raise ValueError(f"simulator {sim.name!r} registered twice")
    REGISTRY[sim.name] = sim
    return sim


def names() -> tuple[str, ...]:
    return tuple(REGISTRY)


def get(name: str) -> Simulator:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown simulator {name!r}; registered: {', '.join(REGISTRY)}") from None


def for_skill(spec: Any, skill: str) -> Simulator:
    return get(spec.simulator(skill))


def make_policy(model_dir: Any, arch_dir: Any, spec: Any, skill: str, device: str = "cuda") -> Any:
    """The policy for a skill, from its simulator."""
    return for_skill(spec, skill).make_policy(model_dir, arch_dir, spec, skill, device=device)


# The simulators this validator ships. Each import registers one.
from . import draw as _draw  # noqa: E402, F401
from . import libero as _libero  # noqa: E402, F401
