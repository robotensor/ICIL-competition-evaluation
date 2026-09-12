"""Which policy an architecture gets, keyed by the architecture and nothing else.

A skill names two things: an architecture (`skills.<skill>.architecture`) and a simulator
(`skills.<skill>.simulator`). The policy belongs to the **architecture**. It is instantiated from
`arch/<architecture>.cfg.json`, its weights are checked against `arch/<architecture>.tensors.json`,
and the observation names it consumes are the ones that template declares - so the conversion
from a benchmark's arrays into those names is part of what the architecture id denotes, which is
why two templates can be byte-identical apart from their name and still be different
architectures.

Dispatching on the simulator instead only worked while every benchmark lived in this repository.
A benchmark in another repository has no policy class here to be asked for: the orchestrator
holds the weights and the template and *serves* the policy to it (`model.host`), so the
simulator's side of the wire never loads one. Asking the simulator for a policy therefore had to
answer "out of process" for exactly the benchmarks this layer exists to run.

Adding an architecture is a module that calls `register` and one entry in `PROVIDERS`.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: architecture name -> (model_dir, arch_dir, spec, skill, device) -> a `PolicyBase`
REGISTRY: dict[str, Callable[..., Any]] = {}

#: The modules that register one when imported, imported once on the first lookup. They are
#: imported lazily so that `icilval spec validate`, the queue and the dashboard's contract checks
#: - none of which has torch or a simulator - can import this module for free.
PROVIDERS = ("icilval.simulators",)

_LOADED = False


def register(name: str, factory: Callable[..., Any]) -> Callable[..., Any]:
    """Claim `name` for `factory`. A second claim is an error, never a silent replacement."""
    if name in REGISTRY:
        raise ValueError(f"architecture {name!r} registered twice")
    REGISTRY[name] = factory
    return factory


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    for module in PROVIDERS:
        importlib.import_module(module)


def names() -> tuple[str, ...]:
    """Every architecture this validator can build a policy for."""
    _ensure_loaded()
    return tuple(sorted(REGISTRY))


def implemented(name: str) -> bool:
    _ensure_loaded()
    return name in REGISTRY


def get(name: str) -> Callable[..., Any]:
    _ensure_loaded()
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no policy for architecture {name!r}; this validator builds "
            f"{', '.join(sorted(REGISTRY))}"
        ) from None


def make_policy(
    model_dir: str | Path, arch_dir: str | Path, spec: Any, skill: str, device: str = "cuda"
) -> Any:
    """The policy for a skill, from the architecture that skill names."""
    return get(spec.architecture(skill))(model_dir, arch_dir, spec, skill, device=device)
