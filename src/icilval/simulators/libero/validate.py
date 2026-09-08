"""Simulator-side validation used while building a pool. Runs in the BPP environment."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ...spec import Spec
from . import bddl as B
from .env import LiberoEnv, load_init_states, save_init_states

log = logging.getLogger(__name__)


def valid_instances(env: LiberoEnv, states: np.ndarray, seed: int = 7) -> list[int]:
    """Initial states where the goal is not already satisfied."""
    out = []
    for i, s in enumerate(states):
        try:
            env.reset(seed, s)
        except Exception as exc:  # noqa: BLE001
            log.warning("instance %d: reset failed: %s", i, exc)
            continue
        if not env.success():
            out.append(i)
    return out


def build_task_env(bddl_path: Path, spec: Spec, skill: str) -> LiberoEnv | None:
    try:
        return LiberoEnv(bddl_path, spec, skill=skill)
    except Exception as exc:  # noqa: BLE001
        log.warning("cannot build %s: %s", bddl_path.name, exc)
        return None


def goal_from_bddl(path: Path) -> list[list[str]]:
    return B.goal_predicates(B.load(path))


__all__ = [
    "valid_instances",
    "build_task_env",
    "goal_from_bddl",
    "load_init_states",
    "save_init_states",
]
