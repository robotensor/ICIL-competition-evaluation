"""Simulator-side checks used while building a catalogue. Runs in the BPP environment."""

from __future__ import annotations

import logging
from pathlib import Path

from ...spec import Spec
from .env import LiberoEnv

log = logging.getLogger(__name__)

RESET_SEEDS = (0, 1, 2)


def goal_unsatisfied_at_reset(env: LiberoEnv, seeds: tuple[int, ...] = RESET_SEEDS) -> bool:
    """A task can be scored from its own resets: the goal does not hold right after `reset`."""
    for seed in seeds:
        try:
            env.reset(seed, None)
        except Exception as exc:  # noqa: BLE001
            log.warning("reset %d failed: %s", seed, exc)
            return False
        if env.success():
            return False
    return True


def build_task_env(bddl_path: Path, spec: Spec, skill: str) -> LiberoEnv | None:
    try:
        return LiberoEnv(bddl_path, spec, skill=skill)
    except Exception as exc:  # noqa: BLE001
        log.warning("cannot build %s: %s", bddl_path.name, exc)
        return None


def generates_from_reset(bddl_path: Path, spec: Spec, skill: str) -> bool:
    env = build_task_env(bddl_path, spec, skill)
    if env is None:
        return False
    try:
        return goal_unsatisfied_at_reset(env)
    finally:
        env.close()


__all__ = ["goal_unsatisfied_at_reset", "build_task_env", "generates_from_reset"]
