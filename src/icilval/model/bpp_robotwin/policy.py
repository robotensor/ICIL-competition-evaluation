"""The `bpp_robotwin_v1` policy: the BPP network, prompted with one bimanual demonstration.

`conversion.py` holds every array transformation and imports nothing but numpy. This module is
the thin torch-side shell around it: build the prompt the network takes, run `predict_action`,
and turn each 10-dimensional delta into one `ee` action row for the robot.

**Cadence.** The network predicts a chunk of `action_horizon` actions from a history of
`obs_history` observations, and the first `exec_horizon` of them are executed; all three come
from `spec.json`. One action leaves the queue per `act()`, so the benchmark's step limit counts
model steps, and a new chunk is asked for only when the queue is empty. A chunk cannot be
returned all at once: each delta is applied to the *measured* pose of the step it is executed at,
so the policy has to see an observation between one action and the next.

**One arm.** The network as released drives a single arm; the robot is bimanual. The arm whose
tool centre travels further over the demonstration is the one driven, and the other is held at
the fixed pose it had in the episode's first observation (`conversion.IdleArmHold`).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from ..policy import PolicyBase
from ..prompt import PromptInfo, chunk_actions
from . import conversion
from .settings import ACTION_DIM, Settings, from_env

log = logging.getLogger(__name__)


class BPPRoboTwinPolicy(PolicyBase):
    """BPP prompted with one demonstration of a bimanual robot, driving one of its arms."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.settings: Settings = from_env(self.spec.env(self.skill))
        self._queue: deque[np.ndarray] = deque()
        self._prompt: conversion.Prompt | None = None
        self._choice: conversion.ArmChoice | None = None
        self._execution: conversion.Execution | None = None
        self.plans = 0

    # ---------------------------------------------------------------- lifecycle

    def reset(self) -> None:
        """Forget the episode: its prompt, its queued actions and its execution state."""
        super().reset()
        self._queue.clear()
        self._prompt = None
        self._choice = None
        self._execution = None
        self.plans = 0

    def episode_info(self) -> dict[str, Any]:
        """What this policy can say about the episode it just ran, for a run record."""
        if self._prompt is None or self._choice is None or self._execution is None:
            return {}
        return {
            **self._choice.info(),
            "prompt_rate_hz": self._prompt.rate_hz,
            "prompt_steps": int(len(self._prompt.actions)),
            "clipped_action_fraction": round(self._prompt.clipped, 6),
            "plans": self.plans,
            **self._execution.info(),
        }

    # ------------------------------------------------------------------- prompt

    def set_prompt(self, demo: dict[str, Any]) -> PromptInfo:
        """Convert one demonstration and hand it to the network, exactly once per episode."""
        self._choice = conversion.choose_arm(demo, self.settings)
        self._prompt = conversion.build_prompt(demo, self.settings, self._choice.arm)
        chunks, idx, info = chunk_actions(
            self._prompt.actions, self.chunk_n, self.pad_end, self.max_chunks
        )
        obs = conversion.prompt_observations(
            demo, self._prompt, idx, self.settings, self.image_size
        )
        self._execution = conversion.Execution(self.settings, self._choice.arm)
        self._queue.clear()
        self.plans = 0
        prompt_dict = {
            "obs": {key: self._lowdim(value) for key, value in obs.items()},
            "action": self._lowdim(chunks),
            "metadata": {"mask": self._mask(np.zeros((info.chunks,), dtype=bool))},
        }
        self.policy.reset(action_exec_horizon=self.exec_horizon)
        self.policy.prompt(prompt_dict)
        log.info(
            "prompt: %s arm (%s), %d steps at %.1f Hz, %d chunks, %.3f of entries clipped",
            self._choice.arm,
            self._choice.rule,
            info.steps,
            self._prompt.rate_hz,
            info.chunks,
            self._prompt.clipped,
        )
        return info

    # ---------------------------------------------------------------------- act

    def act(self, history: list[dict[str, Any]]) -> np.ndarray:
        """history: the last observations, oldest first; returns one (1, 16) `ee` action row."""
        if self._execution is None or self._choice is None:
            raise conversion.ConversionError("act() before set_prompt()")
        if not self._queue:
            self._queue.extend(self._plan(history))
        return self._execution.act(self._queue.popleft(), history[-1])

    def _plan(self, history: list[dict[str, Any]]) -> np.ndarray:
        """One `predict_action` from the last `obs_history` observations, as rows to execute."""
        import torch

        assert self._choice is not None
        states = [
            conversion.observation_state(obs, self.settings, self._choice.arm, self.image_size)
            for obs in self._history(history)
        ]
        obs_dict = {
            key: self._lowdim(np.stack([state[key] for state in states]))
            for key in conversion.OBSERVATION_KEYS
        }
        with torch.inference_mode():
            out = self.policy.predict_action(obs_dict)
        chunk = out["action"][0].detach().float().cpu().numpy()[: self.exec_horizon]
        if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM or len(chunk) < 1:
            raise conversion.ConversionError(
                f"the network returned a chunk of shape {chunk.shape}, expected "
                f"(k, {ACTION_DIM}) with k >= 1"
            )
        self.plans += 1
        return np.asarray(chunk, dtype=np.float64)
