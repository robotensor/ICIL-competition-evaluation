from __future__ import annotations

import os
from pathlib import Path

import pytest

SMOKE_CATALOGUE = Path(
    os.environ.get("ICILVAL_SMOKE_POOL", Path.home() / ".cache" / "icilval" / "pools" / "smoke")
)


PP = "pick_and_place"
DA = "draw_anything"


def _has_sim() -> bool:
    try:
        import libero  # noqa: F401
        import pygame  # noqa: F401
        import pymunk  # noqa: F401
        import robosuite  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="session", autouse=True)
def _sim_env():
    if not _has_sim():
        pytest.skip("simulators not importable (run in the BPP conda env)")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


@pytest.fixture(scope="session")
def smoke_pool():
    """A small catalogue: `icilval catalogue build --limit 1 --fetch --out <dir>`."""
    if not (SMOKE_CATALOGUE / "catalogue.json").exists():
        pytest.skip(f"no smoke catalogue at {SMOKE_CATALOGUE} (icilval catalogue build --limit 1)")
    from icilval.pools.schema import Pool

    return Pool.load(SMOKE_CATALOGUE)
