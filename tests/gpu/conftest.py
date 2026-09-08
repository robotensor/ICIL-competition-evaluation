from __future__ import annotations

import os
from pathlib import Path

import pytest

MODEL_DIR = Path(
    os.environ.get(
        "ICILVAL_GENESIS_DIR", Path.home() / ".cache" / "icilval" / "models" / "bpp-genesis"
    )
)


@pytest.fixture(scope="session", autouse=True)
def _gpu_env():
    if os.environ.get("ICILVAL_TEST_GPU") != "1":
        pytest.skip("set ICILVAL_TEST_GPU=1 to run GPU tests")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


@pytest.fixture(scope="session")
def genesis_dir():
    """The two-skill genesis submission: <skill>/model.safetensors per skill."""
    if not (MODEL_DIR / "pick_and_place" / "model.safetensors").exists():
        pytest.skip(f"no converted genesis submission at {MODEL_DIR}")
    return MODEL_DIR


@pytest.fixture(scope="session")
def smoke_pool_or_skip():
    root = Path(
        os.environ.get("ICILVAL_SMOKE_POOL", Path.home() / ".cache" / "icilval" / "pools" / "smoke")
    )
    if not (root / "catalogue.json").exists():
        pytest.skip(f"no catalogue at {root}")
    from icilval.pools.schema import Pool

    return Pool.load(root)
