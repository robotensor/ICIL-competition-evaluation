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


def bpp_root() -> Path:
    from icilval.spec import _repo_root

    return (_repo_root() or Path.cwd()) / "vendor" / "behavior_prompting"


def grasp_sources_root() -> Path:
    """Where the generator reads the human teleoperation files: the raw cache's copy of the
    grasp-source dataset (`raw/LIBERO-datasets/libero_spatial/*.hdf5`)."""
    from icilval.pools.sources import Sources
    from icilval.spec import _repo_root, load_spec

    spec = load_spec()
    dataset = str(spec.tasks("pick_and_place")["grasp_sources"]["dataset"])
    return Sources.default(_repo_root() or Path.cwd()).dataset_root(dataset)


@pytest.fixture(scope="session", autouse=True)
def _sim_env(tmp_path_factory):
    # the validator's LIBERO config must be in place before libero is first imported
    from icilval.simulators.libero.generate import libero_config

    libero_config(tmp_path_factory.mktemp("libero-config"), bpp_root(), grasp_sources_root())
    if not _has_sim():
        pytest.skip("simulators not importable (run in the BPP conda env)")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


@pytest.fixture(scope="session")
def smoke_pool():
    """A small catalogue: `icilval catalogue build --limit 1 --fetch --out <dir>`."""
    if not (SMOKE_CATALOGUE / "catalogue.json").exists():
        pytest.skip(f"no smoke catalogue at {SMOKE_CATALOGUE} (icilval catalogue build --limit 1)")
    from icilval.pools.schema import Pool

    return Pool.load(SMOKE_CATALOGUE)
