"""The LIBERO prompt generator's pure parts: the LIBERO config, grasp-source lookup, BDDL check."""

import os
import sys
from pathlib import Path

import pytest

from icilval.pools.schema import Pool, PoolTask
from icilval.simulators.libero.generate import (
    BDDL_FILES_REL,
    CONFIG_ENV,
    Attempt,
    GenerationResult,
    grasp_sources,
    libero_config,
    libero_config_doc,
    own_grasp_sources,
    task_name,
    vendored_bddl,
)


def test_libero_config_doc_points_at_the_vendored_checkout():
    doc = libero_config_doc(Path("/bpp"), Path("/raw/LIBERO-datasets"))
    assert set(doc) == {"assets", "bddl_files", "benchmark_root", "datasets", "init_states"}
    assert doc["datasets"] == "/raw/LIBERO-datasets"
    assert doc["assets"] == "/bpp/deps/LIBERO/libero/libero/assets"
    assert doc["bddl_files"] == "/bpp/" + BDDL_FILES_REL
    assert doc["init_states"].endswith("env/libero/init_files")


def test_libero_config_writes_yaml_and_points_libero_at_it(tmp_path, monkeypatch):
    import yaml

    monkeypatch.delenv(CONFIG_ENV, raising=False)
    monkeypatch.delenv("PYNPUT_BACKEND", raising=False)
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delitem(sys.modules, "libero", raising=False)
    path = libero_config(tmp_path / "cfg", Path("/bpp"), Path("/raw"))
    assert path == tmp_path / "cfg" / "config.yaml"
    assert yaml.safe_load(path.read_text())["datasets"] == "/raw"
    assert os.environ[CONFIG_ENV] == str(tmp_path / "cfg")
    assert os.environ["PYNPUT_BACKEND"] == "dummy" and os.environ["MUJOCO_GL"] == "egl"
    # a process that already imported libero under another config must not be re-pointed
    monkeypatch.setitem(sys.modules, "libero", object())
    monkeypatch.setenv(CONFIG_ENV, "/elsewhere")
    with pytest.raises(RuntimeError, match="fresh process"):
        libero_config(tmp_path / "cfg", Path("/bpp"), Path("/raw"))


def make_pool(root: Path):
    bddl = root / "bddl" / "g" / "t.bddl"
    bddl.parent.mkdir(parents=True)
    bddl.write_text("(define (problem P))")
    task = PoolTask(
        task_id="g/t",
        skill="pick_and_place",
        kind="combination",
        suite="view_a",
        bddl="bddl/g/t.bddl",
        language="l",
        n_init=5,
        demos=[],
        max_steps=300,
        meta={"source_split": "view_a"},
    )
    pool = Pool(
        schema=4,
        pool_version="t",
        spec_version=4,
        sources={
            "g": {
                "dataset": "austinpatel/libero_gen_spatial_combination_hdf5",
                "grasp_sources": {
                    "dataset": "o/d",
                    "prefix": "libero_spatial/",
                    "files": {"a_demo.hdf5": "a" * 64},
                },
            }
        },
        tasks={"g/t": task},
        skills={"pick_and_place": {"eligible": ["g/t"], "diagnostic": []}},
        root=root,
    )
    return pool, task


def test_grasp_sources_lookup(spec, tmp_path):
    pool, _ = make_pool(tmp_path / "cat")
    assert grasp_sources(pool, "pick_and_place", spec)["files"] == {"a_demo.hdf5": "a" * 64}
    pool.sources = {}
    with pytest.raises(ValueError, match="grasp sources"):
        grasp_sources(pool, "pick_and_place", spec)


def test_vendored_bddl_must_match_the_catalogue(tmp_path):
    pool, task = make_pool(tmp_path / "cat")
    bpp = tmp_path / "bpp"
    vendored = bpp / BDDL_FILES_REL / "view_a" / "t.bddl"
    vendored.parent.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        vendored_bddl(task, pool, bpp)
    vendored.write_text("(define (problem Q))")
    with pytest.raises(ValueError, match="differs"):
        vendored_bddl(task, pool, bpp)
    vendored.write_text("(define (problem P))")
    assert vendored_bddl(task, pool, bpp) == vendored
    assert task_name(task) == "t"


def test_generation_result_as_dict():
    r = GenerationResult(False, [Attempt(1, False, "Grasp x"), Attempt(2, True)])
    d = r.as_dict()
    assert d["attempts"][0]["failed_stage"] == "Grasp x" and d["attempts"][1]["success"]
    assert r.n_attempts == 2 and d["sha256"] is None and d["steps"] == 0


def test_an_existing_task_lifts_its_grasps_from_its_own_file():
    meta = {
        "execution_steps": ["Grasp akita_black_bowl_1", "Place akita_black_bowl_1 plate_1"],
        "is_existing_task": True,
    }
    assert own_grasp_sources(meta, "pick_up_the_bowl") == {
        "akita_black_bowl_1": {"task": "pick_up_the_bowl", "object": "akita_black_bowl_1"}
    }


def test_a_task_with_its_own_source_is_left_alone():
    steps = ["Grasp akita_black_bowl_1", "Place akita_black_bowl_1 plate_1"]
    with_grasps = {"execution_steps": steps, "grasps_from": {"akita_black_bowl_1": {}}}
    with_actions = {"execution_steps": steps, "actions_from": "other_task"}
    assert own_grasp_sources(with_grasps, "t") is None
    assert own_grasp_sources(with_actions, "t") is None
    assert own_grasp_sources({"execution_steps": ["Place a b"]}, "t") is None
