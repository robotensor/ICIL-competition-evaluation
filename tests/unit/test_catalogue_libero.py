"""The LIBERO catalogue stage's pure parts: metadata, hashes, numbered resets."""

import pytest

from icilval.canon import sha256_file
from icilval.pools.sources import Sources, fetch_by_hash, grasp_source_hashes, hub_file_hashes
from icilval.simulators.libero.episode import instance_seed
from icilval.simulators.libero.pool import (
    METADATA_KEYS,
    METADATA_REL,
    execution_steps,
    load_task_metadata,
    task_meta,
)
from icilval.spec import _repo_root


def test_execution_steps_and_task_meta():
    meta = {
        "execution_steps": ["Grasp akita_black_bowl_2", "Place akita_black_bowl_2 cookies_1"],
        "grasps_from": {"akita_black_bowl_2": {"task": "human", "object": "akita_black_bowl_1"}},
        "is_existing_task": False,
        "demo_idx": 3,
    }
    assert execution_steps(meta) == [
        ["Grasp", "akita_black_bowl_2"],
        ["Place", "akita_black_bowl_2", "cookies_1"],
    ]
    assert execution_steps({}) == []
    m = task_meta("t", {"t": meta}, "libero_spatial", "some_view")
    assert m["base_split"] == "libero_spatial" and m["source_split"] == "some_view"
    assert "demo_idx" not in m
    assert set(m) - {"base_split", "source_split"} <= set(METADATA_KEYS)
    assert m["grasps_from"]["akita_black_bowl_2"]["task"] == "human"


def test_vendored_task_metadata_covers_the_release(spec):
    root = _repo_root()
    view = spec.tasks("pick_and_place")["views"][-1]
    bpp = root / "vendor" / "behavior_prompting"
    if not (bpp / METADATA_REL.format(view=view)).exists():
        pytest.skip("BPP submodule not checked out")
    base, tasks = load_task_metadata(bpp, view)
    assert base == "libero_spatial" and len(tasks) >= 164
    assert all("execution_steps" in m for m in tasks.values())
    assert all(
        ("grasps_from" in m) or ("actions_from" in m) or m.get("is_existing_task")
        for m in tasks.values()
    )


def test_hub_file_hashes_reads_lfs_entries(monkeypatch):
    class Lfs:
        sha256 = "a" * 64

    class Entry:
        def __init__(self, path, lfs):
            self.path, self.lfs = path, lfs

    class Api:
        def list_repo_tree(self, dataset, path_in_repo, repo_type, expand):
            assert repo_type == "dataset" and expand and path_in_repo == "libero_spatial"
            return [
                Entry(f"{path_in_repo}/b_demo.hdf5", Lfs()),
                Entry(f"{path_in_repo}/a_demo.hdf5", Lfs()),
                Entry(f"{path_in_repo}/README.md", None),
            ]

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: Api())
    out = hub_file_hashes("o/d", "libero_spatial/")
    assert list(out) == ["a_demo.hdf5", "b_demo.hdf5"] and set(out.values()) == {"a" * 64}


def test_grasp_source_hashes_offline_and_fetch_by_hash(tmp_path):
    src = Sources(raw=tmp_path / "raw", bpp_root=tmp_path)
    dataset = "o/LIBERO-datasets"
    local = src.dataset_root(dataset) / "libero_spatial"
    local.mkdir(parents=True)
    (local / "x_demo.hdf5").write_bytes(b"hello")
    hashes = grasp_source_hashes(src, dataset, "libero_spatial/", online=False)
    assert hashes == {"x_demo.hdf5": sha256_file(local / "x_demo.hdf5")}
    with pytest.raises(FileNotFoundError):
        grasp_source_hashes(src, "o/other", "libero_spatial/", online=False)
    root = src.dataset_root(dataset)
    got = fetch_by_hash(root, dataset, "libero_spatial/x_demo.hdf5", hashes["x_demo.hdf5"])
    assert got == local / "x_demo.hdf5"
    with pytest.raises(ValueError, match="sha256"):
        fetch_by_hash(root, dataset, "libero_spatial/x_demo.hdf5", "0" * 64)


def test_instance_seed_is_a_function_of_task_and_instance():
    a = instance_seed({"task": "t", "instance": 1, "seed": 5})
    assert a == instance_seed({"task": "t", "instance": 1, "seed": 9})
    assert 0 <= a < (1 << 31)
    assert a != instance_seed({"task": "t", "instance": 2})
    assert a != instance_seed({"task": "u", "instance": 1})
