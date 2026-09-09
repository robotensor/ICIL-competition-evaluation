"""The drawing catalogue stage: one task per primitive family and nothing else."""

from icilval.pools.build import finalize, open_pool
from icilval.simulators.draw.pool import GENERATED_GROUP, family_task_id, stage_families


def test_stage_families_one_generated_task_per_family(spec, tmp_path):
    pool = open_pool(tmp_path, spec, "t")
    added = stage_families(pool, spec, "draw_anything")
    families = [
        k for k in spec.skill_generation("draw_anything")["families"] if not k.startswith("_")
    ]
    assert added == [family_task_id(f) for f in families]
    task = pool.tasks[family_task_id("glyph")]
    assert task.skill == "draw_anything" and not hasattr(task, "demos")
    assert task.suite == GENERATED_GROUP and task.bddl is None and task.init is None
    assert task.n_init == spec.catalogue["init_states_per_task"]
    assert task.meta["family"] == "glyph" and task.meta["characters"]
    assert not any(k.startswith("_") for k in task.meta)
    assert stage_families(pool, spec, "draw_anything") == []  # idempotent
    counts = finalize(pool, spec)
    assert counts["draw_anything"] == {"eligible": len(families)}
    assert (tmp_path / "catalogue.json").exists()
