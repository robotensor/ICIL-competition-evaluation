"""The drawing catalogue stage's pure parts: family tasks and task-name sanitizing."""

from icilval.pools.build import finalize, open_pool
from icilval.simulators.draw.pool import (
    GENERATED_GROUP,
    family_task_id,
    sanitize_task_name,
    stage_families,
)


def test_stage_families_one_generated_task_per_family(spec, tmp_path):
    pool = open_pool(tmp_path, spec, "t")
    added = stage_families(pool, spec, "draw_anything")
    families = [
        k for k in spec.skill_generation("draw_anything")["families"] if not k.startswith("_")
    ]
    assert added == [family_task_id(f) for f in families]
    task = pool.tasks[family_task_id("glyph")]
    assert task.skill == "draw_anything" and task.demos == [] and not task.diagnostic
    assert task.suite == GENERATED_GROUP and task.bddl is None and task.init is None
    assert task.n_init == spec.catalogue["init_states_per_task"]
    assert task.meta["family"] == "glyph" and task.meta["characters"]
    assert not any(k.startswith("_") for k in task.meta)
    assert stage_families(pool, spec, "draw_anything") == []  # idempotent
    counts = finalize(pool, spec)
    assert counts["draw_anything"] == {"eligible": len(families), "diagnostic": 0}
    assert (tmp_path / "catalogue.json").exists()


def test_sanitize_task_name():
    assert sanitize_task_name("draw A") == "draw_A_upper"
    assert sanitize_task_name("draw a") == "draw_a_lower"
    assert sanitize_task_name("draw 5*") == "draw_5star"
    assert sanitize_task_name("draw procedural_1") == "draw_procedural_1_lower"
    assert sanitize_task_name("draw Ab c") == "draw_Ab_c"
