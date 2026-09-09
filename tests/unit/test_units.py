import pytest

from icilval.ids import ModelRef, duel_id
from icilval.pools.schema import Pool, PoolTask
from icilval.pools.units import derive_units
from icilval.simulators.draw.units import draw_instance

PP, DA = "pick_and_place", "draw_anything"
FAMILIES = ("bpp", "polygon", "glyph")


def make_pool(n_tasks=3, n_init=5):
    """A catalogue: LIBERO task definitions and one task per drawing family. No demonstrations."""
    tasks = {}
    skills = {PP: {"eligible": []}, DA: {"eligible": []}}
    for t in range(n_tasks):
        tid = f"libero_gen_spatial_combination/task{t}"
        tasks[tid] = PoolTask(
            task_id=tid,
            skill=PP,
            kind="combination",
            suite="libero_spatial_selected_combinations_inverse_view",
            bddl=f"bddl/{t}.bddl",
            language=f"pick task {t}",
            n_init=n_init,
            goal=[["On", "a", "b"]],
            max_steps=300,
            steps=[["Grasp", "a"], ["Place", "a", "b"]],
            meta={"grasps_from": {"a": {"task": "human", "object": "a"}}},
        )
        skills[PP]["eligible"].append(tid)
    for family in FAMILIES:
        fid = f"drawanything_generated/{family}"
        tasks[fid] = PoolTask(
            task_id=fid,
            skill=DA,
            kind="drawing",
            suite="drawanything_generated",
            language=f"draw a generated {family} drawing",
            n_init=n_init,
            max_steps=400,
            meta={"family": family},
        )
        skills[DA]["eligible"].append(fid)
    pool = Pool(
        schema=4,
        pool_version="test",
        spec_version=4,
        sources={},
        tasks=tasks,
        skills=skills,
    )
    pool.seal()
    return pool


def test_catalogue_roundtrip(tmp_path):
    pool = make_pool()
    pool.save(tmp_path)
    assert (tmp_path / "catalogue.json").exists() and not (tmp_path / "pool.json").exists()
    again = Pool.load(tmp_path)
    assert again.pool_id == pool.pool_id
    assert again.to_dict() == pool.to_dict()
    assert again.tasks["drawanything_generated/glyph"].bddl is None
    assert again.eligible(PP) == sorted(t for t in pool.tasks if pool.tasks[t].skill == PP)
    assert again.eligible(DA) == sorted(f"drawanything_generated/{f}" for f in FAMILIES)
    (tmp_path / "catalogue.json").write_text(
        (tmp_path / "catalogue.json").read_text().replace('"max_steps": 300', '"max_steps": 301')
    )
    with pytest.raises(ValueError):
        Pool.load(tmp_path)


def _scored(units, skill):
    return [u for u in units if u.skill == skill]


def test_units_deterministic_and_spread_over_tasks(spec):
    pool = make_pool()
    did = duel_id(4, spec.track_id, ModelRef.make("a/b", "1" * 40), ModelRef.make("c/d", "2" * 40))
    units = derive_units(pool, spec, did, "heavy")
    again = derive_units(pool, spec, did, "heavy")
    assert [u.as_dict() for u in units] == [u.as_dict() for u in again]
    per = spec.units_per_skill("heavy")
    assert len(units) == len(spec.skills) * per
    for skill in spec.skills:
        scored = _scored(units, skill)
        assert len(scored) == per
        assert [u.index for u in scored] == list(range(per))
        counts = {}
        for u in scored:
            counts[u.task] = counts.get(u.task, 0) + 1
        assert set(counts) == set(pool.eligible(skill))
        assert max(counts.values()) - min(counts.values()) <= 1
        assert all(u.unit_id.startswith(spec.skill_code(skill) + "-") for u in scored)
    ids = [u.unit_id for u in units]
    assert len(set(ids)) == len(ids)
    other = derive_units(
        pool, spec, duel_id(4, spec.track_id, ModelRef.make("a/b", "3" * 40), None), "heavy"
    )
    assert [u.seed for u in other] != [u.seed for u in units]


def test_a_duel_derives_only_scored_units(spec):
    """Nothing is published but the units that count: no diagnostic ever enters a unit list."""
    pool = make_pool()
    did = duel_id(4, spec.track_id, ModelRef.make("a/b", "1" * 40), None)
    for size in ("smoke", "heavy"):
        units = derive_units(pool, spec, did, size)
        per = spec.units_per_skill(size)
        assert len(units) == len(spec.skills) * per
        assert all(not hasattr(u, "diagnostic") for u in units)
        for skill in spec.skills:
            assert [u.index for u in _scored(units, skill)] == list(range(per))
        drawings = _scored(units, DA)
        assert {u.task for u in drawings} <= set(pool.eligible(DA))
        assert all(u.demo == f"generated/{u.unit_id}" for u in drawings)


def test_units_prompts_are_generated_and_instances_in_range(spec):
    pool = make_pool()
    did = duel_id(4, spec.track_id, ModelRef.make("a/b", "1" * 40), None)
    env = spec.env("draw_anything")
    for u in derive_units(pool, spec, did, "heavy"):
        task = pool.tasks[u.task]
        assert u.max_steps <= spec.max_steps(u.skill)
        assert u.substituted_from is None and u.prompt_sha256 is None
        if u.skill == PP:
            assert u.change["kind"] in spec.changes(PP)
            assert u.demo == f"generated/{u.unit_id}"
            assert 0 <= u.instance < 5
            assert u.bddl and u.init is None
            assert u.instance_params == {} and u.steps == task.steps
        if u.skill == DA:
            assert u.change["kind"] in spec.changes(DA)
            p = u.instance_params
            lo, hi = env["board_angle_range_rad"]
            assert lo <= p["angle_rad"] <= hi
            c_lo, c_hi = env["cursor_start_range_px"]
            assert all(c_lo <= c <= c_hi for c in p["cursor_px"])
            assert u.bddl is None and u.init is None
            state = draw_instance(u.task, u.instance, env)
            assert state["angle_rad"] == p["angle_rad"] and state["cursor_px"] == p["cursor_px"]
            assert u.demo == f"generated/{u.unit_id}"
            assert p["family"] == task.meta["family"]


def test_units_need_eligible_tasks(spec):
    pool = make_pool()
    pool.skills[DA]["eligible"] = []
    did = duel_id(4, spec.track_id, ModelRef.make("a/b", "1" * 40), None)
    with pytest.raises(ValueError, match="no eligible tasks"):
        derive_units(pool, spec, did, "smoke")


def test_instances_roundtrip(tmp_path):
    pool = make_pool()
    tid = next(iter(pool.tasks))
    pool.tasks[tid].instances = [1, 3]
    pool.seal()
    pool.save(tmp_path)
    again = Pool.load(tmp_path)
    assert again.tasks[tid].instances == [1, 3] and again.tasks[tid].valid_instances == [1, 3]
    assert again.pool_id == pool.pool_id


def test_old_schemas_are_refused(tmp_path):
    (tmp_path / "catalogue.json").write_text(
        '{"schema": 3, "pool_version": "x", "spec_version": 3}'
    )
    with pytest.raises(ValueError, match="rebuild"):
        Pool.load(tmp_path)
    (tmp_path / "catalogue.json").unlink()
    (tmp_path / "pool.json").write_text('{"schema": 3, "pool_version": "x", "spec_version": 3}')
    with pytest.raises(ValueError, match="pre-v4 pool"):
        Pool.load(tmp_path)
