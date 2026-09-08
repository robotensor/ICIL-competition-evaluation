import pytest

from icilval.ids import ModelRef, duel_id
from icilval.pools.schema import Pool, PoolTask
from icilval.pools.units import derive_units
from icilval.simulators.draw.units import draw_instance

PP, DA = "pick_and_place", "draw_anything"


def make_pool(n_tasks=3, n_init=5, n_demos=4):
    tasks = {}
    skills = {PP: {"eligible": []}, DA: {"eligible": []}}
    for t in range(n_tasks):
        tid = f"libero_spatial/task{t}"
        demos = [f"{tid}/demo_{d:02d}" for d in range(n_demos)]
        tasks[tid] = PoolTask(
            task_id=tid,
            skill=PP,
            kind="base",
            suite="libero_spatial",
            bddl=f"bddl/{t}.bddl",
            language=f"pick task {t}",
            init=f"init/{t}.npz",
            n_init=n_init,
            goal=[["On", "a", "b"]],
            demos=demos,
            max_steps=300,
            demo_init_index={demos[0]: 0},
        )
        skills[PP]["eligible"].append(tid)
        oid = f"libero_gen/swap{t}"
        odemos = [f"{oid}/demo_{d:02d}" for d in range(n_demos)]
        tasks[oid] = PoolTask(
            task_id=oid,
            skill=PP,
            kind="object_swap",
            suite="gen",
            bddl=f"bddl/swap{t}.bddl",
            language=f"put thing {t}",
            init=f"init/swap{t}.npz",
            n_init=n_init,
            goal=[["On", "c", "d"]],
            demos=odemos,
            max_steps=400,
            meta={"swap": {"operator": "place_on", "object": "c", "target": "d"}},
        )
        skills[PP]["eligible"].append(oid)
        did_ = f"drawanything_handmade/draw_{t}"
        ddemos = [f"{did_}/demo_{d:02d}" for d in range(n_demos)]
        tasks[did_] = PoolTask(
            task_id=did_,
            skill=DA,
            kind="drawing",
            suite="drawanything_handmade",
            language=f"draw {t}",
            n_init=n_init,
            demos=ddemos,
            max_steps=400,
            meta={"demo_angles": {d: (-0.7 + 0.4 * i) for i, d in enumerate(ddemos)}},
        )
        skills[DA]["eligible"].append(did_)
    pool = Pool(
        schema=3,
        pool_version="test",
        spec_version=3,
        sources={},
        tasks=tasks,
        skills=skills,
    )
    pool.seal()
    return pool


def test_pool_roundtrip(tmp_path):
    pool = make_pool()
    pool.save(tmp_path)
    again = Pool.load(tmp_path)
    assert again.pool_id == pool.pool_id
    assert again.to_dict() == pool.to_dict()
    assert again.tasks["drawanything_handmade/draw_0"].bddl is None
    assert again.eligible(PP) == sorted(t for t in pool.tasks if pool.tasks[t].skill == PP)
    (tmp_path / "pool.json").write_text(
        (tmp_path / "pool.json").read_text().replace('"max_steps": 300', '"max_steps": 301')
    )
    with pytest.raises(ValueError):
        Pool.load(tmp_path)


def test_units_deterministic_and_spread_over_tasks(spec):
    pool = make_pool()
    did = duel_id(3, spec.track_id, ModelRef.make("a/b", "1" * 40), ModelRef.make("c/d", "2" * 40))
    units = derive_units(pool, spec, did, "heavy")
    again = derive_units(pool, spec, did, "heavy")
    assert [u.as_dict() for u in units] == [u.as_dict() for u in again]
    per = spec.units_per_skill("heavy")
    assert len(units) == len(spec.skills) * per
    for skill in spec.skills:
        skill_units = [u for u in units if u.skill == skill]
        assert len(skill_units) == per
        assert [u.index for u in skill_units] == list(range(per))
        counts = {}
        for u in skill_units:
            counts[u.task] = counts.get(u.task, 0) + 1
        assert set(counts) == set(pool.eligible(skill))
        assert max(counts.values()) - min(counts.values()) <= 1
        assert all(u.unit_id.startswith(spec.skill_code(skill) + "-") for u in skill_units)
    ids = [u.unit_id for u in units]
    assert len(set(ids)) == len(ids)
    other = derive_units(
        pool, spec, duel_id(3, spec.track_id, ModelRef.make("a/b", "3" * 40), None), "heavy"
    )
    assert [u.seed for u in other] != [u.seed for u in units]


def test_units_prompt_disjoint_and_instances(spec):
    pool = make_pool()
    did = duel_id(3, spec.track_id, ModelRef.make("a/b", "1" * 40), None)
    env = spec.env("draw_anything")
    for u in derive_units(pool, spec, did, "heavy"):
        task = pool.tasks[u.task]
        assert u.max_steps <= spec.max_steps(u.skill)
        assert u.demo in task.demos
        if u.skill == PP:
            assert task.demo_init_index.get(u.demo) != u.instance
            assert 0 <= u.instance < 5
            assert u.bddl and u.init
            assert u.instance_params == {}
        if u.skill == DA:
            p = u.instance_params
            lo, hi = env["board_angle_range_rad"]
            assert lo <= p["angle_rad"] <= hi
            assert task.meta["demo_angles"][u.demo] == pytest.approx(p["demo_angle_rad"], abs=1e-5)
            c_lo, c_hi = env["cursor_start_range_px"]
            assert all(c_lo <= c <= c_hi for c in p["cursor_px"])
            assert u.bddl is None and u.init is None
            state = draw_instance(u.task, u.instance, env)
            assert state["angle_rad"] == p["angle_rad"] and state["cursor_px"] == p["cursor_px"]


def test_units_need_eligible_tasks(spec):
    pool = make_pool()
    pool.skills[DA]["eligible"] = []
    did = duel_id(3, spec.track_id, ModelRef.make("a/b", "1" * 40), None)
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


def test_old_pool_schema_is_refused(tmp_path):
    (tmp_path / "pool.json").write_text('{"schema": 2, "pool_version": "x", "spec_version": 2}')
    with pytest.raises(ValueError, match="upgrade"):
        Pool.load(tmp_path)
