import pytest

from icilval.ids import ModelRef, duel_id
from icilval.pools.schema import Pool, PoolTask
from icilval.pools.units import derive_units, diagnostic_units
from icilval.simulators.draw.units import draw_instance

PP, DA = "pick_and_place", "draw_anything"
FAMILIES = ("bpp", "polygon", "glyph")


def make_pool(n_tasks=3, n_init=5, n_demos=4):
    """A catalogue: LIBERO task definitions and drawing families (no demonstrations), plus the
    handmade drawings as a diagnostic (stored demonstrations)."""
    tasks = {}
    skills = {PP: {"eligible": [], "diagnostic": []}, DA: {"eligible": [], "diagnostic": []}}
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
            demos=[],
            max_steps=300,
            steps=[["Grasp", "a"], ["Place", "a", "b"]],
            meta={"grasps_from": {"a": {"task": "human", "object": "a"}}},
        )
        skills[PP]["eligible"].append(tid)
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
            diagnostic=True,
        )
        skills[DA]["diagnostic"].append(did_)
    for family in FAMILIES:
        fid = f"drawanything_generated/{family}"
        tasks[fid] = PoolTask(
            task_id=fid,
            skill=DA,
            kind="drawing",
            suite="drawanything_generated",
            language=f"draw a generated {family} drawing",
            n_init=n_init,
            demos=[],
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
    assert again.tasks["drawanything_handmade/draw_0"].bddl is None
    assert again.tasks["drawanything_handmade/draw_0"].diagnostic is True
    assert again.tasks["drawanything_generated/glyph"].diagnostic is False
    assert again.eligible(PP) == sorted(t for t in pool.tasks if pool.tasks[t].skill == PP)
    assert again.eligible(DA) == sorted(f"drawanything_generated/{f}" for f in FAMILIES)
    assert again.diagnostic(DA) == sorted(f"drawanything_handmade/draw_{t}" for t in range(3))
    (tmp_path / "catalogue.json").write_text(
        (tmp_path / "catalogue.json").read_text().replace('"max_steps": 300', '"max_steps": 301')
    )
    with pytest.raises(ValueError):
        Pool.load(tmp_path)


def _scored(units, skill):
    return [u for u in units if u.skill == skill and not u.diagnostic]


def test_units_deterministic_and_spread_over_tasks(spec):
    pool = make_pool()
    did = duel_id(4, spec.track_id, ModelRef.make("a/b", "1" * 40), ModelRef.make("c/d", "2" * 40))
    units = derive_units(pool, spec, did, "heavy")
    again = derive_units(pool, spec, did, "heavy")
    assert [u.as_dict() for u in units] == [u.as_dict() for u in again]
    per = spec.units_per_skill("heavy")
    n_diag = sum(
        diagnostic_units(spec, "heavy", d) for d in spec.diagnostics.values() if d["skill"] == DA
    )
    assert len(units) == len(spec.skills) * per + n_diag
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


def test_diagnostic_units_follow_the_scored_ones(spec):
    pool = make_pool()
    did = duel_id(4, spec.track_id, ModelRef.make("a/b", "1" * 40), None)
    for size in ("smoke", "heavy"):
        units = derive_units(pool, spec, did, size)
        diag = [u for u in units if u.diagnostic]
        assert diag and all(u.skill == DA for u in diag)
        expected = sum(
            diagnostic_units(spec, size, d) for d in spec.diagnostics.values() if d["skill"] == DA
        )
        assert len(diag) == expected
        first = spec.units_per_skill(size)
        assert [u.index for u in diag] == list(range(first, first + expected))
        assert {u.task for u in diag} <= set(pool.diagnostic(DA))
        for u in diag:
            assert u.demo in pool.tasks[u.task].demos
            assert "demo_angle_rad" in u.instance_params and "family" not in u.instance_params
    heavy = diagnostic_units(spec, "heavy", spec.diagnostics["handmade_drawings"])
    assert heavy == spec.diagnostics["handmade_drawings"]["units_per_duel"]
    assert 1 <= diagnostic_units(spec, "smoke", spec.diagnostics["handmade_drawings"]) <= heavy
    assert diagnostic_units(spec, "heavy", {"units_per_duel": 0}) == 0


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
            assert u.change == {"kind": "none"}
            p = u.instance_params
            lo, hi = env["board_angle_range_rad"]
            assert lo <= p["angle_rad"] <= hi
            c_lo, c_hi = env["cursor_start_range_px"]
            assert all(c_lo <= c <= c_hi for c in p["cursor_px"])
            assert u.bddl is None and u.init is None
            state = draw_instance(u.task, u.instance, env)
            assert state["angle_rad"] == p["angle_rad"] and state["cursor_px"] == p["cursor_px"]
            if u.diagnostic:
                assert task.meta["demo_angles"][u.demo] == pytest.approx(
                    p["demo_angle_rad"], abs=1e-5
                )
            else:
                assert u.demo == f"generated/{u.unit_id}"
                assert p["family"] == task.meta["family"]


def test_units_need_eligible_and_diagnostic_tasks(spec):
    pool = make_pool()
    pool.skills[DA]["eligible"] = []
    did = duel_id(4, spec.track_id, ModelRef.make("a/b", "1" * 40), None)
    with pytest.raises(ValueError, match="no eligible tasks"):
        derive_units(pool, spec, did, "smoke")
    pool = make_pool()
    pool.skills[DA]["diagnostic"] = []
    with pytest.raises(ValueError, match="diagnostic"):
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
