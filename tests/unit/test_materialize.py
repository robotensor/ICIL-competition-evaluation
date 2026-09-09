"""The materializing phase over fake generators: prompts, substitution, finalization, reports."""

import json
from types import SimpleNamespace

import numpy as np

from icilval.canon import sha256_file
from icilval.duel import materialize as M
from icilval.ids import ModelRef, duel_id
from icilval.pools.units import derive_units
from icilval.simulators import for_skill as real_for_skill

from .test_units import DA, PP, make_pool


def fake_simulators(fail_tasks, calls):
    """Simulators whose generator writes a tiny npz, or fails for `fail_tasks`."""

    def factory(spec, skill):
        real = real_for_skill(spec, skill)

        def generate_prompt(unit, task, pool, spec_, seeds, out_npz, ctx):
            calls.append((unit["unit_id"], unit["task"], tuple(seeds)))
            if task.task_id in fail_tasks:
                return {
                    "success": False,
                    "attempts": [{"seed": s, "success": False} for s in seeds],
                    "error": "grasp missed",
                }
            dims = 3 if skill == DA else 7
            out_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                out_npz,
                meta=json.dumps({"demo_id": unit["demo"], "steps": 5}),
                actions=np.zeros((5, dims), np.float32),
                boundary_angle=np.asarray(0.3),
                agent_pos=np.array([[120.0, 130.0]] * 5, np.float32),
            )
            return {
                "success": True,
                "attempts": [{"seed": seeds[0], "success": True}],
                "steps": 5,
                "sha256": sha256_file(out_npz),
            }

        return SimpleNamespace(
            name=real.name,
            make_unit=real.make_unit,
            generate_prompt=generate_prompt,
            finalize_unit=real.finalize_unit,
        )

    return factory


def _strip(units):
    return [{k: v for k, v in u.items() if k != "prompt_sha256"} for u in units]


def test_materialize_generates_substitutes_and_finalizes(spec, tmp_path, monkeypatch):
    pool = make_pool()
    did = duel_id(4, spec.track_id, ModelRef.make("a/b", "1" * 40), None)
    units = [u.as_dict() for u in derive_units(pool, spec, did, "smoke")]
    order = M.task_order(pool, spec, did, PP)
    fail = {order[0]}
    calls = []
    monkeypatch.setattr(M, "for_skill", fake_simulators(fail, calls))
    ctx = M.GenerationContext(tmp_path, tmp_path, tmp_path)
    out, report = M.materialize(units, pool, spec, did, tmp_path / "assets", ctx, inline=True)
    assert len(out) == len(units) and [u["unit_id"] for u in out] == [u["unit_id"] for u in units]
    min_delta = spec.changes(DA)["board_angle"]["min_delta_rad"]
    for u in out:
        assert u["prompt_sha256"] and (tmp_path / "assets" / f"{u['unit_id']}.npz").exists()
        assert u["prompt_sha256"] == sha256_file(tmp_path / "assets" / f"{u['unit_id']}.npz")
        assert u["generation"]["attempts"] >= 1 and u["demo"] == f"generated/{u['unit_id']}"
        if u["skill"] == DA:
            p = u["instance_params"]
            assert p["demo_angle_rad"] == 0.3 and p["family"]
            if u["change"]["kind"] == "board_angle":
                assert abs(p["angle_rad"] - 0.3) >= min_delta and p["cursor_px"] == [120, 130]
                assert u["change"]["delta_rad"] >= min_delta
            else:
                assert p["angle_rad"] == 0.3 and u["change"]["cursor_px"] == p["cursor_px"]
    subs = [u for u in out if u.get("substituted_from")]
    assert subs and all(u["substituted_from"] in fail and u["task"] not in fail for u in subs)
    for u in subs:
        assert u["task"] == order[(order.index(u["substituted_from"]) + 1) % len(order)]
        assert u["change"]["kind"] in spec.changes(PP) and u["bddl"] == pool.tasks[u["task"]].bddl
        first, second = [c for c in calls if c[0] == u["unit_id"]]
        assert first[1] in fail and second[1] == u["task"] and first[2] != second[2]
    summary = report.summary()
    assert summary[PP]["substituted"] == len(subs) and summary[PP]["failed"] == 0
    assert summary[DA]["generated"] == len([u for u in out if u["skill"] == DA])
    assert any(" -> " in n for n in report.notes()) and any(
        "materializing took" in n for n in report.notes()
    )
    again, _ = M.materialize(units, pool, spec, did, tmp_path / "assets2", ctx, inline=True)
    assert _strip(again) == _strip(out)


def test_materialize_gives_up_after_the_substitutions(spec, tmp_path, monkeypatch):
    pool = make_pool()
    did = duel_id(4, spec.track_id, ModelRef.make("a/b", "2" * 40), None)
    units = [u.as_dict() for u in derive_units(pool, spec, did, "smoke")]
    calls = []
    monkeypatch.setattr(M, "for_skill", fake_simulators(set(pool.eligible(PP)), calls))
    ctx = M.GenerationContext(tmp_path, tmp_path, tmp_path)
    out, report = M.materialize(units, pool, spec, did, tmp_path / "assets", ctx, inline=True)
    pp = [u for u in out if u["skill"] == PP]
    assert pp and all(u["prompt_sha256"] is None and u["generation"]["failed"] for u in pp)
    assert all(not (tmp_path / "assets" / f"{u['unit_id']}.npz").exists() for u in pp)
    assert report.summary()[PP]["failed"] == len(pp)
    assert all(u["prompt_sha256"] for u in out if u["skill"] == DA)
    assert any("no prompt" in n for n in report.notes())


def test_seeds_and_task_order(spec):
    pool = make_pool()
    did = duel_id(4, spec.track_id, ModelRef.make("a/b", "3" * 40), None)
    unit = {"skill": PP, "index": 2}
    first, second = M.seeds_for(spec, did, unit, 0), M.seeds_for(spec, did, unit, 1)
    n = spec.generation["max_attempts"]
    assert len(first) == n == len(second) and not set(first) & set(second)
    assert sorted(M.task_order(pool, spec, did, PP)) == pool.eligible(PP)
    assert M.task_order(pool, spec, did, PP) == M.task_order(pool, spec, did, PP)
