from icilval.duel.score import (
    SCORE_EPSILON,
    crown_moves,
    diagnostic_rates,
    lookup,
    paired_outcome,
    skill_scores,
    sub_scores,
    tally,
    verdict,
    void_fraction,
)

SKILLS = ("pick_and_place", "draw_anything")


def unit(skill, k, c, void=False, i=0):
    return {
        "unit_id": f"{skill[:2]}-{i:03d}",
        "skill": skill,
        "king_success": k,
        "challenger_success": c,
        "outcome": paired_outcome(k, c),
        "void": void,
    }


def test_paired_outcome():
    assert paired_outcome(True, True) == "tie"
    assert paired_outcome(False, False) == "tie"
    assert paired_outcome(False, True) == "challenger"
    assert paired_outcome(True, False) == "king"
    assert paired_outcome(None, True) == "tie"


def test_skill_scores_exclude_void_and_unscored():
    units = [
        unit("pick_and_place", True, False),
        unit("pick_and_place", True, True, void=True),
        unit("pick_and_place", None, True),
    ]
    k = skill_scores(units, "king", SKILLS)
    c = skill_scores(units, "challenger", SKILLS)
    assert k["pick_and_place"] == 1.0 and c["pick_and_place"] == 0.5
    assert k["draw_anything"] is None
    assert k["average"] == 1.0 and c["average"] == 0.5
    assert list(k) == ["pick_and_place", "draw_anything", "average"]


def test_crown_rule_boundary():
    # 0.60 and 0.66 average 0.63 ; king 0.60 -> +3 points exactly at the margin
    assert crown_moves(0.60, 0.63, 3.0)
    assert not crown_moves(0.60, 0.63 - 1e-6, 3.0)
    assert crown_moves(0.5, 0.5, 0.0)
    assert not crown_moves(None, 0.9, 3.0)
    assert not crown_moves(0.9, None, 3.0)
    assert SCORE_EPSILON < 1e-6


def test_verdict_average_and_margin():
    units = []
    i = 0
    for skill, (kw, cw) in zip(SKILLS, [(6, 8), (3, 4)], strict=True):
        for n in range(10):
            units.append(unit(skill, n < kw, n < cw, i=i))
            i += 1
    v = verdict(units, 3.0, SKILLS)
    assert v.king_scores["average"] == (0.6 + 0.3) / 2
    assert v.challenger_scores["average"] == (0.8 + 0.4) / 2
    assert v.dethroned and v.reason == "margin-met"
    assert round(v.delta_points, 6) == 15.0
    assert v.tally.wins == 2 + 1 and v.tally.losses == 0 and v.tally.decided == 3
    assert v.tally.ties == 17


def test_copy_of_king_never_moves_crown():
    units = [unit(s, n % 2 == 0, n % 2 == 0, i=n) for s in SKILLS for n in range(6)]
    v = verdict(units, 3.0, SKILLS)
    assert v.delta_points == 0.0 and not v.dethroned and v.reason == "short-of-margin"
    assert v.tally.decided == 0
    v0 = verdict(units, 0.0, SKILLS)
    assert (
        v0.dethroned
    )  # margin 0 means >= ; the copy ties and moves the crown, which is why margin > 0


def test_one_skill_unscored_averages_the_other():
    units = [unit("pick_and_place", True, True, i=n) for n in range(4)]
    v = verdict(units, 3.0, SKILLS)
    assert v.king_scores["draw_anything"] is None and v.king_scores["average"] == 1.0


def test_verdict_edge_cases():
    assert verdict([], 3.0, SKILLS).reason == "no-units"
    v = verdict([unit("pick_and_place", None, None)], 3.0, SKILLS)
    assert v.reason == "unscored" and not v.dethroned
    t = tally([unit("pick_and_place", True, False, void=True), unit("pick_and_place", False, True)])
    assert t.void == 1 and t.wins == 1 and t.decided == 1
    assert (
        void_fraction(
            [unit("draw_anything", True, True, void=True), unit("draw_anything", True, True)]
        )
        == 0.5
    )


def test_diagnostic_units_never_score_but_are_reported():
    units = [unit("draw_anything", True, False, i=0), unit("draw_anything", False, False, i=1)]
    units[1]["diagnostic"] = True
    units[1]["task"] = "drawanything_handmade/draw_x"
    assert skill_scores(units, "king", SKILLS)["draw_anything"] == 1.0
    assert skill_scores(units, "challenger", SKILLS)["draw_anything"] == 0.0
    diag = {"handmade_drawings": {"skill": "draw_anything", "group": "drawanything_handmade"}}
    assert diagnostic_rates(units, "king", diag) == {"handmade_drawings": 0.0}
    assert diagnostic_rates(units[:1], "king", diag) == {"handmade_drawings": None}
    assert verdict(units, 3.0, SKILLS).tally.units == 2


def test_sub_scores_group_by_dotted_key():
    pp = [
        unit("pick_and_place", True, True, i=0),
        unit("pick_and_place", True, False, i=1),
        unit("pick_and_place", False, True, i=2),
    ]
    pp[0]["change"] = {"kind": "camera", "pos": [0.0, 0.0, 0.0]}
    pp[1]["change"] = {"kind": "camera"}
    pp[2]["change"] = {"kind": "lighting"}
    da = [unit("draw_anything", True, True, i=3)]
    da[0]["instance_params"] = {"family": "glyph"}
    keys = {"pick_and_place": "change.kind", "draw_anything": "instance_params.family"}
    king = sub_scores(pp + da, "king", keys)
    assert king["pick_and_place"] == {"camera": 1.0, "lighting": 0.0}
    assert king["draw_anything"] == {"glyph": 1.0}
    assert sub_scores(pp + da, "challenger", keys)["pick_and_place"]["camera"] == 0.5
    assert lookup(pp[0], "change.kind") == "camera" and lookup(pp[0], "change.pos.x") is None
    pp[2].pop("change")
    assert "none" in sub_scores(pp, "king", keys)["pick_and_place"]
    pp[0]["diagnostic"] = True
    assert "camera" in sub_scores(pp, "king", keys)["pick_and_place"]  # pp[1] still counts
    assert sub_scores(pp[:1], "king", keys)["pick_and_place"] == {}
