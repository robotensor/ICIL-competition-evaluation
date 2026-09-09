from icilval.ids import ModelRef, duel_id, event_id, model_key, unit_id, unit_seed
from icilval.rng import HashRng


def test_ids_are_stable():
    ref = ModelRef.make("owner/model", "0123456789abcdef0123456789abcdef01234567")
    assert ref.key == model_key("owner/model", "0123456789abcdef0123456789abcdef01234567")
    assert len(ref.key) == 16
    king = ModelRef.make("org/king", "abcdef0123456789abcdef0123456789abcdef01")
    did = duel_id(2, "icil_1demo", ref, king)
    assert len(did) == 64 and did == duel_id(2, "icil_1demo", ref, king)
    assert did != duel_id(1, "icil_1demo", ref, king)
    assert did != duel_id(2, "icil_1demo", ref, None)
    assert unit_seed(did, "pick_and_place", 0) != unit_seed(did, "pick_and_place", 1)
    assert unit_seed(did, "pick_and_place", 0) != unit_seed(did, "draw_anything", 0)
    assert 0 <= unit_seed(did, "draw_anything", 3) < 2**32
    assert unit_id("da", 7) == "da-007"
    assert event_id("duel", "icil_1demo", 4, did) != event_id("duel", "icil_1demo", 5, did)


def test_golden_values():
    # Frozen: a change here changes every published id.
    assert model_key("a/b", "c") == "fdd11077ff89f6bf"
    assert unit_seed("d", "pick_and_place", 0) == 2563984751


def test_hash_rng_determinism_and_range():
    a, b = HashRng("x", 1), HashRng("x", 1)
    assert [a.below(10) for _ in range(20)] == [b.below(10) for _ in range(20)]
    c = HashRng("x", 2)
    assert [a.below(10) for _ in range(20)] != [c.below(10) for _ in range(20)]
    r = HashRng("s")
    assert all(0 <= r.below(7) < 7 for _ in range(500))
    assert all(0.0 <= r.uniform() < 1.0 for _ in range(500))
    perm = HashRng("p").shuffled(list(range(50)))
    assert sorted(perm) == list(range(50)) and perm != list(range(50))


def test_prompt_seed_varies_with_every_part():
    from icilval.ids import prompt_seed

    a = prompt_seed("d" * 64, "pick_and_place", 3, 0)
    assert a == prompt_seed("d" * 64, "pick_and_place", 3, 0) and 0 <= a < (1 << 32)
    others = {
        prompt_seed("d" * 64, "pick_and_place", 3, 1),
        prompt_seed("d" * 64, "pick_and_place", 4, 0),
        prompt_seed("e" * 64, "pick_and_place", 3, 0),
        prompt_seed("d" * 64, "draw_anything", 3, 0),
    }
    assert len(others | {a}) == 5
