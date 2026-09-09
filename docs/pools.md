# Catalogue

A catalogue is a content-addressed directory the validator generates units from. It holds task
*definitions*, not demonstrations: every prompt is generated when a duel runs.

```
catalogue.json     manifest (tasks, the eligible and diagnostic tasks per skill, the grasp-source
                   hashes, pool_id = sha256 of the rest)
bddl/<group>/…     LIBERO BDDL files, as BPP's LIBERO-Gen release ships them
demos/<task>/…     stored demonstrations of the diagnostic tasks only (the handmade drawings):
                   the board's frames + pen state + actions + the strokes drawn
```

Build (needs the BPP conda environment; raw inputs live under `~/.cache/icilval/raw`):

```bash
MUJOCO_GL=egl icilval catalogue build --out catalogues/<version> --version <version> --fetch \
    [--stage pick_and_place draw_anything finalize] [--limit N] [--no-validate]
icilval catalogue verify catalogues/<version>
icilval catalogue push catalogues/<version> --repo <owner>/icil-competition-pools
```

Stages are named after the skills, plus `finalize`; where a skill's tasks come from is
`spec.json` `skills.<skill>.tasks`. `--fetch` lists the tasks that have demonstrations in the
Hugging Face release and downloads their BDDL files (a few KB each); nothing else is downloaded.
Unless `--no-validate`, every LIBERO scene is built once and reset a few times to check that the
goal is not already satisfied. The pool id is pinned in `spec.json` (`catalogue.pool_id`); the
validator refuses a catalogue that does not match.

## What a skill's stage imports

**`pick_and_place`** - the tasks of BPP's LIBERO-Gen Combination release
(`austinpatel/libero_gen_spatial_combination_hdf5`) that have demonstrations, both views alike
(the view is recorded per task in `meta.source_split`). Per task: the BDDL as the release ships
it, and BPP's task metadata for the view - the execution steps and which human demonstration each
grasp is lifted from (`meta.grasps_from` or `meta.actions_from`) - read from the vendored BPP
checkout at the commit `spec.json` pins, since the release does not ship `task_metadata.yaml`.
No demonstration or initial-state file is imported: a unit's prompt is generated for it, and its
scored scene is a numbered reset of the BDDL (`instance_seed`). The stage also records the sha256
of the ten LIBERO-Spatial teleoperation files the grasps come from (`tasks.grasp_sources`, read
from the hub's tree listing), so a generator host can fetch them by hash.

**`draw_anything`** - one task per primitive family in `skills.draw_anything.generation.families`
(`drawanything_generated/<family>`); a unit of such a task gets its target and demonstration
generated. Beside them, the skill's diagnostic (`spec.json` `diagnostics.handmade_drawings`):
BPP's human-drawn evaluation set (`eval_handmade.zarr`, 50 drawings, `demos_per_task` stored
demonstrations each) imported under `drawanything_handmade/` and flagged `diagnostic`, so its
units are published and never scored.

A task is eligible when a prompt can be generated for it; diagnostic tasks are listed apart.

## Catalogue 2026.09-v4 (spec v4, schema 4)

`pool_id` `9cad7d49496441eedbfed8baa176de889438d9f6875842dec7ba2f97969c30b4` - 227 tasks, the pinned
catalogue, published as `pools/2026.09-v4` of `robotensor/icil-competition-pools`. Built on
2026-09-08 with `--fetch` (about 10 minutes; nothing but the BDDL files was downloaded).

| skill | tasks | source |
|---|---|---|
| pick_and_place | 174 | LIBERO-Gen Combination, both views (10 `selected_view` + 164 `_inverse_view`); every scene validated by reset |
| draw_anything | 3 generated families (`bpp`, `polygon`, `glyph`) + 50 `handmade_drawings` diagnostic tasks (250 stored demonstrations) | DrawAnything-Sim |

Grasp sources: the ten `libero_spatial/*_demo.hdf5` files of `yifengzhu-hf/LIBERO-datasets`,
recorded by sha256. The catalogue is 445 MB.

Genesis for this catalogue: `robotensor/bpp-genesis@e680d99fc79ac5d54d33e696a635514c4abfc8cf`,
one converted public checkpoint per skill (`pick_and_place`: 690,455,718 parameters,
`draw_anything`: 344,772,217).

## Earlier pools

Spec v1-v3 drew units from stored demonstrations; the pools below are kept for the record.

## Pool 2026.09-v3 (spec v3, schema 3)

`pool_id` `73a98b0821be1fdbb5fb7b08219be6041c644ba35d634a7068688aaf501db843` - 2378 tasks, 23530
demonstrations, the pinned pool. Built on 2026-09-08 with `--fetch --evict` (the LIBERO-Gen
releases are ~310 GB of hdf5; the pool keeps `demos_per_task` demonstrations of each task).

| skill | tasks | source |
|---|---|---|
| pick_and_place | 174 | LIBERO-Gen Combination, both views (10 `selected_view` + 164 `_inverse_view`) |
| goal_chain | 154 | LIBERO-Gen Chain, both chain views (10 `selected_view` + 144 `_inverse_view`) |
| draw_anything | 2050 | DrawAnything-Sim: 50 `eval_handmade` + 2000 `procedural_2000_10` |

Every task is eligible: each has at least one initial state where its goal is not already
satisfied, and 10 demonstrations. No LIBERO-Gen demonstration starts from an evaluation initial
state (`demo_init_index` is `null` throughout), so the prompt is never the scored episode.

Genesis for this pool: `robotensor/bpp-genesis@aa24179bcc6d18185b4b07c995bbd0d15ac10b8a`, one
converted public checkpoint per skill. Both LIBERO-Gen checkpoints instantiate
`arch/bpp_libero_v1` unchanged (967 tensors, 690,455,718 parameters each).

## Pool 2026.09-v2 (spec v2, schema 2)

`pool_id` `ae9645cdbc2ed24cdbea436d6677925070f0a4221c604a09e18c96dbd0dad83a` - 98 tasks, 730
demonstrations. Superseded by `2026.09-v3`; its pick-and-place tasks are the organizer selection
spec v3 replaces.

| skill | tasks |
|---|---|
| pick_and_place | 27 base (libero_spatial 10, libero_object 10, libero_goal 6, libero_10 1) + 21 object-swap (LIBERO-Gen) |
| draw_anything | 50 human drawings (eval_handmade) |

## Baseline on pool 2026.09-v3

`scripts/baseline.py` ran the pinned genesis over every eligible task of each skill: two initial
states per task on the LIBERO skills, one each on 300 of the 2050 drawings (spread evenly over the
list). Nothing was void.

| skill | success rate | episodes | tasks |
|---|---|---|---|
| pick_and_place | 0.876 | 348 | 174 |
| goal_chain | 0.860 | 308 | 154 |
| draw_anything | 0.900 | 300 | 300 of 2050 |
| **average over skills** | **0.879** | | |

Per source view, which is what the pool does not distinguish and a duel therefore does not either:

| skill | view BPP trained on | view BPP held out |
|---|---|---|
| pick_and_place | 0.875 (328 episodes) | 0.900 (20) |
| goal_chain | 0.861 (288) | 0.850 (20) |
| draw_anything | 0.911 procedural (292) | 0.500 handmade (8) |

On both LIBERO skills the held-out view is within noise of the trained one, so importing both
views costs nothing in difficulty and buys an order of magnitude more tasks. The drawing skill is
the exception: the baseline is much stronger on the procedural drawings it was trained on than on
the human-drawn ones, and since the pool is 97.6% procedural the skill's rate is essentially the
procedural rate. The handmade figure rests on 8 episodes, so treat it as a direction, not a
number.

## Calibrating the drawing threshold

`skills.draw_anything.success.threshold` is 4 px - a third of the 12 px pen, so a success is a
faithful reproduction rather than a rough one. It was chosen on pool `2026.09-v2`, whose drawing
tasks were the 50 human-drawn ones. Re-measured on `2026.09-v3` over the 300 sweep units, the
best Chamfer distance per unit is distributed:

| percentile | 10 | 25 | 50 | 75 | 90 |
|---|---|---|---|---|---|
| best Chamfer (px), v3 | 0.72 | 0.97 | 1.60 | 2.23 | 3.88 |
| best Chamfer (px), v2 (handmade only) | 0.96 | 1.39 | 2.46 | 3.83 | 7.21 |

| threshold (px) | 2.5 | 3 | 3.5 | 4 | 5 | 6 | 12 |
|---|---|---|---|---|---|---|---|
| genesis success, v3 | 0.777 | 0.840 | 0.883 | 0.900 | 0.930 | 0.957 | 0.973 |
| genesis success, v2 | 0.51 | 0.61 | 0.69 | 0.79 | 0.83 | 0.86 | 0.93 |

The distribution moved left: procedural drawings are shorter and simpler than the human-drawn
ones, so the same threshold now passes 0.900 rather than 0.79. The threshold stays at 4 px, which
keeps the meaning of a success unchanged - what moved is the pool, not the rule. The eight units
the baseline misses among the handmade drawings are the long multi-part ones (`custom1`,
`custom7_birds`, `long5_robot`, `star`), the same failure mode as on v2.

Worth knowing when reading a duel: the drawing skill now has less headroom above the baseline
than the two LIBERO skills.
