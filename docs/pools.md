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

## Calibrating the drawing threshold

`skills.draw_anything.success.threshold` was chosen by running the converted BPP drawing
checkpoint over 80 units derived from pool 2026.09-v2 (eight `heavy` draws' worth of drawing
units; the per-unit best Chamfer distances are the `*_metric` values a duel publishes). The
distribution of the best Chamfer distance, in canvas pixels on a 12 px pen:

| percentile | 10 | 25 | 50 | 75 | 90 |
|---|---|---|---|---|---|
| best Chamfer (px) | 0.96 | 1.39 | 2.46 | 3.83 | 7.21 |

| threshold (px) | 2.5 | 3 | 3.5 | 4 | 5 | 6 | 12 |
|---|---|---|---|---|---|---|---|
| genesis success | 0.51 | 0.61 | 0.69 | 0.79 | 0.83 | 0.86 | 0.93 |

The threshold is 4 px - a third of the pen width, so a success is a faithful reproduction rather
than a rough one - where the baseline misses the long multi-part drawings (`long5_robot`,
`long3_horizon`, `f2`, `custom6`) and little else. Every episode finished within 400 steps; none
was void. Those units were drawn with a 15° minimum angle delta that spec v3 no longer imposes;
the calibration is re-checked when the v3 pool is built.
