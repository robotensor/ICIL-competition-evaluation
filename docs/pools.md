# Pools

A pool is a content-addressed directory the validator draws units from:

```
pool.json          manifest (tasks, the eligible tasks per skill, pool_id = sha256 of the rest)
bddl/<group>/…     LIBERO BDDL files, as BPP's LIBERO-Gen release ships them
init/<group>/…     LIBERO initial states as npz (converted from pickled .pruned_init at build time)
demos/<task>/…     demonstrations as npz: LIBERO cameras + proprioception + actions + initial state,
                   or the drawing board's frames + pen state + actions + the strokes drawn
```

Build (needs the BPP conda environment; raw inputs live under `~/.cache/icilval/raw`):

```bash
MUJOCO_GL=egl icilval pools build --out pools/<version> --version <version> --fetch --evict \
    [--stage pick_and_place goal_chain draw_anything finalize] [--limit N]
icilval pools verify pools/<version>
icilval pools push pools/<version> --repo <owner>/icil-competition-pools
```

Stages are named after the skills, plus `finalize`; where a skill's tasks come from is
`spec.json` `skills.<skill>.tasks` (the Hugging Face dataset and its LIBERO-Gen views or
DrawAnything-Sim files), so a new LIBERO-Gen or drawing skill is a spec entry. `--fetch` downloads each task's files from the
Hugging Face dataset on demand and `--evict` deletes a demonstration hdf5 once its
`demos_per_task` demonstrations are npz in the pool - the Combination release alone is about
160 GB of hdf5, more than a build host has to hold. A task is eligible when it has at least one
usable initial state (the goal is not already satisfied there) and one demonstration. The pool id
is pinned in `spec.json` (`pools.pool_id`); the validator refuses a pool that does not match.

The pick-and-place tasks are BPP's LIBERO-Gen Combination release
(`austinpatel/libero_gen_spatial_combination_hdf5`), both of its views - the 10 combinations BPP
held out and the 164 it trained on - imported alike; the view is recorded per task in
`meta.source_split`. Each task has 50 initial states and the pool keeps 10 of its 50 demonstrations,
spread over the file. BPP collected those demonstrations by motion planning from states of its
own, so none coincides with an evaluation initial state; `demo_init_index` records that
(every entry `null`), and unit derivation would skip a coincidence if there were one.

The goal-chain tasks are the two chain views of BPP's LIBERO-Gen Chain release
(`austinpatel/libero_gen_goal_chain_hdf5`): the 10 chains BPP held out and the 144 it trained on,
154 two-step tasks, imported and sampled alike. Its first-step and second-step views are single
steps, not chains, and stay out.

The `draw_anything` stage imports both public DrawAnything-Sim sets (`austinpatel/drawanything_sim`):
the human-drawn evaluation set (`eval_handmade.zarr`, 50 drawings, 5 demonstrations each) and
the procedural training set (`procedural_2000_10.zarr.zip`, 2000 drawings, 10 each, read in place
from the zip through zarr's `ZipStore`), each under its own group and sampled alike. A drawing
task has no initial-state file: its `init_states_per_task` instances are board angles and
cursor starts derived from the task id.

Organizer-generated drawings, never published as training data: `icilval pools generate-draw
--base-seed <secret>` runs BPP's `procedural_generate_drawings.py` and imports its drawings under
`drawanything_generated/`.

Upgrading a schema-2 pool (perturbation groups and variants) keeps every task and its files
without re-simulating and drops the variants; it is a stop-gap for a pool built before spec v3,
not a way to get the v3 task set:

```bash
icilval pools upgrade --old pools/2026.09-v2 --out pools/<version>
```

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

`scripts/baseline.py` over every eligible task with the pinned genesis
(`robotensor/bpp-genesis@aa24179b…`): two initial states per task on the LIBERO skills, one each
on 300 drawing tasks spread evenly over the 2050. No episode was void.

| skill | success rate | episodes | tasks |
|---|---|---|---|
| pick_and_place | 0.876 | 348 | 174 |
| goal_chain | 0.860 | 308 | 154 |
| draw_anything | 0.900 | 300 | 300 of 2050 |
| **mean over skills** | **0.879** | | |

Per source split, which is published per task in `meta.source_split` and never scored:

| skill | split | rate | episodes |
|---|---|---|---|
| pick_and_place | `libero_spatial_selected_combinations_inverse_view` | 0.875 | 328 |
| pick_and_place | `libero_spatial_selected_combinations_view` | 0.900 | 20 |
| goal_chain | `libero_goal_chain_selected_inverse_view` | 0.861 | 288 |
| goal_chain | `libero_goal_chain_selected_view` | 0.850 | 20 |
| draw_anything | `drawanything_procedural_2000_10` | 0.911 | 292 |
| draw_anything | `drawanything_handmade` | 0.500 | 8 |

On both LIBERO skills the split BPP held out and the split it trained on score within about two
points of each other, which is why the pool samples them uniformly and the store never
distinguishes them. The drawing skill is the exception: the baseline reproduces a procedural
drawing far more often than a human-drawn one. The handmade sample here is only 8 episodes, too
small to read closely, but the gap is large enough to watch. Since the pool is 2000 procedural
drawings to 50 handmade, a drawing score is in practice a score on the procedural set.

## Calibrating the drawing threshold

`skills.draw_anything.success.threshold` is 4 px: a third of the 12 px pen, so a success is a
faithful reproduction rather than a rough one. It was set on pool `2026.09-v2` (handmade drawings
only) and re-checked on `2026.09-v3` over the 300 sweep units above. The best Chamfer distance per
unit, in canvas pixels:

| percentile | 10 | 25 | 50 | 75 | 90 |
|---|---|---|---|---|---|
| 2026.09-v2 (50 handmade) | 0.96 | 1.39 | 2.46 | 3.83 | 7.21 |
| 2026.09-v3 (2050, 97.5% procedural) | 0.72 | 0.97 | 1.59 | 2.24 | 3.89 |

| threshold (px) | 2.5 | 3 | 3.5 | 4 | 5 | 6 | 12 |
|---|---|---|---|---|---|---|---|
| genesis on v2 | 0.51 | 0.61 | 0.69 | 0.79 | 0.83 | 0.86 | 0.93 |
| genesis on v3 | 0.78 | 0.84 | 0.88 | 0.90 | 0.93 | 0.96 | 0.97 |

The distribution is tighter on v3 because procedural drawings are easier to reproduce than
human-drawn ones. The threshold stays at 4 px: the rationale is the pen width, not the difficulty
of a particular set, and moving it would change every published score for no principled reason.
The consequence to keep in view is headroom - the baseline already reproduces 0.90 of drawing
units within 4 px, so that skill discriminates less between strong entrants than the LIBERO
skills do. Tightening to 2.5 px would put the baseline at 0.78 and restore the spread; that is a
protocol decision, not a calibration one, and would bump `spec_version`.

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

`scripts/baseline.py` over every eligible task with the pinned genesis
(`robotensor/bpp-genesis@aa24179b…`): two initial states per task on the LIBERO skills, one each
on 300 drawing tasks spread evenly over the 2050. No episode was void.

| skill | success rate | episodes | tasks |
|---|---|---|---|
| pick_and_place | 0.876 | 348 | 174 |
| goal_chain | 0.860 | 308 | 154 |
| draw_anything | 0.900 | 300 | 300 of 2050 |
| **mean over skills** | **0.879** | | |

Per source split, which is published per task in `meta.source_split` and never scored:

| skill | split | rate | episodes |
|---|---|---|---|
| pick_and_place | `libero_spatial_selected_combinations_inverse_view` | 0.875 | 328 |
| pick_and_place | `libero_spatial_selected_combinations_view` | 0.900 | 20 |
| goal_chain | `libero_goal_chain_selected_inverse_view` | 0.861 | 288 |
| goal_chain | `libero_goal_chain_selected_view` | 0.850 | 20 |
| draw_anything | `drawanything_procedural_2000_10` | 0.911 | 292 |
| draw_anything | `drawanything_handmade` | 0.500 | 8 |

On both LIBERO skills the split BPP held out and the split it trained on score within about two
points of each other, which is why the pool samples them uniformly and the store never
distinguishes them. The drawing skill is the exception: the baseline reproduces a procedural
drawing far more often than a human-drawn one. The handmade sample here is only 8 episodes, too
small to read closely, but the gap is large enough to watch. Since the pool is 2000 procedural
drawings to 50 handmade, a drawing score is in practice a score on the procedural set.

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
