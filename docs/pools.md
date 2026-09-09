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
