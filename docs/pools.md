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

## Generating a LIBERO prompt

`simulators/libero/generate.py` drives BPP's own generator (`TaskDemonstrationGenerator` in the
vendored `scripts/libero/generate_demonstrations.py`) for one prompt at a time: seeded attempts
until one succeeds - BPP resets the scene with the seed, lifts the grasp from the human
demonstration the task's metadata names, scripts the transport and the place with randomized
waypoints and records the states - then BPP's `create_dataset` renders both cameras and the
result is one npz in the demo format, with the seed, the attempt count and the BPP commit in its
`meta`. Measured on this host: ~10 s per attempt, +7-9 s to render, ~15 s to build the scene and
read the grasp poses, 50-100 % success per attempt depending on the task.

The generator runs headless (`PYNPUT_BACKEND=dummy`) under a **validator-owned LIBERO config**
(`libero_config`, `LIBERO_CONFIG_PATH`): `datasets` points at the raw cache's copy of the
grasp-source files (`raw/LIBERO-datasets/libero_spatial/*.hdf5`, fetched by the catalogue's
hashes with `ensure_grasp_sources`), every other path at the vendored checkout; `~/.libero` is
never read. LIBERO reads the config when it is imported, so a process that generates calls
`libero_config()` first - the simulator test sessions do. The task's vendored BDDL, next to
which BPP reads its metadata, must equal the catalogue's byte for byte.
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

## Generating a drawing

`simulators/draw/generate.py` is BPP's `procedural_generate_drawings.py` ported in process and
driven by one seeded NumPy generator: a target is a list of parts - BPP's lines, Bezier curves,
ovals and pen-up movements (`bpp`), the edges of a closed polygon (`polygon`), or one polyline per
skeleton stroke of a font character rendered with Pillow and thinned with scikit-image (`glyph`,
`DejaVuSans` through matplotlib's font files) - turned into 10 Hz pen actions at a sampled speed
with BPP's noise, inter-part delays and final hold, rotated onto a board at a sampled angle and
executed on `DrawEnv` (BPP's pen-up positioning first) while frames, pen state and actions are
recorded. One npz in the demo format comes out, with the family, seed, part count and character
in `meta`. About 1-5 s per drawing; 50-150 actions.

## Calibrating catalogue 2026.09-v4

`pool_id` `9cad7d49496441eedbfed8baa176de889438d9f6875842dec7ba2f97969c30b4` - 174 `pick_and_place`
tasks, 3 `draw_anything` tasks (one per family) and the 50 handmade drawings as the
`handmade_drawings` diagnostic; the pinned catalogue. The genesis is `robotensor/bpp-genesis` at
`e680d99f`, BPP's public checkpoints converted for the two skills.

A v4 unit is harder than a v3 unit in two ways: its prompt is a generated demonstration of a
fresh reset rather than a stored human or scripted one, and the scored scene carries one change
from the skill's menu. `scripts/baseline.py` runs the genesis over units built exactly as a duel
builds them (`--change <kind>` forces one entry of the menu on every unit, `--family` picks one
drawing family) and reports the rate per change and per family. The ranges in `spec.json` were
set from two passes of that sweep; the first published difficulty is deliberately mild.

### LIBERO change ranges

Pass 1, provisional ranges, 40 tasks spread over the 174 (one unit each):

| change | provisional range | genesis |
|---|---|---|
| `displace` | radius 0.05 m, min delta 0.03 m, yaw ±45°, clearance 0.08 m | 0.475 |
| `camera` | position ±0.05 m, angle ±5° | 0.641 |
| `lighting` | diffuse ×[0.6, 1.3], ambient +[0, 0.2], specular ×[0.5, 1.3], position ±0.5 m, direction ±0.25 rad, headlight ×[0.7, 1.2] | 0.658 |
| `observation` | brightness ×[0.8, 1.2], noise σ [0, 8] | 0.730 |
| `robot_pose` | joint noise 0.05 rad (LIBERO default 0.02) | 0.703 |

Two things the first pass taught. The `displace` applier initially voided 13 of 40 units: it
required the moved bowl to keep `clearance_m` from every object and to stay on the table even
when it had started on a fixture, so on the crowded scenes no candidate was acceptable. The
rule became: a candidate is refused only when it *closes in* on another object within the
clearance, and the table bound applies only to a bowl that started on the table; 16 candidates
are drawn per unit and the first acceptable one applied. Nothing is void at that rule (the
0.475 above is measured with it). And three of the sampled tasks never generated a
demonstration in any of the 8 seeded attempts, so every pass-1 sweep substituted three units
(`substituted_from` on the unit). Generating one prompt for every task of the catalogue found
ten such tasks: the ten LIBERO-Spatial itself ships, each placing the bowl on the plate, for
which BPP's metadata names no grasp source. Since #31 they lift their grasps from their own
teleoperation file and generate in one to five attempts; the other 164 generate in 1.7 attempts
on average (105 at the first seed, 2 at the eighth) at about 48 s per attempt.

Pass 2, every range roughly halved, 24 tasks spread over the 174. These are the published
ranges:

| change | published range | genesis |
|---|---|---|
| `displace` | radius 0.03 m, min delta 0.015 m, yaw ±22.5°, clearance 0.08 m | 0.667 |
| `camera` | position ±0.02 m, angle ±2° | 0.667 |
| `lighting` | diffuse ×[0.8, 1.2], ambient +[0, 0.1], specular ×[0.8, 1.2], position ±0.25 m, direction ±0.1 rad, headlight ×[0.85, 1.1] | 0.667 |
| `observation` | brightness ×[0.9, 1.1], noise σ [0, 4] | 0.667 |
| `robot_pose` | joint noise 0.035 rad | 0.583 |
| **skill (mean over the menu)** | | **0.650** |

A reference sweep of the same 24 tasks with a generated prompt and **no change at all** scores
0.667 as well, and misses the same eight units; four of the five kinds miss exactly those eight,
`robot_pose` two more. So at the published ranges a change costs the genesis nothing measurable
on this sample, and the whole distance from the v3 baseline (0.876 with stored prompts) is the
prompt: a demonstration generated for a fresh reset, rather than a human or scripted one
recorded on the scored scene's own initial state. The seven units missed under every kind all
pick the bowl off a fixture (the stove, the cookie box, the wooden cabinet). Sampling noise on
24 units is about ±0.1.

### Drawing families

Generated drawings, 60 units per family, the change drawn from the menu as in a duel
(`board_angle` or `pen_start`):

| family | genesis | best Chamfer p25 / p50 / p75 (px) |
|---|---|---|
| `bpp` (BPP's procedural parts) | 0.733 | 1.2 / 1.7 / 2.6 |
| `polygon` (one closed polygon, 3-6 vertices) | 0.817 | 0.8 / 1.2 / 2.0 |
| `glyph` (one character's skeleton, DejaVuSans) | 0.250 | 2.5 / 4.8 / 9.8 |
| **skill (mean over the families)** | **0.600** | |

Two readings. Per change, a turned board is the harder case on the two families the genesis
handles (`bpp` 0.54 against 0.91 for a moved pen start, `polygon` 0.79 against
0.86); for glyphs both are low (0.32 and 0.17). And the glyph family is hard for
the genesis at any character: over the 60 units only `Y`, `J`, `N` and `V` pass every time, and
the misses spread over most of the alphabet, with the multi-stroke letters (`A`, `H`, `K`, `Q`,
`R`, `Z`) missing by 8-30 px. The genesis was trained on BPP's procedural parts, and a glyph's
skeleton - several short strokes with pen lifts and sharp corners - is far from them. The family
stays in the menu at its published parameters; its rate is the third of the skill's mean that a
challenger can most improve on.

### The drawing threshold at 2.5 px

`skills.draw_anything.success.threshold` moved from 4 px to 2.5 px with v4 - a fifth of the
12 px pen. The v3 re-measurement below already showed that 4 px passed 0.90 of procedural
drawings and left the skill little headroom; on the generated families the genesis passes
0.600 at 2.5 px against 0.789 at 4 px. The handmade drawings, prompted with a stored human
demonstration, are no longer scored; their rate at the same rule is published on every record
as the `handmade_drawings` diagnostic.

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
