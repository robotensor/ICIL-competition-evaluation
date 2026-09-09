# Changelog

## Unreleased

### Spec v4: generated prompts and one change per unit (store schema 4, live schema 4)

- published: catalogue `2026.09-v4` (`9cad7d49…`, 227 tasks, 445 MB) as `pools/2026.09-v4` of
  `robotensor/icil-competition-pools`, and the two-skill genesis as
  `robotensor/bpp-genesis@e680d99fc79ac5d54d33e696a635514c4abfc8cf`; both pinned in `spec.json`
  (#22).
- (feat): `spec.json` version 4. Every unit's prompt demonstration is generated at duel time with
  BPP's own generators and the scored scene differs from it by exactly one change drawn from the
  skill's menu, so a skill carries `changes` (public ranges) and `sub_scores`; the spec gains a
  `generation` block, drawing `generation.families` (BPP primitives, polygons, glyphs), an
  unscored `diagnostics` block (BPP's 50 handmade drawings) and `catalogue` in place of `pools`.
  `goal_chain` leaves the skill list; the drawing threshold is 2.5 px (#15).
- (feat): store and live schema 4: a unit publishes its `change`, `substituted_from`,
  `diagnostic` flag and `prompt.sha256`; records carry `sub_scores` and `diagnostics`; live frames
  have a `materializing` phase (#15).
- (feat): scoring excludes diagnostic units, reports sub-scores per `sub_scores.by` group and
  the rate of each diagnostic; the crown still rests on the mean over skills (#15).
- (chore): `icilval pools upgrade` is retired; an old pool is refused with a pointer to the build
  command (#15).
- (docs): `docs/protocol.md` describes the v4 protocol; `docs/plan-spec-v4.md` holds the plan and
  work breakdown for the milestone (#14, #15).
- (feat): the pool becomes a **catalogue** (schema 4, `catalogue.json`): task definitions, not
  demonstrations. A LIBERO task is its BDDL as the release ships it plus BPP's task metadata for
  the view (execution steps, which human demonstration each grasp is lifted from) from the
  vendored checkout; the sha256 of the ten grasp-source teleoperation files is recorded from the
  hub's tree listing. Drawing tasks are one per primitive family; BPP's handmade set is imported
  only as the `handmade_drawings` diagnostic, flagged and listed apart. Unit derivation numbers
  the scored reset (`instance_seed`), names the prompt it will generate and appends each
  diagnostic's units, scaled with the duel size, after the scored ones (#16).
- (feat): `icilval catalogue build / verify / push / pull` replace `icilval pools`; `--evict`
  and `generate-draw` go; `Spec.catalogue` replaces `Spec.pools` (#16).
- (feat): `simulators/libero/generate.py` generates a LIBERO prompt for a catalogue task with
  BPP's `TaskDemonstrationGenerator`: seeded attempts until one succeeds, BPP's render pass, one
  npz in the demo format (seed, attempts and BPP commit in `meta`). It runs headless under a
  validator-owned LIBERO config pointing `datasets` at the grasp-source files fetched by the
  catalogue's hashes; the simulator and GPU test sessions install that config. The GPU parity
  test prompts the genesis with generated demonstrations from fresh resets (#17).
- (feat): `simulators/libero/changes.py`: the one change a scored LIBERO scene carries. Sampled
  from the skill's menu when the unit is derived (pure, from the unit's RNG; a `displace` draw
  carries candidate moves and the first the scene accepts - on the table, `clearance_m` from
  every other object, goal still unsatisfied - is applied); applied after the scored reset
  (`displace`, `camera`, `lighting` mutate the model, `observation` scales and noises the
  policy's images with a seeded generator, `robot_pose` builds the scene with larger joint
  noise). `LiberoEnv` snapshots its lights and cameras and restores them on every reset; the
  side record carries `change_applied` (#18).
- (feat): `simulators/draw/generate.py` generates a drawing target and its demonstration in
  process from one seed: BPP's procedural parts ported (`bpp`), closed polygons (`polygon`) and
  font-glyph skeleton strokes (`glyph`), BPP's actions conversion, board rotation and recording
  on `DrawEnv`; one npz in the demo format with family, seed and character in `meta`. Drawing
  units of generated tasks draw their change kind (`board_angle` or `pen_start`) from the menu.
  The `sim` extra lists Pillow, scikit-image and matplotlib for the glyph family (#19).
- (feat): the **materializing** phase (`duel/materialize.py`): between checking and evaluating,
  every scored unit's prompt is generated into `<run>/assets/<unit_id>.npz` by the skill's
  simulator in worker processes under the validator's LIBERO config, from the unit's prompt
  seeds (`generation.prompt_seed`); when none of `max_attempts` succeeds the unit takes the next
  task in the duel's shuffled order (up to three times) and records `substituted_from`; a
  drawing unit is finalized against its prompt (`board_angle` turns the board at least
  `min_delta_rad` from the demonstration's, `pen_start` keeps its angle). Both sides read prompts
  from the assets directory (mounted read-only into the container, `run-side --assets`); every
  prompt is published by hash and `store verify` checks it; the event notes carry the generation
  summary. The runtime gains the generation context (`--bpp-root`, `--raw`) and `--workers` (#20).

### Pluggable simulators

- (refactor): `icilval.simulators` is a registry of `Simulator` records (policy factory, unit
  runner, pool stage, unit-instance builder, demo frames, spec checks), one package per
  simulator under `simulators/`; the side runner, unit derivation, demo rendering, pool build
  and spec validation look a skill's simulator up and never name one. `sim/` and the
  per-simulator halves of `model/` and `pools/` moved under `simulators/libero/` and
  `simulators/draw/`; unit lists and pool ids are unchanged (#6).

### Spec v3: BPP's unit protocol (pool schema 3, store schema 3, live schema 3)

- baseline on pool `2026.09-v3` (`scripts/baseline.py`, 956 episodes, none void): pick_and_place
  0.876, goal_chain 0.860, draw_anything 0.900, average 0.879. On both LIBERO skills the view BPP
  held out scores within noise of the view it trained on (0.900 vs 0.875; 0.850 vs 0.861), so
  importing both views costs no difficulty; on drawing the gap is real (0.911 procedural vs 0.500
  handmade on 8 episodes). The 4 px Chamfer threshold is unchanged and re-measured: the genesis
  passes 0.900 of drawing units at it, against 0.79 on the v2 pool, because the distribution moved
  left with the procedural drawings. See `docs/pools.md`.
- published: pool `2026.09-v3` (`73a98b08…`, 2378 tasks, 23530 demonstrations) as
  `robotensor/icil-competition-pools`, and the three-skill genesis as
  `robotensor/bpp-genesis@aa24179bcc6d18185b4b07c995bbd0d15ac10b8a`; both pinned in `spec.json`.
  The store is rebuilt and mirrored with `--prune` after this lands, since schema 3 replaces the
  v2 records.

- (feat): a third skill, `goal_chain`: the two chain views of BPP's LIBERO-Gen Chain release
  (154 two-step tasks), scored like the others; the converted `austinpatel/liberogen_goal_chain`
  checkpoint is its genesis and fits `bpp_libero_v1` unchanged (#5). Pool stages are named after
  the skills and driven by `spec.json` `skills.<skill>.tasks` (dataset, views or files), so
  adding a LIBERO-Gen or drawing skill is a spec entry.
- (feat): the draw pool imports BPP's `procedural_2000_10` set (2000 drawings, 10 demonstrations
  each) beside the 50 handmade drawings, read straight from the zip; both are sampled alike (#4).
- (feat): pick and place is BPP's LIBERO-Gen Combination domain, both views (174 tasks), sampled
  uniformly; the original-LIBERO base tasks, the pick-and-place goal filter, the Goal-Chain
  first-step import and the organizer task generator (`affordance.yaml`, `pools generate`) are
  gone (#2). `pools build --fetch --evict` streams the 160 GB of demonstration files from the hub
  one task at a time; `demo_init_index` is filled at import (no LIBERO-Gen demonstration starts
  from an evaluation initial state). `scripts/baseline.py` sweeps a model over every eligible
  task for the numbers in `docs/pools.md`. The genesis for this skill is the converted
  `austinpatel/liberogen_spatial_combination` checkpoint, whose template is byte-identical to
  `bpp_libero_v1`.

- (feat): the organizer perturbation groups are gone. A unit is one task, one of its benchmark
  initial states, one prompt demonstration and a seed, spread evenly over a skill's eligible
  tasks - what BPP's own runner evaluates. Removed: the L1..L5 displacement ladder, LIBERO-PRO
  swap and pose variants, table swaps, lighting draws and the drawing board's 15° minimum angle
  delta; the board angle range moves to `environment.board_angle_range_rad`, the range `DrawEnv`
  samples from (#3).
- (feat): pool schema 3 has no variants and one eligible task list per skill; `pools upgrade`
  converts a schema-2 pool without re-simulation. Published units carry `instance_params`
  (empty on LIBERO; angle, pen start and demonstration angle on the drawing board) instead of a
  group, a variant and a perturbation object.
- (refactor): `sim/perturb.py`, `sim/lighting.py`, `sim/perturb_math.py` and the BDDL rewriters
  are deleted (about 1.4k lines); `sim/bddl.py` is a reader.
- `pools.pool_id` still pins pool `2026.09-v2` until the spec v3 pool is built and pinned at the
  end of the milestone.

### Skills (spec v2, store schema 2, live schema 2)

- (feat): the four perturbation axes become **skills**: `pick_and_place` (LIBERO, `bpp_libero_v1`) and `draw_anything` (DrawAnything-Sim, `bpp_draw_v1`). Each skill is one success rate; the final score is the mean over skills; a skill's units are spread evenly over its perturbation groups (spatial / environment / object; rotation), which are published per unit and never scored. Composition is dropped (BPP's separate chain domain).
- (feat): pick-and-place scope follows BPP's definition — one Grasp then one Place (`skills.pick_and_place.task_filter`): 27 base + 21 object-swap tasks survive from the v1 pool.
- (feat): DrawAnything-Sim skill: `sim/draw_env.py` (BPP's `DrawEnv` behind the LIBERO-shaped surface), `sim/draw_episode.py` (Chamfer-threshold success, BPP's idle stop), `model/draw.py`, `pools/build_draw.py` (the human-drawn evaluation set; organizer-generated drawings through BPP's generator), drawing instances derived from ids (no init files).
- (feat): submissions hold one directory per skill; `fingerprint.check_submission` reports per skill; `side_runner` loads each skill's checkpoint in turn; `convert-ckpt` renames `umi_day.` targets.
- (feat): `pools upgrade` migrates a schema-1 pool without re-simulation; pool `2026.09-v2` (`ae9645cd…`): 48 pick-and-place tasks, 50 drawings, 730 demonstrations.
- (feat): draw-board tests (`-m sim`) and a drawing parity test (`-m gpu`); the container image runs pygame headless.
- smoke (spec v2): `icilval smoke` on the two-skill genesis over the upgraded smoke pool publishes 2 signed events with 12 clips; both sides reproduce identical trajectories on both simulators (6 ties, Δ = 0, identical clip hashes, crown stays); `store verify` passes; the dashboard renders the store.
- parity (draw): the converted `austinpatel/drawanything_sim` checkpoint redraws human demonstrations on a turned board within 4 px on 0.79 of 80 calibration units (median best Chamfer 2.5 px).
- (feat): `icilval store mirror --prune` makes the dataset exactly the local store in one commit (upload every file, delete every stale path), for a store rebuilt after a schema change. Dotfiles (the validator lock) never leave the machine, and the repository's own files — `.gitattributes` and the dataset card — are not the store's to delete.
- published (spec v2): the two-skill genesis as `robotensor/bpp-genesis` (pinned in `baseline.revision`), pool `2026.09-v2` as `robotensor/icil-competition-pools`, and a fresh store — genesis plus the baseline against a copy of itself on the real pool — as `robotensor/icil-competition-results`, replacing the v1-schema records.
- container mode (spec v2): `icilval duel --docker-image icilval/model:dev` runs both sides of a two-skill duel inside the rebuilt image with `--network none` (6 units per side, 3 per skill, identical outcomes and Chamfer values on both sides, 12 clips, `store verify` OK); `pytest -m container` passes 3/3 including a headless `DrawEnv` reset.

- (feat): contract (`spec.json`, `store-schema.json`), ids, scoring, unit derivation, signed store, queue, admin intake, live reporter.
- (feat): BPP model path (converter, fingerprint, inference), simulator runner, pool builders, duel orchestration, mirror, daemon, Docker recipe.
- parity: converted `austinpatel/libero` checkpoint reaches 0.96 success on libero_spatial (10 tasks x 5 initial states, one prompt demo).
- smoke: `icilval smoke` (genesis + genesis-vs-genesis, size smoke, in-process) publishes 2 signed events with 16 clips; both sides reproduce identical trajectories (8 ties, Δ = 0, crown stays); `store verify` passes.
- container mode: `icilval duel --docker-image icilval/model:dev` runs both sides inside the image with `--network none` (8 units, 3 clips each, identical outcomes on both sides, `store verify` OK); `pytest -m container` passes 3/3.
- live path: `icilval smoke --live http://127.0.0.1:20202 --live-token …` delivered 22 frames to the dashboard's `/api/live` (all 200) and they stream back as `progress` events; a validator-built frame with per-axis progress and `recent_media` parses on the strict route.
- published: the genesis checkpoint as `robotensor/bpp-libero-genesis` and a smoke duel on pool `2026.09-v1` as the public dataset `robotensor/icil-competition-results`, which the dashboard reads.
