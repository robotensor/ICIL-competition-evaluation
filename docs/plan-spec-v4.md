# Spec v4: generated demonstrations and per-unit scene changes

Status: plan, not yet implemented. Every constant named here lands in `spec.json`; this document
explains the protocol and the work, it is not normative on its own.

## Why

Under spec v3 a unit draws its task, initial state and prompt demonstration from a fixed,
published pool. That pool is BPP's public training data: the pick-and-place tasks are the
LIBERO-Gen Combination release, demonstrations included, and the drawings are BPP's
`procedural_2000_10` training set. An entrant can train on the exact prompts and initial states a
duel will score. A fixed pool of any size has the same problem; only a pool that does not exist
until the duel runs does not.

Spec v4 therefore **generates every prompt demonstration at duel time** with BPP's own LIBERO-Gen
and DrawAnything-Sim generators, and scores each unit on a scene that differs from the
demonstration by one published, randomly drawn change.

## Protocol

### Skills

Two skills, scored as before (one success rate each, final score the mean):

| skill | task | simulator |
|---|---|---|
| `pick_and_place` | move the black bowl from one pick location to one placement in the LIBERO-Spatial scene: the 174 LIBERO-Gen Combination tasks BPP generated demonstrations for | LIBERO |
| `draw_anything` | reproduce a drawing shown once | DrawAnything-Sim |

`goal_chain` is removed.

### The catalogue replaces the pool

A duel no longer draws from stored demonstrations. The published, content-addressed catalogue
(`pool_id` seals it) holds only what generation needs:

- `pick_and_place`: the 174 BDDL files and BPP's `task_metadata.yaml` (which human demonstration
  each grasp is lifted from), plus the sha256 of the ten public LIBERO-Spatial teleoperation
  files those grasps come from (`yifengzhu-hf/LIBERO-datasets`, `libero_spatial/*_demo.hdf5`,
  5.9 GB, not re-hosted).
- `draw_anything`: the 50 human drawings of `eval_handmade.zarr`, kept as an **unscored
  diagnostic** (their rate is published, never enters a score), and the generator version.
- The per-unit change menu and its ranges (below).

### A unit

One skill, one task, one seed; everything else is derived from the duel id and the seed:

1. **Prompt demonstration**, generated: on LIBERO, `env.reset(seed_demo)` then BPP's
   `generate_demonstrations` (human grasp lifted from the source teleoperation demo, scripted
   transport and place with randomized waypoints), both cameras rendered at 128 px. On the board,
   BPP's procedural generator draws a new target and one demonstration of it at a random board
   angle. The demonstration is recorded on an **unchanged** scene.
2. **Scored episode**: `env.reset(seed_eval)` — a different initial state — with exactly **one
   change** drawn from the skill's menu applied to the scene. The change and its parameters are
   published on the unit (`instance_params`).
3. Both sides of a duel run the identical unit list with identical prompts and changes.

Prompt and scored state are disjoint by construction; `prompt_instance_disjoint` stays true.

### Change menu, `pick_and_place`

One entry per unit, uniform over the menu; ranges are public in `spec.json` and calibrated so
the genesis checkpoint scores about 0.8 on the skill (calibration recorded in `docs/pools.md`
as for v3).

| change | what is randomized | mechanism |
|---|---|---|
| `displace` | the target bowl moves by at least `min_delta_m` within `radius_m`, optional yaw; never onto another object's named location | joint qpos after reset, settle, verify the goal is not already satisfied |
| `camera` | agentview position and orientation within a small cone | `mj_model.cam_pos` / `cam_quat` |
| `lighting` | diffuse, ambient, specular scale, light position and direction jitter, headlight scale | `mj_model.light_*` |
| `observation` | brightness scale and Gaussian pixel noise on the policy's images | image post-processing on the policy input only; clips show the rendered frame |
| `robot_pose` | larger joint initialization noise | robosuite `initialization_noise` |

### Change menu, `draw_anything`

| change | what is randomized |
|---|---|
| `board_angle` | board angle uniform in `board_angle_range_rad`, at least `min_delta_rad` from the demonstration's angle |
| `pen_start` | cursor start uniform in `cursor_start_range_px` |

The red orientation edge stays. Generated targets come from three primitive families: BPP's
lines, curves and ovals; **closed polygons**; and **glyph skeletons** rendered from fonts
(letters and digits). Success is a symmetric Chamfer distance of at most **2.5 px** (pen 12 px).

### Scoring

Per skill, `success rate = successes / scored units`; final score the mean over skills; crown rule
unchanged. New: **sub-scores** per change kind (and, for drawing, per primitive family) are
published on every record and on the dashboard as diagnostics. They never decide the crown.

### Generation failures and substitution

The scripted grasp can miss. Each unit's prompt is attempted with seeds
`hash(duel_id | skill | index | attempt)` up to `generation.max_attempts`. If none succeeds, the
unit takes the **next task** in the duel's shuffled task order and records `substituted_from`;
the unit count per skill is always met. A task with no successful attempt in a duel is reported
in the event's notes.

### What is published

Per unit, in addition to schema 3: the generated prompt (`prompt.sha256`, the npz in `media/`,
~9 MB on LIBERO, ~1.2 MB on the board), the demonstration clip as today, the change kind and
parameters, `substituted_from`. Verification is **by hash**: anyone can check that the published
prompt is the one both sides saw and re-run a model against it. Bit-exact regeneration is not
claimed, since MuJoCo and EGL rendering are deterministic only on identical hardware.

## Pipeline

```
fetching -> checking -> materializing -> evaluating(challenger) -> evaluating(king) -> publishing
```

`materializing` runs after `derive_units` in `duel/orchestrate.py`, on the validator host in the
BPP environment, never in the model container: it generates every unit's prompt and change into a
per-duel assets directory under `/var/lib/icilval/runs/<event>/assets` (gocryptfs cannot be
bind-mounted), which both side containers mount read-only. The sides read prompts from there
instead of from `pool/demos/`.

Measured on this host (one worker, L40S shared with two other jobs):

| step | cost |
|---|---|
| LIBERO demonstration attempt (collection) | ~10 s; success 50-100 % per attempt depending on the task |
| LIBERO re-render pass (both cameras) | +7-9 s per demonstration |
| per-task environment start and grasp-pose loading | ~15 s |
| drawing task + demonstration | ~4.6 s |

A heavy duel (42 units per skill) materializes in about 15-25 min single-worker, 3-6 min with
four to six workers; well inside `duel_wall_seconds`. `side_wall_seconds` is unaffected.

## Work breakdown

One issue per line, one branch per issue, small commits; order is the dependency order.

1. **spec v4**: two skills, `catalogue` replaces `pools`, change menus and ranges, drawing
   threshold 2.5 px, `generation.max_attempts`, sub-score definitions; `spec_version = 4`.
   `store-schema.json` and `live` schema 4: `prompt.sha256`, `change`, `substituted_from`,
   `sub_scores`, diagnostic block for the handmade drawings.
2. **catalogue build**: `icilval catalogue build` writes BDDLs, task metadata, source-file hashes
   and the handmade drawings; `pool_id` semantics kept; `pools upgrade` retired.
3. **LIBERO generator wrapper**: an in-process wrapper around BPP's `TaskDemonstrationGenerator`
   (import the class, not the CLI): seeded reset, collection with cameras on, npz output in the
   pool demo format, no hdf5 round trip. Headless requirements: `PYNPUT_BACKEND=dummy`,
   `LIBERO_CONFIG_PATH`. Source teleoperation files fetched by hash to the raw cache.
4. **scene changes**: `simulators/libero/changes.py` with the five entries; the displacement and
   lighting code returns from git history (`a30e11a^`, `sim/perturb.py`, `sim/lighting.py`);
   every change records what it applied.
5. **drawing generator**: an in-process generator writing npz directly (the vendored
   `procedural_generate_drawings.py` at `ec29e62` passes `trajectory_speed_min/max` to a
   function that has no such parameters and fails; upstream HEAD has the same bug), plus the
   polygon and glyph families; minimum angle delta reinstated in `draw_unit`.
6. **materializing phase**: attempts, substitution, assets directory, docker mounts,
   `run-side --assets`; live frames report generation progress.
7. **calibration**: genesis sweep per change kind and per family; set ranges for ~0.8 per skill;
   `docs/pools.md` gets the tables as for v3.
8. **dashboard**: vocabulary from the new spec, change kind per unit, sub-score table, handmade
   diagnostic, prompt hash link.
9. **publish**: catalogue, two-skill genesis (`pick_and_place`, `draw_anything`), store rebuilt on
   schema 4 and mirrored with `--prune`.

Tests: pure tests for derivation, substitution and schema; `-m sim` for each change and for one
generated prompt per simulator; `-m gpu` parity (genesis on a generated prompt reproduces the same
trajectory on both sides); `-m container` for the assets mount.

## Risks

- Generation success rate varies by task; the substitution rule bounds the cost, and the
  per-task attempt statistics published in the notes show whether the catalogue needs pruning.
- The displacement change must respect LIBERO-Spatial's location-named tasks: the target bowl may
  not be moved to where the other bowl, the plate, the stove, the ramekin or the cookie box stands.
- Entrants can run the same public generators for training data. That is intended: the score
  measures a distribution, not a set.
