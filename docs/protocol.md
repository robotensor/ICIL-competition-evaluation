# Protocol

The competition scores a submission on one track: for every skill, the model is shown **one
demonstration** of a task and **no language**, then must perform the task on a scene that differs
from the demonstration by **one change**. The demonstration is **generated when the duel runs**;
no stored demonstration is ever drawn. Every constant below comes from `spec.json`; nothing here
is normative on its own.

## Skills

A **skill** is what is scored. Each has its own simulator and its own Behavior Prompting Policy
architecture (one checkpoint per skill in a submission). A skill's score is one success rate over
its units; the **final score** is the mean over skills.

| Skill | What the model does | Simulator / architecture | What a unit varies |
|---|---|---|---|
| `pick_and_place` | Grasp the black bowl and place it at a destination in the LIBERO-Spatial scene: the 174 LIBERO-Gen Combination tasks BPP generated demonstrations for (`skills.pick_and_place.tasks`). | LIBERO (MuJoCo) / `bpp_libero_v1` | The task, a generated demonstration, the scored initial state, and one change from the menu: `displace`, `camera`, `lighting`, `observation` or `robot_pose`. |
| `draw_anything` | Reproduce a drawing shown once. The target and its demonstration are generated from one of three primitive families (`skills.draw_anything.generation.families`): BPP's lines, curves and ovals; closed polygons; glyph skeletons of a font. | DrawAnything-Sim (pygame/pymunk) / `bpp_draw_v1` | The target, a generated demonstration, and one change from the menu: `board_angle` (at least `min_delta_rad` from the demonstration's angle) or `pen_start`. |

## A unit

A unit is one skill, one task and one seed. Everything else is derived from the duel id and the
seed, in this order:

1. **Prompt demonstration**, generated on an unchanged scene. On LIBERO the scene is reset with the
   prompt seed and BPP's `generate_demonstrations` records a demonstration: the grasp is lifted
   from the human teleoperation demonstration BPP's task metadata names
   (`tasks.grasp_sources`), the transport and the place are scripted with randomized
   waypoints, both cameras are rendered. On the board BPP's procedural generator draws a new
   target of one family and one demonstration of it at a random board angle.
2. **Scored episode**: the scene is reset with the unit seed to a different initial state and
   exactly **one change** from the skill's menu (`skills.<skill>.changes`) is applied, drawn
   uniformly over the menu with parameters uniform in the published ranges. The change and what
   it applied are published on the unit (`change`); the board's angle, pen start and the
   target's family on `instance_params`.

Both sides of a duel run the identical unit list with identical prompts and changes. The prompt
never starts from the scored state (`prompt_instance_disjoint`).

A skill's units are spread evenly over its catalogue tasks (`pools/units.py`).

### Change menus

`pick_and_place` - ranges in `skills.pick_and_place.changes`:

| change | what is randomized |
|---|---|
| `displace` | the target bowl moves by at least `min_delta_m` within `radius_m`, turned by up to `yaw_max_rad`, never onto another object's named location; the goal must still be unsatisfied |
| `camera` | the agentview camera's position (`pos_jitter_m`) and orientation (`angle_jitter_rad`); the wrist camera is untouched |
| `lighting` | every light's diffuse, ambient and specular terms, position and direction, and the headlight |
| `observation` | brightness and Gaussian pixel noise on the images the policy sees; clips keep the rendered frame |
| `robot_pose` | the arm's joint initialization noise (`init_noise_magnitude`) |

`draw_anything`:

| change | what is randomized |
|---|---|
| `board_angle` | the board angle, uniform in `environment.board_angle_range_rad` and at least `changes.board_angle.min_delta_rad` from the demonstration's; the pen starts where the demonstration's did |
| `pen_start` | the pen start, uniform in `environment.cursor_start_range_px`; the board keeps the demonstration's angle |

Ranges are public and were set so that the genesis checkpoint scores about 0.8 on each skill
(`docs/pools.md`).

### Generation failures

A unit's prompt is attempted with seeds `generation.prompt_seed`
(`int(sha256(duel_id|skill|index|attempt)[:8], 16)`) up to `generation.max_attempts`. If none
succeeds, the unit takes the **next task** in the duel's shuffled task order and records the task
it was first derived for as `substituted_from`; the unit count per skill is always met.

## Success

- `pick_and_place`: every goal predicate of the task's BDDL holds at some step before the cap
  (`skills.pick_and_place.max_steps`).
- `draw_anything`: the symmetric Chamfer distance, in canvas pixels, between the demonstrated
  strokes (turned upright) and the drawn strokes (turned upright by the unit's board angle) is at
  most `skills.draw_anything.success.threshold` (2.5 px on a 12 px pen) at the best step of the
  episode. This is BPP's own reward, negated; the best value is published on the unit as
  `*_metric`. An episode also ends when the predicted pen positions stay within `idle_stop_px`
  for `idle_stop_steps` steps, as in BPP's runner.

## Scoring

Per skill, `success rate = successes / scored units` (void units excluded). The
**final score** is the mean of the skill rates. Scores are stored as fractions in `[0, 1]`.

Published beside the scores, never part of them:

- **Sub-scores** (`sub_scores`): per side and skill, the success rate per value of
  `skills.<skill>.sub_scores.by` - the change kind on `pick_and_place`, the primitive family on
  `draw_anything`.

## Crown rule

The challenger takes the crown iff `challenger.average >= king.average + duel.score_margin / 100`.
Paired per-unit outcomes (challenger / king / tie) are published as a diagnostic and never decide
the crown. A copy of the reigning model scores identically, so it never clears the margin.

## Determinism and verification

`duel_id = sha256(spec_version|track|challenger.key|challenger.revision|king.key|king.revision)`.
Unit lists are derived from the catalogue and the duel id with a sha256-counter generator; per-unit
seeds are `int(sha256(duel_id|skill|index)[:8], 16)` and prompt attempts use
`generation.prompt_seed`. The diffusion sampler is seeded with the unit seed.

The generated prompt of every unit is published (`media/<sha[:2]>/<sha>.npz`, its sha256 on the
unit as `prompt.sha256`) beside its clip. Verification is **by hash**: anyone can check that the
published prompt is the one both sides saw and re-run a model against it. Bit-exact regeneration
of a prompt is not claimed, since MuJoCo and EGL rendering are deterministic only on identical
hardware.

## What is published

For every duel: the signed index record (with `sub_scores`), the full event
(every unit with both outcomes, its change, its instance parameters, its prompt's hash and length
and, on the drawing skill, both sides' Chamfer distances) and, per unit, the prompt npz and three
clips: the prompt demonstration, the reigning model's rollout and the challenger's rollout.
Drawing clips show the board with the demonstrated strokes overlaid in red and the model's in
blue. See `store-schema.json`.

## Adding a skill

A skill is an entry in `spec.json` `skills` (code, title, architecture, simulator, `tasks`,
environment, `changes`, `sub_scores`, success rule) and an architecture template under `arch/`.
If its simulator is already registered (`icilval.simulators`), that is all: the simulator's
package supplies the catalogue stage, the prompt generator, the changes, the episode loop and the
policy. A new simulator is a new package under `simulators/` registering those things, plus one
import. Adding a skill bumps `spec_version`, since it changes every duel id and every submission's
layout.
