# Protocol

The competition scores a submission on one track: for every skill, the model is shown **one
demonstration** of a task and **no language**, then must perform the task from another initial
state. Every constant below comes from `spec.json`; nothing here is
normative on its own.

## Skills

A **skill** is what is scored. Each has its own simulator and its own Behavior Prompting Policy
architecture (one checkpoint per skill in a submission). A skill's score is one success rate over
its units; the **final score** is the mean over skills.

| Skill | What the model does | Simulator / architecture | What a unit varies |
|---|---|---|---|
| `pick_and_place` | Grasp one object and place it at a destination: BPP's LIBERO-Gen Combination domain. Every task moves the black bowl from one pick location to one placement in the LIBERO-Spatial scene; the pool holds every combination BPP generated demonstrations for. | LIBERO (MuJoCo) / `bpp_libero_v1` | The task and one of its initial states; the prompt is another demonstration of the same task. |
| `goal_chain` | Do two things in order - open a drawer, turn on the stove or push a plate, then place an object: BPP's LIBERO-Gen Chain domain, every two-step chain BPP generated demonstrations for. | LIBERO (MuJoCo) / `bpp_libero_v1` | The task and one of its initial states; the prompt is another demonstration of the same chain. |
| `draw_anything` | Reproduce a drawing shown once: BPP's DrawAnything-Sim domain, its 50 human drawings and its 2000 procedural ones. The demonstration is someone drawing a shape on a square whiteboard; the model draws it again on a blank board. | DrawAnything-Sim (pygame/pymunk) / `bpp_draw_v1` | The drawing, the board angle (uniform in `environment.board_angle_range_rad`, the range `DrawEnv` samples from) and the pen start (`cursor_start_range_px`). |

A **unit** is one skill + one task + one initial state + one prompt demonstration + one seed,
exactly what BPP's own runner evaluates. A skill's units are spread evenly over its eligible
tasks (`pools/units.py`). The initial state and, on the drawing board, the angle and pen start
are published on every unit (`instance`, `instance_params`); nothing is scored below the skill.
Both sides of a duel run the identical unit list. The demonstration never starts from the scored
initial state.

## Success

- `pick_and_place`, `goal_chain`: every goal predicate of the task's BDDL holds at some step
  before the cap (`skills.<skill>.max_steps`). A chain has two predicates; how many hold at best
  is published on the unit as `*_progress`, and the step the first one held at as well.
- `draw_anything`: the symmetric Chamfer distance, in canvas pixels, between the demonstrated
  strokes (turned upright) and the drawn strokes (turned upright by the unit's board angle) is at
  most `skills.draw_anything.success.threshold` at the best step of the episode. This is BPP's
  own reward, negated; the best value is published on the unit as `*_metric`. An episode also ends
  when the predicted pen positions stay within `idle_stop_px` for `idle_stop_steps` steps, as in
  BPP's runner. The threshold was calibrated on the genesis checkpoint (see `docs/pools.md`).

## Scoring

Per skill, `success rate = successes / scored units` (void units excluded). The **final score** is
the mean of the skill rates. Scores are stored as fractions in `[0, 1]`.

## Crown rule

The challenger takes the crown iff `challenger.average >= king.average + duel.score_margin / 100`.
Paired per-unit outcomes (challenger / king / tie) are published as a diagnostic and never decide
the crown. A copy of the reigning model scores identically, so it never clears the margin.

## Determinism

`duel_id = sha256(spec_version|track|challenger.key|challenger.revision|king.key|king.revision)`.
Unit lists are derived from the pool and the duel id with a sha256-counter generator, so a third
party holding the pool can regenerate them from the published record. Per-episode seeds are
`int(sha256(duel_id|skill|index)[:8], 16)`; the diffusion sampler is seeded with them. A drawing
instance's board angle and cursor start are a pure function of the task id and instance index
(`pools/units.py`, `draw_instance`), so the pool carries no initial-state files for that skill.

## What is published

For every duel: the signed index record, the full event (every unit with both outcomes, its
instance, the prompt length and, on the drawing skill, both sides' Chamfer distances) and three
clips per unit: the prompt demonstration, the reigning model's rollout and the challenger's
rollout. Drawing clips show the board with the demonstrated strokes overlaid in red and the
model's in blue. See `store-schema.json`.

## Adding a skill

A skill is an entry in `spec.json` `skills` (code, title, architecture, simulator, `tasks`,
environment, success rule) and an architecture template under `arch/`. If its simulator is
already registered (`icilval.simulators`), that is all: the simulator's package supplies the
pool stage, the unit instances, the episode loop and the policy. A new simulator is a new
package under `simulators/` registering those six things, plus one import. Adding a skill bumps
`spec_version`, since it changes every duel id and every submission's layout.
