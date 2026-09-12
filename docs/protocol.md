# Protocol

The competition has two **fields**. In both, the model is shown **one demonstration** of a task
and **no language**; what differs is what the demonstration contains and where the model is then
put. Every constant below comes from `spec.json`; nothing here is normative on its own.

| Field | What the model is shown | Where it starts | Benchmark |
|---|---|---|---|
| **Sensorimotor** | the frames, the action trajectory and the proprioception | a **different** initial state | LIBERO and DrawAnything-Sim |
| **Video-only** | the frames alone | the **identical** initial scene | RoboTwin 2.0 |

Each field has its own king, queue, lineage, skills, baseline and duel sizes. An entrant may
enter one or both; nothing is compared across them.

## Why the two differ, and why that is the design

A model that can see what the robot did, in the very scene it will be scored in, is being asked
to copy a trajectory. It is not being asked to imitate from watching, and a benchmark that lets
it copy measures almost nothing: RoboTwin's own replay oracle - which plays the demonstration's
actions back verbatim - scores 18/18 on the V1 suite and is documented as the harness ceiling.

So each field closes that shortcut, by a **different** mechanism:

- **Sensorimotor** shows the whole demonstration but scores a *different* initial state. The
  actions it was shown do not fit the episode it is in. This is `prompt_instance_disjoint`.
- **Video-only** scores the very scene it demonstrated, and withholds the *actions and
  proprioception*. There is nothing to replay.

Reading those as interchangeable is the most consequential mistake this contract can encode, so
`validate_spec` refuses a field that claims one while running the other.

Two consequences worth stating plainly:

- **A video-only score has no computable ceiling.** Remove the actions and the information needed
  to reproduce the expert's joint trajectory is simply not in the prompt; anything that recovers
  it has solved the field. A replay oracle bounds what is achievable on the sensorimotor field
  and bounds nothing here. A low score means the model failed *or* that nothing could yet do
  better, and those cannot be told apart from the number.
- **The video-only field publishes its prompts with the event, not before.** A demonstration of
  the scored scene *is* the answer for that scene, so a pool published in advance would be close
  to an answer key. Verification is immediate-after: anyone can check the bytes by hash and
  rebuild the scene from the published seed, they just cannot train on them first.

## Skills

A **skill** is what is scored. Each belongs to exactly one field, and has its own benchmark and
its own architecture (one checkpoint per skill in a submission). A skill's score is one success
rate over its units; a field's **final score** is the mean over *its own* skills.

| Skill | What the model does | Simulator / architecture | What a unit varies |
|---|---|---|---|
| `pick_and_place` | Grasp one object and place it at a destination: BPP's LIBERO-Gen Combination domain. Every task moves the black bowl from one pick location to one placement in the LIBERO-Spatial scene; the pool holds every combination BPP generated demonstrations for. | LIBERO (MuJoCo) / `bpp_libero_v1` | The task and one of its initial states; the prompt is another demonstration of the same task. |
| `goal_chain` | Do two things in order - open a drawer, turn on the stove or push a plate, then place an object: BPP's LIBERO-Gen Chain domain, every two-step chain BPP generated demonstrations for. | LIBERO (MuJoCo) / `bpp_libero_v1` | The task and one of its initial states; the prompt is another demonstration of the same chain. |
| `draw_anything` | Reproduce a drawing shown once: BPP's DrawAnything-Sim domain, its 50 human drawings and its 2000 procedural ones. The demonstration is someone drawing a shape on a square whiteboard; the model draws it again on a blank board. | DrawAnything-Sim (pygame/pymunk) / `bpp_draw_v1` | The drawing, the board angle (uniform in `environment.board_angle_range_rad`, the range `DrawEnv` samples from) and the pen start (`cursor_start_range_px`). |

The **video-only** field, on RoboTwin 2.0, all three on `uniskill_v1`:

| Skill | What the model does | What a unit varies |
|---|---|---|
| `rt_pick_and_place` | Watch one bimanual pick and place as frames, then do it from the identical scene. | The task and its scene seed. |
| `rt_stacking` | The same, stacking two objects. The longest horizon of the three. | The task and its scene seed. |
| `rt_press_push` | The same, pressing or pushing something. The shortest. | The task and its scene seed. |

They are namespaced because RoboTwin's own task table also has a pick-and-place category, and
skill ids and codes are unique across the whole contract so a unit id is too.

A **unit** is one skill + one task + one initial state + one prompt demonstration + one seed,
exactly what BPP's own runner evaluates. A skill's units are spread evenly over its eligible
tasks (`pools/units.py`). The initial state and, on the drawing board, the angle and pen start
are published on every unit (`instance`, `instance_params`); nothing is scored below the skill.
Both sides of a duel run the identical unit list.

Whether the demonstration starts from the scored initial state is the field's
`prompt_instance_disjoint`: false on the sensorimotor field, and true on the video-only one,
where the demonstration is of the very scene being scored.

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

Per skill, `success rate = successes / scored units` (void units excluded). A field's **final
score** is the mean of *its own* skill rates - never of every skill in the contract. Scores are
stored as fractions in `[0, 1]`.

The two fields' numbers are not comparable and are never combined. They run different benchmarks,
over different denominators, and answer different questions.

## Crown rule

One crown **per field**. The challenger takes it iff
`challenger.average >= king.average + score_margin / 100`, where the margin is the field's own.
Paired per-unit outcomes (challenger / king / tie) are published as a diagnostic and never decide
the crown. A copy of the reigning model scores identically, so it never clears the margin.

## Determinism

`duel_id = sha256(spec_version|track|challenger.key|challenger.revision|king.key|king.revision)`.
Unit lists are derived from the pool and the duel id with a sha256-counter generator, so a third
party holding the pool can regenerate them from the published record. Per-episode seeds are
`int(sha256(duel_id|skill|index)[:8], 16)`; the diffusion sampler is seeded with them. A drawing
instance's board angle and cursor start are a pure function of the task id and instance index
(`pools/units.py`, `draw_instance`), so the pool carries no initial-state files for that skill.

## What a policy is shown

A field declares a **demonstration view**, and it is enforced rather than agreed:

- The arrays a field withholds are never put in the mapping a policy is handed
  (`icilval.demoview`). A view names the channels it *keeps*, so an array claimed by no channel
  is dropped - a benchmark that grows a new one cannot leak it into a restricted view by default.
- The architecture must have no input for a withheld channel either
  (`icilval.arch`), and `model/fingerprint.py` pins every template value outside
  `model.mutable_keys`, so a submission cannot flip the switch back on.
- Every unit publishes the view it was read under and `handed_sha256`, a digest of exactly the
  arrays the policy received. Anyone holding the published prompt can recompute it.

Be accurate about what that is worth. No entrant code runs anywhere, so these guard against
organizer mistakes, template mistakes and future refactors - not against an adversary running
code. What is adversary-proof is narrower: the withheld bytes are not on the filesystem the model
container mounts, and the architecture has no input to read them with.

## What is published

For every duel: the signed index record, the full event (every unit with both outcomes, its
instance, the prompt length and, on the drawing skill, both sides' Chamfer distances) and three
clips per unit: the prompt demonstration, the reigning model's rollout and the challenger's
rollout. Drawing clips show the board with the demonstrated strokes overlaid in red and the
model's in blue. See `store-schema.json`.

## Adding a skill, or a field

A skill is an entry in `spec.json` `skills` (code, title, architecture, simulator, `tasks`,
environment, success rule), a claim on it by exactly one field's `skills` list, and an
architecture template under `arch/`.

A **field** is an entry in `tracks`: its demonstration view, its protocol, its skills, its
baseline, its pool or catalogue, and any duelling constants that differ from the defaults. It
brings its own king, queue, lineage and crown. See `docs/benchmarks.md` for the benchmark behind
it.

Two registries decide what runs a skill, and they are deliberately separate. The **simulator**
(`icilval.simulators`) supplies the pool stage, the unit instances and the episode loop; a new
one is a package under `simulators/` registering those, plus one import, or a distribution
advertising itself through the `icilval.benchmarks` entry point group. The **architecture**
(`icilval.model.architectures`) supplies the policy, because that is what the policy is built
from: `arch/<architecture>.cfg.json` instantiates it, `arch/<architecture>.tensors.json` checks
its weights, and the observation names it consumes are the ones that template declares. A
benchmark in another repository could never have supplied a policy - the orchestrator holds the
weights and *serves* the policy to it - which is why the key is the architecture and not the
simulator. Adding a skill bumps `spec_version`, since it changes every duel id and every
submission's layout.
