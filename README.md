# icilval

Validator for the RoboTensor **one-demonstration in-context imitation learning** competition.
A submission holds one Behavior Prompting Policy (BPP) checkpoint per **skill** — pick and place
(LIBERO) and draw anything (DrawAnything-Sim). For every unit the model is shown one demonstration
of the task, no language, and must do it from another initial state. Each skill
is one success rate; the final score is their mean; a challenger takes the crown by beating the
reigning model's average by `duel.score_margin` points on an identical unit list. Everything
published is signed and reproducible from `spec.json`, the pool id and the two model references.

- `spec.json`, `store-schema.json` — the contract (also vendored by the dashboard).
- `arch/` — the allow-listed architecture templates, one per skill, exported from the genesis checkpoints.
- `docs/` — protocol, submissions, pools, operations.
- `vendor/behavior_prompting` — BPP pinned as a git submodule (`git submodule update --init --recursive`).

## Quick start (organizer)

```bash
uv venv --python 3.10 .venv && uv pip install -e ".[dev]"          # host tools, no simulator
pytest -m "not sim and not gpu and not container"                   # pure tests

# simulator work happens in the BPP conda environment (see docs/pools.md)
MUJOCO_GL=egl icilval pools build --out pools/2026.09-v2
MUJOCO_GL=egl icilval convert-ckpt --ckpt libero_behavior_prompting.ckpt --arch-name bpp_libero_v1 --out models/genesis/pick_and_place --emit-arch arch
icilval convert-ckpt --ckpt drawanything_sim_behavior_prompting.ckpt --arch-name bpp_draw_v1 --out models/genesis/draw_anything --emit-arch arch
MUJOCO_GL=egl SDL_VIDEODRIVER=dummy icilval smoke --store /tmp/store --pool pools/smoke --model-dir models/genesis --same-model
icilval store verify /tmp/store
```

## Layout

```
src/icilval/
  spec.py canon.py ids.py rng.py        contract, canonical JSON + ed25519, ids, hash RNG
  model/   rotations prompt fingerprint convert bpp draw
  sim/     bddl libero_env episode draw_env draw_episode video
  pools/   schema sources validate build build_gen build_draw upgrade demos units hub
  duel/    score side_runner orchestrate
  store/   records writer verify mirror
  queue.py admin.py live.py daemon.py submission.py cli.py
```

Licensed under Apache-2.0. BPP, LIBERO and DrawAnything-Sim are MIT; LIBERO-PRO data is CC-BY-4.0.
