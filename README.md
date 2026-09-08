# icilval

Validator for the RoboTensor **one-demonstration in-context imitation learning** competition.
A submission holds one Behavior Prompting Policy (BPP) checkpoint per **skill** — pick and place
(BPP's LIBERO-Gen Combination), goal chain (LIBERO-Gen Chain) and draw anything
(DrawAnything-Sim). For every unit the model is shown one demonstration of the task, no
language, and must do it from another initial state. Each skill
is one success rate; the final score is their mean; a challenger takes the crown by beating the
reigning model's average by `duel.score_margin` points on an identical unit list. Everything
published is signed and reproducible from `spec.json`, the pool id and the two model references.

- `spec.json`, `store-schema.json` — the contract (also vendored by the dashboard).
- `arch/` — the allow-listed architecture templates, one per architecture, exported from the genesis checkpoints.
- `docs/` — protocol, submissions, pools, operations.
- `vendor/behavior_prompting` — BPP pinned as a git submodule (`git submodule update --init --recursive`).

## Quick start (organizer)

```bash
uv venv --python 3.10 .venv && uv pip install -e ".[dev]"          # host tools, no simulator
pytest -m "not sim and not gpu and not container"                   # pure tests

# simulator work happens in the BPP conda environment (see docs/pools.md)
MUJOCO_GL=egl icilval pools build --out pools/<version> --version <version> --fetch --evict
MUJOCO_GL=egl icilval convert-ckpt --ckpt liberogen_spatial_combination_behavior_prompting.ckpt --arch-name bpp_libero_v1 --out models/genesis/pick_and_place
MUJOCO_GL=egl icilval convert-ckpt --ckpt liberogen_goal_chain_behavior_prompting.ckpt --arch-name bpp_libero_v1 --out models/genesis/goal_chain
icilval convert-ckpt --ckpt drawanything_sim_behavior_prompting.ckpt --arch-name bpp_draw_v1 --out models/genesis/draw_anything
MUJOCO_GL=egl SDL_VIDEODRIVER=dummy icilval smoke --store /tmp/store --pool pools/smoke --model-dir models/genesis --same-model
icilval store verify /tmp/store
```

## Layout

```
src/icilval/
  spec.py canon.py ids.py rng.py video.py   contract, canonical JSON + ed25519, ids, hash RNG, clips
  model/        policy (loading, tensors) prompt (chunking) fingerprint convert
  simulators/   the registry; one package per simulator, each registering its policy, unit
                runner, pool stage, unit instances, demo frames and spec checks:
    libero/     env episode bddl policy prompt rotations demos pool validate units run
    draw/       env episode policy prompt demos pool units run
  pools/        schema sources build demos units hub
  duel/         score side_runner orchestrate
  store/        records writer verify mirror
  queue.py admin.py live.py daemon.py submission.py cli.py
```

Licensed under Apache-2.0. BPP, LIBERO, LIBERO-Gen and DrawAnything-Sim are MIT.
