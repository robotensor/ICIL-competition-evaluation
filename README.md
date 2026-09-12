# icilval

Orchestration layer for the RoboTensor **one-demonstration in-context imitation learning**
competition: duels, scoring, the signed store and the dashboard feed. Benchmarks are plugins.
There are two **fields**, each with its own king, queue, lineage and crown:

| Field | What the model is shown | Where it starts | Benchmark |
|---|---|---|---|
| **Sensorimotor** | frames, the action trajectory and proprioception | a **different** initial state | LIBERO, DrawAnything-Sim |
| **Video-only** | the frames alone | the **identical** scene | RoboTwin 2.0 |

They close the same shortcut by different means. A model shown what the robot did, in the very
scene it is scored in, is being asked to copy a trajectory rather than imitate from watching — so
the sensorimotor field scores a state it did not demonstrate, and the video-only field withholds
the actions instead. See [`docs/protocol.md`](docs/protocol.md).

A submission is to one field: one checkpoint per skill of that field, weights only, no
participant code. Each skill is one success rate; a field's score is the mean over *its own*
skills; a challenger takes that field's crown by beating the reigning model's average by
`score_margin` points on an identical unit list. Everything published is signed and reproducible
from `spec.json`, the pool id and the two model references.

- `spec.json`, `store-schema.json` — the contract (also vendored by the dashboard).
- `arch/` — the allow-listed architecture templates, one per architecture, exported from the genesis checkpoints.
- `docs/` — protocol, submissions, benchmarks, pools, operations.
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
  pools/        schema sources build upgrade demos units hub
  duel/         score side_runner orchestrate
  store/        records writer verify mirror
  queue.py admin.py live.py daemon.py submission.py cli.py
```

Licensed under Apache-2.0. BPP, LIBERO, LIBERO-Gen and DrawAnything-Sim are MIT.
