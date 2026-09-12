# icilval — ICIL competition validator

Python 3.10, package `icilval` under `src/`. The **orchestration layer** of the competition:
duels, scoring, the signed store, the dashboard feed. The benchmarks it scores on are plugins -
LIBERO and DrawAnything-Sim ship here, anything else is its own distribution found through the
`icilval.benchmarks` entry point group.

Two **fields** (`spec.json` `tracks`), each with its own king, queue, lineage, skills, baseline
and duel sizes:

- **sensorimotor** - the model is shown frames, actions and proprioception, and scored from a
  *different* initial state.
- **video_only** - the model is shown frames alone, and scored from the *identical* scene.

They close the same shortcut - replaying the demonstration - in opposite ways, and confusing the
two is the mistake `validate_spec` exists to refuse. A unit is one task, one initial state, one
prompt demonstration and a seed, spread evenly over a skill's eligible tasks. A field's score is
the mean over its own skills; the two fields' numbers are never combined. Follows the conventions
in `../CLAUDE.md`.

## Commands

- Host env (pure, no simulator): `uv venv .venv && uv pip install -e ".[dev]"`; `ruff check . && ruff format --check .`; `pytest -m "not sim and not gpu and not container"`.
- Simulator/GPU env: the pinned BPP conda env `/root/miniconda3/envs/bp` (`BP=/root/miniconda3/envs/bp/bin/python`, with `PYTHONPATH=src` and `SDL_VIDEODRIVER=dummy`); `$BP -m pytest -m sim`, `ICILVAL_TEST_GPU=1 $BP -m pytest -m gpu`.
- End to end: `icilval smoke --store <dir> --model-dir <genesis dir with one subdir per skill>` then `icilval store verify <dir>`.
- Container: `docker/build.sh`; `pytest -m container`.

## Rules

- `spec.json` and `store-schema.json` are the contract. No number from them is duplicated as a literal; read through `icilval.spec`.
- Submissions are `<skill>/model.safetensors` + `<skill>/config.yaml` per skill, nothing else. The validator never unpickles entrant data. Model-side runs happen in a container with `--network none`.
- Fields and skills are data: iterate `spec.tracks` and `spec.skills(track)`; never name a field,
  a skill or a simulator in generic code. `spec.all_skills` is every skill in the contract and is
  almost never what a duel wants - a duel scores one field's. `Spec.sole_track` is transitional
  and raises rather than guessing.
- What a policy may see of a demonstration is the field's, and enforced twice: `icilval.demoview`
  keeps the withheld arrays out of the mapping (an allow-list, so a new array cannot leak), and
  `icilval.arch` requires the architecture to have no input for them. Say what that is worth -
  no entrant code runs, so it guards against our mistakes, not an adversary.
- Skills are data: iterate `spec.all_skills`; never name a skill or a simulator in generic code. Everything simulator-specific lives in `simulators/<sim>/` behind the `Simulator` record it registers in `icilval.simulators`; generic code (`side_runner`, `pools.units`, `pools.build`, `pools.demos`, `spec.validate_spec`) looks the simulator up by `skills.<skill>.simulator`. Adding a simulator is a new package plus one import in `simulators/__init__.py`.
- Stored scores are fractions `[0, 1]`; `duel.score_margin` is percentage points; convert only in `icilval.duel.score`.
- Simulator imports (`libero`, `robosuite`, `pygame`, `torch`, `behavior_prompting`) are function-local so the host CLI imports without them; a `simulators/<sim>/__init__.py` imports its own modules lazily for the same reason.
- Pools are built offline (`icilval pools build` / `pools upgrade`); pickled `.pruned_init`/hdf5/zarr are read only at build time; runtime reads npz.
- Everything published is deterministic from `spec.json` + `pool_id` + the two model refs: unit lists, seeds, ids.
- BPP is vendored as a pinned git submodule at `vendor/behavior_prompting` (with `deps/LIBERO`); do not import its runner/workspace/dataset code, only the policy/model classes, the LIBERO env utilities and `DrawEnv`. Its generator scripts run as subprocesses at build time.

## Conventions

- Small commits. One concern per commit (a rename, a schema change, a new stage, a doc update),
  never a whole issue in one commit. Each commit builds and passes the pure tests on its own, so
  the history bisects and reverts cleanly. Split mechanical moves from behaviour changes.
- Commit title: `(feat): …`, `(fix): …`, `(refactor): …`, `(docs): …`, `(test): …`, `(chore): …`;
  imperative, lower-case after the prefix, under 72 characters, no trailing period. Body: why the
  change, not what the diff shows; short bullets; `Refs #N` for the issue it advances,
  `Closes #N` only on the commit that finishes it.
- One branch per issue (`issue-N-short-slug`) off `main`; one PR per issue with `Closes #N`, tests
  and a CHANGELOG entry. Rebase, do not merge `main` into the branch.
- Published results are hosted: the signed store is mirrored to a Hugging Face dataset and pools
  are pushed to the hub (`store mirror`, `pools push`). What stays a plain file in the repository
  or a run directory is everything not published - run logs, side output, local records.
