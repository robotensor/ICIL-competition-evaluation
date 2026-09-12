# Benchmarks

A **benchmark** is what a skill is scored on: it builds the scene, produces the demonstration,
runs the rollout and says whether the task was done. LIBERO and DrawAnything-Sim ship inside this
repository; anything else is its own distribution, installed beside the validator and found
through an entry point.

Everything simulator-specific lives behind the `Simulator` record a package registers in
`icilval.simulators`. Generic code — the side runner, unit derivation, the pool builder, the spec
validator — looks a skill's simulator up by `skills.<skill>.simulator` and never names one. A
test fails if a simulator's name appears in `src/icilval` outside `simulators/`.

## The contract

`icilval.benchmarks.api` is the third contract of this repository, beside `spec.json` and
`store-schema.json`. It is deliberately **one-directional**: a benchmark must never import
`icilval`, because a benchmark that stands on its own cannot depend on the competition that
scores it. So there is nothing to subclass — `Benchmark` is a `Protocol`, and
`validate_plugin(obj)` checks a plugin structurally and returns everything wrong with it.

The surface splits in two, and the split is why the design works.

**Pure** — `info`, `catalogue`, `derive_units`, `verify_prompt`, `read_result`. These must import
and run with **no simulator, no assets and no GPU**: the validator host reads a catalogue, CI
checks a unit list, and a third party verifies a published prompt, none of them able to build a
scene. Keep every simulator import function-local, as this repository already does for LIBERO and
torch.

**Commands** — `materialize_command`, `run_command`. These return an **argv**, not a result.
Everything needing a simulator happens in a subprocess the orchestrator launches, so `icilval`
never imports SAPIEN or MuJoCo, and the simulator side can run in another image, or on another
host, without a line changing here.

Two consequences worth stating plainly:

- Unit derivation belongs to the benchmark, because only the benchmark knows what a unit of it
  means — a scene seed, an initial-state index, a drawing. It must be reproducible by anyone
  holding the published record, so derive from a hash of the seed material, never from a global
  RNG whose stream depends on a library version.
- `verify_prompt` must read the file rather than trust a manifest. It is what lets a third party
  confirm that the demonstration a duel used is the one that was published.

## Writing a plugin

```toml
[project]
name = "my-benchmark-icil"
dependencies = ["my-benchmark"]          # NOT icilval

[project.entry-points."icilval.benchmarks"]
my_benchmark = "my_benchmark_icil.plugin"
```

The entry point's **name** is the simulator name that `skills.<skill>.simulator` will carry, and
its value is a module that calls `icilval.simulators.register()` for exactly that name. A
distribution that registers nothing, or registers a different name, is refused loudly: a
half-installed benchmark must never be silently absent.

## Installing one, and finding out you have not

Discovery is lazy: the registry imports the entry point group the first time anything asks it
what simulators exist. Three different behaviours, deliberately:

| where | a benchmark that is not installed |
| --- | --- |
| `validate_spec` | **accepted.** Only the name shape and the `benchmarks` declaration are checked. This is what lets CI, a laptop and the dashboard's vendored copy validate the contract with no simulator anywhere. |
| `icilval benchmarks list\|verify`, `icilval spec validate --strict` | **reported**, with the distribution to install. The validator host's deploy check; `verify` exits non-zero. |
| a duel, a genesis, a pool build | **raises `MissingBenchmark`** before anything is fetched or written. |

The daemon logs and skips a track whose benchmark is missing rather than stopping: one absent
benchmark must not halt another field's queue, and must never be silently scored as empty.

```console
$ icilval benchmarks list
draw             icilval                            skills=draw_anything                ok
libero           icilval                            skills=pick_and_place,goal_chain    ok
robotwin         robotwin-icil-competition          skills=rt_stacking                  not installed
```

## `api_version`

Version 1 is **provisional**. It was designed against one benchmark, and the second will bend it.
The number lives on the plugin and is checked rather than assumed, so a mismatch is refused
before anything runs instead of failing somewhere expensive and quiet.
