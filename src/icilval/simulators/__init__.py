"""The simulator registry.

A skill names its simulator in `spec.json` (`skills.<skill>.simulator`). Everything that is
specific to a simulator - the policy class, the episode loop, the pool stage, how a unit's
instance is set up, how a demonstration is rendered, what its spec entry must carry - lives
behind `Simulator`, one package per simulator under this one. Generic code (the side runner,
unit derivation, the pool builder, the spec validator) looks the simulator up here and never
names one. Adding a simulator is a new package that calls `register` and an import below.

A benchmark that lives in another repository registers the same way, from its own distribution,
found through the `icilval.benchmarks` entry point group (`load_plugins`). The registry is the
one place that knows the difference, and it knows it only as `Simulator.distribution`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..benchmarks.subprocess_runner import demo_frames_from, make_run_units

log = logging.getLogger(__name__)

#: Where a benchmark in another repository advertises itself. The entry point's name is the
#: simulator name a skill will carry; its value is a module that registers exactly that name.
ENTRY_POINT_GROUP = "icilval.benchmarks"


@dataclass(frozen=True)
class Simulator:
    name: str
    #: (model_dir, arch_dir, spec, skill, device) -> a loaded-on-demand policy (`PolicyBase`)
    make_policy: Callable[..., Any]
    #: (ctx, skill, policy, pool, units, spec, media_dir, record_video) -> None; runs each unit
    #: and hands `ctx.finish` a record (see `duel.side_runner.SideContext`)
    run_units: Callable[..., None]
    #: (pool, spec, src, skill, *, limit, validate, fetch_missing, evict_demos) -> None
    build_stage: Callable[..., None]
    #: (spec, skill, index, task, seed, rng) -> `pools.units.Unit`
    make_unit: Callable[..., Any]
    #: demo (npz arrays) -> frames for the demonstration clip
    demo_frames: Callable[[dict[str, Any]], list[Any]]
    #: (skill id, its spec.json entry) -> validation errors for what this simulator needs there
    validate_skill: Callable[[str, dict[str, Any]], list[str]]
    #: (pool, `pools.schema.PoolTask`) -> errors `pools.build.verify_pool` cannot check
    #: generically, because reading them needs this simulator's own file formats
    verify_pool_task: Callable[[Any, Any], list[str]] = field(default=lambda pool, task: [])
    #: The distribution providing it, named in the error when it turns out not to be installed
    distribution: str = "icilval"
    #: The `icilval.benchmarks` plugin object, when this simulator comes from a distribution
    #: that publishes one. In-repo simulators have none: their prompts are drawn from a pool
    #: built offline, so nothing asks them to materialize one.
    benchmark: Any = None
    #: channel -> the demonstration arrays that carry it, for `icilval.demoview`. A field keeps
    #: the channels its demonstration declares; an array claimed by no channel is never handed
    #: to a policy, so growing a new one cannot leak it into a restricted view by accident.
    demo_channels: dict[str, tuple[str, ...]] = field(default_factory=dict)


REGISTRY: dict[str, Simulator] = {}


def register(sim: Simulator) -> Simulator:
    if sim.name in REGISTRY:
        raise ValueError(f"simulator {sim.name!r} registered twice")
    REGISTRY[sim.name] = sim
    return sim


class MissingBenchmark(RuntimeError):
    """A skill names a simulator that no installed distribution provides."""


_LOADED = False


def load_plugins() -> tuple[str, ...]:
    """Import every installed distribution advertising `ENTRY_POINT_GROUP`; return what it added.

    An entry point's name is the simulator name and its value is a module that must register
    exactly that name. A distribution that registers nothing, or something else, raises rather
    than being skipped: a benchmark that half-installs must fail where it is installed, not
    later as an empty score.
    """
    global _LOADED
    from importlib.metadata import entry_points

    added: list[str] = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name in REGISTRY:
            continue
        try:
            loaded = ep.load()
        except Exception as exc:  # noqa: BLE001
            raise MissingBenchmark(
                f"the benchmark {ep.name!r} ({ep.value}) is installed but did not import: {exc}"
            ) from exc
        register(adapt(ep.name, _benchmark_of(ep, loaded), distribution=_distribution(ep)))
        added.append(ep.name)
    _LOADED = True
    return tuple(added)


def _benchmark_of(ep: Any, loaded: Any) -> Any:
    """The plugin object behind an entry point.

    A plugin may not import `icilval`, so it cannot register itself: it exposes an object, and the
    orchestrator adapts it. An entry point may name that object or the module holding it as
    `BENCHMARK`.
    """
    candidate = loaded if hasattr(loaded, "api_version") else getattr(loaded, "BENCHMARK", None)
    if candidate is None:
        raise MissingBenchmark(
            f"{ep.value} exposes no benchmark: expected the object itself, or a module with a "
            "`BENCHMARK` attribute"
        )
    from ..benchmarks import validate_plugin

    errors = validate_plugin(candidate)
    if errors:
        raise MissingBenchmark(f"{ep.value} is not a usable benchmark: {'; '.join(errors)}")
    if candidate.id != ep.name:
        raise MissingBenchmark(
            f"{ep.value} is advertised as {ep.name!r} but calls itself {candidate.id!r}"
        )
    return candidate


def _distribution(ep: Any) -> str:
    dist = getattr(ep, "dist", None)
    return getattr(dist, "name", None) or "unknown"


def adapt(name: str, benchmark: Any, *, distribution: str) -> Simulator:
    """Wrap a plugin object as the `Simulator` the rest of the validator dispatches on.

    An out-of-repo benchmark runs its simulator in a subprocess, so the hooks that would run one
    in *this* process raise rather than pretend. The orchestrator uses the plugin's command
    builders for that work, which is the whole point of the argv split.
    """

    def out_of_process(what: str):
        def refuse(*_a: Any, **_k: Any):
            raise MissingBenchmark(
                f"{name}: {what} runs out of process for a benchmark in another repository; "
                "the orchestrator drives it through the plugin's command builders"
            )

        return refuse

    info = benchmark.info() if hasattr(benchmark, "info") else {}
    channels = {k: tuple(v) for k, v in (info.get("demo_channels") or {}).items()}
    return Simulator(
        name=name,
        # A plugged benchmark's policy is *served*, not loaded here: the orchestrator owns the
        # weights and the architecture template, the benchmark's subprocess connects to it.
        make_policy=out_of_process("loading a policy"),
        # This one is no longer a refusal. `run_command` plus `read_result` is a complete
        # execution path, and leaving it refused was what kept this repository from being an
        # orchestration layer at all.
        run_units=make_run_units(name, benchmark),
        # A pool is built offline from a benchmark's own sources; a plugged benchmark that needs
        # one builds it on its own side. `prompts: "materialized"` fields need none.
        build_stage=out_of_process("building a pool"),
        make_unit=out_of_process("building a unit"),
        demo_frames=lambda demo: demo_frames_from(demo, channels),
        validate_skill=getattr(benchmark, "validate_skill", lambda skill, doc: []),
        distribution=distribution,
        benchmark=benchmark,
        demo_channels=channels,
    )


def _ensure_loaded() -> None:
    if not _LOADED:
        load_plugins()


def names() -> tuple[str, ...]:
    _ensure_loaded()
    return tuple(REGISTRY)


def installed(name: str) -> bool:
    _ensure_loaded()
    return name in REGISTRY


def get(name: str) -> Simulator:
    _ensure_loaded()
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown simulator {name!r}; registered: {', '.join(REGISTRY)}") from None


def for_skill(spec: Any, skill: str) -> Simulator:
    return get(spec.simulator(skill))


def require(spec: Any, skills: Any = None) -> None:
    """Raise before any work starts if a skill's benchmark is not installed.

    Called by the orchestrator, genesis and the pool builder. The message names the distribution
    to install, which the spec declares, because "unknown simulator 'robotwin'" tells an operator
    nothing about what to do next.
    """
    _ensure_loaded()
    missing: dict[str, list[str]] = {}
    for skill in skills if skills is not None else spec.all_skills:
        name = spec.simulator(skill)
        if name not in REGISTRY:
            missing.setdefault(name, []).append(skill)
    if not missing:
        return
    pins = benchmark_pins(spec)
    parts = []
    for name, affected in sorted(missing.items()):
        dist = (pins.get(name) or {}).get("distribution") or "an unknown distribution"
        parts.append(f"{name} (for {', '.join(affected)}): install {dist}")
    raise MissingBenchmark(
        "a benchmark is not installed; each advertises itself through the "
        f"{ENTRY_POINT_GROUP!r} entry point group. " + "; ".join(parts)
    )


def benchmark_pins(spec: Any) -> dict[str, dict[str, Any]]:
    """`spec.json`'s `benchmarks` block: what each simulator's distribution must be.

    Absent until the spec declares it, so this returns an empty mapping rather than raising -
    the in-repo simulators need no declaration to work.
    """
    raw = getattr(spec, "raw", None) or {}
    block = raw.get("benchmarks") or {}
    return {k: dict(v) for k, v in block.items() if isinstance(v, dict)}


def audit(spec: Any) -> list[dict[str, Any]]:
    """One row per simulator a skill names: what is declared, and what is actually installed.

    The deploy check behind `icilval benchmarks list|verify`. Never raises: an operator asking
    what is wrong should be told everything at once.
    """
    _ensure_loaded()
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as dist_version

    pins = benchmark_pins(spec)
    wanted = {spec.simulator(skill) for skill in spec.all_skills} | set(pins)
    rows: list[dict[str, Any]] = []
    for name in sorted(wanted):
        pin = pins.get(name) or {}
        sim = REGISTRY.get(name)
        distribution = pin.get("distribution") or (sim.distribution if sim else None)
        row: dict[str, Any] = {
            "simulator": name,
            "declared": name in pins,
            "distribution": distribution,
            "installed": sim is not None,
            "version": None,
            "problems": [],
        }
        if sim is None:
            row["problems"].append("not installed")
        if distribution and distribution != "icilval":
            try:
                row["version"] = dist_version(distribution)
            except PackageNotFoundError:
                row["problems"].append(f"distribution {distribution} not found")
        pinned = pin.get("version")
        if pinned and row["version"] and pinned != row["version"]:
            row["problems"].append(f"version {row['version']} is not the pinned {pinned}")
        row["skills"] = [s for s in spec.all_skills if spec.simulator(s) == name]
        rows.append(row)
    return rows


def make_policy(model_dir: Any, arch_dir: Any, spec: Any, skill: str, device: str = "cuda") -> Any:
    """The policy for a skill, from its simulator."""
    return for_skill(spec, skill).make_policy(model_dir, arch_dir, spec, skill, device=device)


# No simulator is imported here. This validator ships no benchmark: what it can score is exactly
# what is plugged into it through the `icilval.benchmarks` entry point group, and `load_plugins`
# finds that. A benchmark living here again would be a benchmark the orchestrator cannot be run
# without, which is the thing being undone.
