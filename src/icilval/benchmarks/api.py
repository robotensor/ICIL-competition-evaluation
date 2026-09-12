"""What a benchmark distribution must expose, and how this validator checks it.

A benchmark is a separate repository, installed as its own distribution and found through the
`icilval.benchmarks` entry point group. This module is the contract between the two, and it is
deliberately one-directional: a benchmark must never import `icilval`, because a benchmark that
stands on its own cannot depend on the competition that scores it. So `Benchmark` is a
`Protocol`, nothing is subclassed, and a plugin is checked structurally by `validate_plugin`.

The surface splits in two, and the split is the reason the design works:

`PURE_METHODS` must import and run with **no simulator, no assets and no GPU**. The validator
host, CI and a laptop all read a benchmark's catalogue, derive its unit list and verify a
prompt; none of them can build a scene. A plugin therefore keeps every simulator import
function-local, exactly as this repository already does for LIBERO and torch.

`COMMAND_METHODS` return an **argv**, not a result. Everything that needs a simulator happens in
a subprocess the orchestrator launches, so `icilval` never imports SAPIEN or MuJoCo, and the
simulator side can run in another image, or on another host, without a line changing here.

`api_version` is checked, and version 1 is **provisional**: it was designed against one
benchmark, and the second will bend it. Bumping it is expected, and is why the number is on the
plugin rather than implied.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

#: The ABI this validator speaks. A plugin declaring another number is refused rather than
#: called: the failure modes of a half-matching benchmark are silent and expensive.
BENCHMARK_API_VERSION = 1

#: Importable and callable with no simulator, no assets and no GPU.
PURE_METHODS = ("info", "catalogue", "derive_units", "verify_prompt", "read_result")

#: Return the argv of a subprocess that does need those things.
COMMAND_METHODS = ("materialize_command", "run_command")

METHODS = PURE_METHODS + COMMAND_METHODS

#: Keyword-only parameters each method must accept, so the orchestrator can call it by name.
REQUIRED_KEYWORDS: dict[str, tuple[str, ...]] = {
    "derive_units": ("seed_material", "count", "suite"),
    "verify_prompt": ("path", "unit"),
    "read_result": ("out_dir",),
    "materialize_command": ("unit", "out_dir"),
    "run_command": ("unit", "prompt", "out_dir", "policy_address"),
}


@runtime_checkable
class Benchmark(Protocol):
    """One benchmark, as the orchestrator sees it."""

    #: Stable identifier, matching the entry point's name and `skills.<skill>.simulator`.
    id: str
    #: The `BENCHMARK_API_VERSION` this plugin was written against.
    api_version: int

    # -- pure ---------------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """Identity and shape: id, api_version, protocol, the tracks it can serve, its action
        space, its cameras, and the commits it is pinned to. Recorded on every duel it runs."""

    def catalogue(self) -> dict[str, Any]:
        """What can be drawn from: suites, tasks and skill categories. No demonstrations - a
        catalogue is the menu, not the meal."""

    def derive_units(self, *, seed_material: str, count: int, suite: str) -> list[dict[str, Any]]:
        """`count` units, a pure function of its arguments.

        Only the benchmark knows what a unit of it means - a scene seed, an initial-state index,
        a drawing - which is why derivation lives here and not in the orchestrator. It must be
        reproducible by a third party holding the published record, so derive from a hash of
        `seed_material`, never from a global RNG whose stream depends on a library version.
        """

    def verify_prompt(self, *, path: str, unit: Mapping[str, Any]) -> dict[str, Any]:
        """Check a materialized prompt is the one `unit` asked for, without a simulator.

        Returns at least `{"ok": bool, "sha256": str, "problems": [str, ...]}`. This is what lets
        anyone holding the published prompt confirm it, so it must read the file rather than
        trust a manifest.
        """

    def read_result(self, *, out_dir: str) -> dict[str, Any]:
        """One unit's result, as written by `run_command`'s subprocess.

        Returns at least `{"success": bool | None, "void": bool, "steps": int | None,
        "error": str | None}`. `success` is None exactly when the unit is void.
        """

    # -- commands -----------------------------------------------------------------------

    def materialize_command(self, *, unit: Mapping[str, Any], out_dir: str) -> Sequence[str]:
        """The argv that produces one unit's prompt, its clip and its hash, in `out_dir`."""

    def run_command(
        self,
        *,
        unit: Mapping[str, Any],
        prompt: str,
        out_dir: str,
        policy_address: str,
        **extra: Any,
    ) -> Sequence[str]:
        """The argv that runs one unit against a policy already served at `policy_address`."""


def validate_plugin(obj: Any) -> list[str]:
    """Everything wrong with `obj` as a benchmark plugin, in the order a reader would find it.

    Structural, not nominal: a plugin cannot subclass anything of ours without importing us.
    Returns an empty list when the plugin is usable.
    """
    errors: list[str] = []

    identifier = getattr(obj, "id", None)
    if not isinstance(identifier, str) or not identifier:
        errors.append("id: expected a non-empty string")

    version = getattr(obj, "api_version", None)
    if not isinstance(version, int) or isinstance(version, bool):
        errors.append("api_version: expected an integer")
    elif version != BENCHMARK_API_VERSION:
        errors.append(
            f"api_version: speaks {version}, this validator speaks {BENCHMARK_API_VERSION}"
        )

    for name in METHODS:
        method = getattr(obj, name, None)
        if method is None:
            errors.append(f"{name}: missing")
            continue
        if not callable(method):
            errors.append(f"{name}: not callable")
            continue
        errors.extend(_keyword_errors(name, method))

    return errors


def _keyword_errors(name: str, method: Any) -> list[str]:
    """The keyword arguments the orchestrator passes, which `method` must accept by name."""
    required = REQUIRED_KEYWORDS.get(name, ())
    if not required:
        return []
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):  # a builtin or a C callable: take it on trust
        return []
    parameters = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return []
    accepted = {
        n
        for n, p in parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return [f"{name}: does not accept {n}" for n in required if n not in accepted]
