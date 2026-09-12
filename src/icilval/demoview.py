"""What a field's policies may see of a demonstration, and proof of what they were handed.

A field declares a **demonstration view**: `sensorimotor` shows the frames, the action trajectory
and the proprioception, and `video_only` shows the frames alone. Video-only is the point of the
second field - a model that is shown what the robot did, in the very scene it will be scored in,
is being asked to copy a trajectory rather than to imitate from watching.

Three things make this more than a convention:

- **The arrays are never in the dict.** `apply` builds a new mapping holding only the channels the
  view allows; nothing downstream can reach what was left out, because it was never put in.
- **It is an allow-list, not a deny-list.** A view names the channels it keeps, and an array
  belonging to no kept channel is dropped. So a benchmark that grows a new array does not leak it
  into a video-only prompt by default; it has to be claimed by a channel first.
- **What was handed over is published.** `handed_sha256` hashes exactly the arrays the policy
  received, so a third party holding the prompt can confirm no action array was among them.

Be accurate about the claim. No entrant code runs anywhere - a submission is weights loaded into
a template the validator instantiates - so this guards against organizer mistakes, template
mistakes and future refactors, not against an adversary running code. What is adversary-proof is
elsewhere: the bytes are not on the filesystem the model container mounts, and the video-only
architecture has no demonstration-action input at all.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Keys that are metadata about the demonstration rather than a channel of it. Kept under every
#: view: they carry no observation, and the side runner needs them to identify what it loaded.
#: A benchmark with its own metadata arrays names them in a `metadata` channel rather than
#: hoping this constant covers them - RoboTwin's frame timestamps (`times`) are the case that
#: showed the constant cannot: they are not an observation, every view needs them, and no view's
#: modalities list would ever claim them.
METADATA_KEYS = ("meta",)

#: The channel a benchmark puts its own always-kept arrays in. Not a modality: no field lists it
#: in `demonstration.modalities`, and every view keeps it.
METADATA_CHANNEL = "metadata"

#: A channel entry ending in this is a **prefix**, matching every array whose name starts with the
#: rest of it. One benchmark array per camera (`frames_head_camera`, `frames_far_side_camera`, …)
#: cannot be enumerated by the orchestrator, which does not know a benchmark's camera list, so the
#: benchmark declares `frames_*` and the allow-list stays an allow-list.
PREFIX_MARK = "*"


@dataclass(frozen=True)
class DemoView:
    """One field's reading of a demonstration."""

    name: str
    #: The channels this view keeps, from the field's `demonstration.modalities`.
    keep: tuple[str, ...]
    #: The channels it does not, declared so the record can say what was withheld.
    withheld: tuple[str, ...]

    @property
    def is_restrictive(self) -> bool:
        return bool(self.withheld)


def view_for(spec: Any, track: str) -> DemoView:
    demo = spec.track(track)["demonstration"]
    return DemoView(
        name=str(demo["view"]),
        keep=tuple(demo["modalities"]),
        withheld=tuple(demo.get("withheld", ())),
    )


@dataclass(frozen=True)
class Allowance:
    """What a view admits: exact array names, and prefixes for families of them.

    Kept as a predicate rather than a set because a prefix cannot be enumerated - the orchestrator
    does not know how many cameras a benchmark records, which is the whole reason prefixes exist.
    """

    names: frozenset[str]
    prefixes: tuple[str, ...]

    def admits(self, key: str) -> bool:
        return key in self.names or (bool(self.prefixes) and key.startswith(self.prefixes))


def allowed_keys(channels: dict[str, tuple[str, ...]], view: DemoView) -> Allowance:
    """What a view permits, given a benchmark's channel map.

    The benchmark's `metadata` channel is kept under every view; the field's own modalities decide
    the rest. An entry ending in `*` is a prefix (see `PREFIX_MARK`).
    """
    names: set[str] = set(METADATA_KEYS)
    prefixes: list[str] = []
    for channel in (*view.keep, METADATA_CHANNEL):
        for entry in channels.get(channel, ()):
            if entry.endswith(PREFIX_MARK):
                prefixes.append(entry[: -len(PREFIX_MARK)])
            else:
                names.add(entry)
    return Allowance(names=frozenset(names), prefixes=tuple(sorted(prefixes)))


def apply(
    demo: dict[str, Any], channels: dict[str, tuple[str, ...]], view: DemoView
) -> dict[str, Any]:
    """A new demonstration holding only what `view` allows.

    Unknown arrays are dropped rather than kept: a benchmark that adds one must claim it in a
    channel before a policy can see it.
    """
    allowed = allowed_keys(channels, view)
    out = {k: v for k, v in demo.items() if allowed.admits(k)}
    meta = out.get("meta")
    if isinstance(meta, dict):
        out["meta"] = {**meta, "view": view.name}
    return out


def check(demo: dict[str, Any], channels: dict[str, tuple[str, ...]], view: DemoView) -> list[str]:
    """Anything in `demo` that this view should never have let through."""
    allowed = allowed_keys(channels, view)
    return sorted(
        f"{k}: withheld under the {view.name} view" for k in demo if not allowed.admits(k)
    )


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(f"{array.dtype.str}|{array.shape}|".encode())
    h.update(array.tobytes())
    return h.hexdigest()


def handed_sha256(demo: dict[str, Any]) -> str:
    """A digest of exactly the arrays a policy was handed.

    Order-independent and reproducible from the published prompt, so anyone holding the pool can
    recompute it and see what the model did - and did not - receive.
    """
    digests = {
        k: _array_digest(v)
        for k, v in sorted(demo.items())
        if k not in METADATA_KEYS and hasattr(v, "dtype")
    }
    payload = json.dumps(digests, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
