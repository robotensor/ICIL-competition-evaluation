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
METADATA_KEYS = ("meta",)


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


def allowed_keys(channels: dict[str, tuple[str, ...]], view: DemoView) -> set[str]:
    """The array names a view permits, given a benchmark's channel map."""
    keys: set[str] = set(METADATA_KEYS)
    for channel in view.keep:
        keys.update(channels.get(channel, ()))
    return keys


def apply(
    demo: dict[str, Any], channels: dict[str, tuple[str, ...]], view: DemoView
) -> dict[str, Any]:
    """A new demonstration holding only what `view` allows.

    Unknown arrays are dropped rather than kept: a benchmark that adds one must claim it in a
    channel before a policy can see it.
    """
    allowed = allowed_keys(channels, view)
    out = {k: v for k, v in demo.items() if k in allowed}
    meta = out.get("meta")
    if isinstance(meta, dict):
        out["meta"] = {**meta, "view": view.name}
    return out


def check(demo: dict[str, Any], channels: dict[str, tuple[str, ...]], view: DemoView) -> list[str]:
    """Anything in `demo` that this view should never have let through."""
    allowed = allowed_keys(channels, view)
    return sorted(f"{k}: withheld under the {view.name} view" for k in demo if k not in allowed)


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
