"""The wire format between the orchestrator and a benchmark running one of its units.

The ABI says `run_command` drives "a policy already served at `policy_address`". This is what is
spoken to that address.

**Why the orchestrator serves rather than the benchmark.** The weights are an entrant's
submission and the network around them is an architecture template this repository owns and
fingerprints. Handing a benchmark the weights would put the one thing the competition must
control inside the thing it does not. So the policy stays here, in a process built from the
template, and the benchmark connects to it.

**Why not reuse a benchmark's own transport.** RoboTwin has a good one, but it carries its own
`Demonstration`, `Frame` and `Observation` dataclasses. An orchestrator that spoke it would know
one benchmark's types, which is the coupling this whole design exists to avoid. So this format
carries **named arrays and nothing else**: the orchestrator moves `rgb`, `qpos`, `actions` without
knowing what any of them mean, and the benchmark and the architecture template agree on the names
between themselves.

**What it deliberately cannot do.** It never pickles - `Connection.send`/`recv` do, and unpickling
runs code, so only `send_bytes`/`recv_bytes` are used. Arrays are bool, integer or float; an
object dtype is refused at both ends. A message is one JSON header frame followed by one raw frame
per array, in the order the header lists them, so a malformed message is detected before any of it
is interpreted.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

#: Bumped whenever a message changes shape. Both ends check it on `hello` and refuse a mismatch
#: rather than discovering it halfway through a duel.
PROTOCOL_VERSION = 1

#: Every dtype an array may have on the wire, little-endian. Anything else - object above all - is
#: refused, on the way out as well as the way in.
DTYPES = frozenset(
    np.dtype(name).newbyteorder("<").str
    for name in (
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    )
)

#: What a client may ask for. An unknown op is an error reply, never a guess.
OPS = ("hello", "seed", "reset", "prompt", "act", "close")


class WireError(RuntimeError):
    """A value that cannot be sent, or a message that arrived malformed."""


def encode(op: str, fields: dict[str, Any] | None = None, arrays: dict[str, Any] | None = None):
    """One message, as the frames to send in order.

    Returned rather than sent so that encoding cannot fail halfway through a send and leave the
    connection holding half a message - the failure mode that turns a bad array into a hang.
    """
    if op not in OPS and op not in ("ok", "error", "action"):
        raise WireError(f"unknown op {op!r}")
    described = []
    payloads = []
    for name, value in sorted((arrays or {}).items()):
        array = _checked(name, value)
        described.append({"name": name, "dtype": array.dtype.str, "shape": list(array.shape)})
        payloads.append(array.tobytes())
    header = {
        "protocol": PROTOCOL_VERSION,
        "op": op,
        "fields": fields or {},
        "arrays": described,
    }
    return [json.dumps(header, separators=(",", ":"), allow_nan=False).encode()] + payloads


def _checked(name: str, value: Any) -> np.ndarray:
    array = np.ascontiguousarray(value)
    if array.dtype.str.lstrip("<>|") not in {d.lstrip("<>|") for d in DTYPES}:
        raise WireError(f"{name}: dtype {array.dtype} cannot be sent")
    return array.astype(array.dtype.newbyteorder("<"), copy=False)


def send(
    conn: Any, op: str, fields: dict[str, Any] | None = None, arrays: dict[str, Any] | None = None
) -> None:
    for frame in encode(op, fields, arrays):
        conn.send_bytes(frame)


def recv(conn: Any) -> tuple[str, dict[str, Any], dict[str, np.ndarray]]:
    """The next message: its op, its fields and its arrays.

    The header is validated before a single array frame is read, so a message claiming an
    impossible array is refused rather than used to size an allocation.
    """
    try:
        header = json.loads(conn.recv_bytes().decode())
    except (ValueError, UnicodeDecodeError) as exc:
        raise WireError(f"malformed header: {exc}") from None
    if not isinstance(header, dict):
        raise WireError("header is not an object")
    if header.get("protocol") != PROTOCOL_VERSION:
        raise WireError(f"protocol {header.get('protocol')!r}, this end speaks {PROTOCOL_VERSION}")
    op = header.get("op")
    if not isinstance(op, str):
        raise WireError("header carries no op")
    described = header.get("arrays") or []
    if not isinstance(described, list):
        raise WireError("header arrays is not a list")

    arrays: dict[str, np.ndarray] = {}
    for entry in described:
        if not isinstance(entry, dict):
            raise WireError("array description is not an object")
        name, dtype, shape = entry.get("name"), entry.get("dtype"), entry.get("shape")
        if not isinstance(name, str) or not isinstance(dtype, str) or not isinstance(shape, list):
            raise WireError(f"array {name!r}: incomplete description")
        if dtype not in DTYPES:
            raise WireError(f"array {name!r}: dtype {dtype!r} is not one this format carries")
        if not all(isinstance(d, int) and d >= 0 for d in shape):
            raise WireError(f"array {name!r}: shape {shape!r} is not a shape")
        raw = conn.recv_bytes()
        expected = (
            int(np.prod(shape)) * np.dtype(dtype).itemsize if shape else np.dtype(dtype).itemsize
        )
        if len(raw) != expected:
            raise WireError(f"array {name!r}: {len(raw)} bytes for a {shape} {dtype}")
        arrays[name] = np.frombuffer(raw, dtype=dtype).reshape(shape)

    fields = header.get("fields") or {}
    if not isinstance(fields, dict):
        raise WireError("header fields is not an object")
    return op, fields, arrays


def pack_history(history: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """A list of observation mappings, as flat named arrays plus the index that rebuilds it.

    `<i>/<key>`, with the count in the fields. The observation names are the benchmark's and the
    template's to agree on; nothing here reads them.
    """
    arrays: dict[str, Any] = {}
    for i, obs in enumerate(history):
        for key, value in obs.items():
            arrays[f"{i}/{key}"] = value
    return {"steps": len(history)}, arrays


def unpack_history(fields: dict[str, Any], arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    steps = fields.get("steps")
    if not isinstance(steps, int) or steps < 0:
        raise WireError("act: the history length is missing")
    history: list[dict[str, Any]] = [{} for _ in range(steps)]
    for name, value in arrays.items():
        index, _, key = name.partition("/")
        if not key or not index.isdigit() or int(index) >= steps:
            raise WireError(f"act: {name!r} belongs to no observation")
        history[int(index)][key] = value
    return history
