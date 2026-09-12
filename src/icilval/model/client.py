"""The other end of `icilval.model.host`, for a benchmark to **vendor**.

A benchmark must never import `icilval` - it has to stand on its own, and depending on the
competition that scores it would invert that. So this file is written to be copied: standard
library and numpy only, no imports from this package except the wire format, which is small
enough to copy alongside it.

It is here rather than only in the documentation because a protocol with no executable reference
drifts: the tests in this repository drive the real host through this client over a real socket,
so the thing a benchmark copies is the thing that is known to work.

    with PolicyClient(address, authkey) as policy:
        policy.seed(1234)
        policy.reset()
        info = policy.prompt({"frames_head": frames, "qpos": qpos})
        action = policy.act([{"rgb": rgb, "qpos": qpos}])
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import wire


class PolicyError(RuntimeError):
    """The served policy refused, failed, or is not there."""


class PolicyClient:
    """A policy served by the orchestrator, reached over `policy_address`."""

    def __init__(self, address: str, authkey: bytes, *, timeout_s: float = 600.0) -> None:
        from multiprocessing import AuthenticationError
        from multiprocessing.connection import Client

        family = "AF_UNIX" if address.startswith("/") else "AF_INET"
        target: Any = address
        if family == "AF_INET":
            host, _, port = address.rpartition(":")
            target = (host, int(port))
        try:
            self._conn = Client(target, family=family, authkey=authkey)
        except AuthenticationError as exc:
            # Not an OSError, and the one failure a caller is most likely to have caused itself.
            raise PolicyError(f"the policy at {address} refused this key: {exc}") from None
        except OSError as exc:
            raise PolicyError(f"cannot reach the policy at {address}: {exc}") from None
        self._timeout = timeout_s
        self.hello()

    # -- lifecycle ------------------------------------------------------------------

    def __enter__(self) -> PolicyClient:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._call("close")
        except PolicyError:
            pass
        finally:
            try:
                self._conn.close()
            except OSError:
                pass

    # -- operations -----------------------------------------------------------------

    def hello(self) -> dict[str, Any]:
        return self._call("hello")[0]

    def seed(self, seed: int) -> None:
        self._call("seed", {"seed": int(seed)})

    def reset(self) -> None:
        self._call("reset")

    def prompt(self, demo: dict[str, Any]) -> dict[str, Any]:
        """Hand over the demonstration. Whatever the field withholds is simply not in `demo`."""
        return self._call("prompt", arrays=demo)[0]

    def act(self, history: list[dict[str, Any]]) -> np.ndarray:
        fields, arrays = wire.pack_history(history)
        reply_fields, reply_arrays = self._call("act", fields, arrays)
        action = reply_arrays.get("action")
        if action is None:
            raise PolicyError(f"the policy returned no action: {reply_fields}")
        return action

    # -- transport ------------------------------------------------------------------

    def _call(
        self, op: str, fields: dict[str, Any] | None = None, arrays: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        try:
            wire.send(self._conn, op, fields, arrays)
            if not self._conn.poll(self._timeout):
                raise PolicyError(f"{op}: the policy did not answer within {self._timeout:.0f}s")
            reply_op, reply_fields, reply_arrays = wire.recv(self._conn)
        except (EOFError, ConnectionError, OSError) as exc:
            raise PolicyError(f"{op}: the policy went away: {exc}") from None
        except wire.WireError as exc:
            raise PolicyError(f"{op}: {exc}") from None
        if reply_op == "error":
            raise PolicyError(f"{op}: {reply_fields.get('message', 'refused')}")
        return reply_fields, reply_arrays
