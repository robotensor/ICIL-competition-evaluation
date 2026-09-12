"""Serving an entrant's policy to the benchmark that will score it.

The orchestrator holds the weights and the architecture template; the benchmark holds the
simulator. Neither can import the other's stack, so they meet on a socket: this module is the
server, and `policy_address` in the benchmark ABI is the address it is listening on.

Everything here is about what a *failure* does, because the failure modes are what make a duel
wrong rather than merely broken:

- **The policy is never reloaded between units.** `load()` once per skill is the frozen-policy
  guarantee; a host that reconnected and reloaded would let inference-time state be laundered
  between units.
- **A model error is one unit's, not the duel's.** An exception from `act` is returned to the
  benchmark as an error reply and counted; the benchmark decides whether to void the unit. The
  server stays up, because the remaining units are still owed a fair run.
- **No server outlives its client.** The listener is closed and the thread joined when the host
  exits its context, whatever happened inside it.

The transport is `icilval.model.wire`: named arrays only, no pickling. A benchmark implements
the client; it never imports this repository.
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import socket
import tempfile
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

from . import wire

log = logging.getLogger(__name__)

#: The environment variable a benchmark subprocess reads its key from. Never on a command line,
#: where it would be visible in `ps` to anything on the host.
AUTHKEY_ENV = "ICILVAL_POLICY_AUTHKEY"

#: How long a client may hold the connection open with nothing to say before the host assumes it
#: has gone. Generous: a benchmark building a scene between units is not idle, it is working.
IDLE_TIMEOUT_S = 1800.0


class PolicyHost:
    """One policy, served for as long as this object is open.

    Use as a context manager. `address` is what goes to `run_command`; `authkey_file` is where the
    key was written, for a benchmark that wants it from a file rather than the environment.
    """

    def __init__(self, policy: Any, *, directory: str | Path | None = None) -> None:
        self.policy = policy
        self._own_dir = directory is None
        self._dir = (
            Path(directory) if directory else Path(tempfile.mkdtemp(prefix="icilval-policy-"))
        )
        self._dir.mkdir(parents=True, exist_ok=True)
        # 0700: the socket and the key are readable only by the validator's own user.
        os.chmod(self._dir, 0o700)
        self.address = str(self._dir / "policy.sock")
        self.authkey = secrets.token_bytes(32)
        self.authkey_file = self._dir / "authkey"
        self.authkey_file.write_bytes(self.authkey)
        os.chmod(self.authkey_file, 0o600)
        #: Exceptions raised by the policy, which the side runner reports as `model_errors`.
        self.errors = 0
        self._listener: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- lifecycle ------------------------------------------------------------------

    def __enter__(self) -> PolicyHost:
        from multiprocessing.connection import Listener

        self._listener = Listener(self.address, family="AF_UNIX", authkey=self.authkey)
        os.chmod(self.address, 0o600)
        self._thread = threading.Thread(target=self._serve, name="icilval-policy-host", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Stop serving. Deterministic rather than timed out.

        Closing the listener does not interrupt a thread already blocked in `accept()`, so the
        stop flag is set and then a throwaway connection is made to our own address to wake it -
        we hold the key, so it authenticates. Waiting for `accept()` to time out instead cost ten
        seconds on every host, which is ten seconds on every unit of every duel.
        """
        self._stop.set()
        if self._listener is not None:
            with suppress(Exception):
                from multiprocessing.connection import Client

                Client(self.address, family="AF_UNIX", authkey=self.authkey).close()
            with suppress(OSError):
                self._listener.close()
            self._listener = None
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        if self._own_dir:
            shutil.rmtree(self._dir, ignore_errors=True)

    def env(self) -> dict[str, str]:
        """What a benchmark subprocess needs in its environment to connect."""
        return {AUTHKEY_ENV: self.authkey.hex()}

    # -- serving --------------------------------------------------------------------

    def _serve(self) -> None:
        while not self._stop.is_set() and self._listener is not None:
            try:
                conn = self._listener.accept()
            except (OSError, EOFError):
                return  # the listener was closed, which is how `close()` stops us
            except Exception:  # noqa: BLE001 - an authentication failure must not kill the host
                log.warning("policy host: rejected a connection", exc_info=True)
                continue
            if self._stop.is_set():
                # The wake-up from `close()`. Nothing to serve.
                with suppress(OSError):
                    conn.close()
                return
            try:
                self._session(conn)
            finally:
                with suppress(OSError):
                    conn.close()

    def _session(self, conn: Any) -> None:
        """One benchmark subprocess, for as long as it holds the connection."""
        while not self._stop.is_set():
            if not conn.poll(IDLE_TIMEOUT_S):
                return
            try:
                op, fields, arrays = wire.recv(conn)
            except (EOFError, ConnectionError):
                return
            except wire.WireError as exc:
                self._reply_error(conn, str(exc))
                continue
            if op == "close":
                with suppress(OSError):
                    wire.send(conn, "ok")
                return
            self._dispatch(conn, op, fields, arrays)

    def _dispatch(self, conn: Any, op: str, fields: dict[str, Any], arrays: dict[str, Any]) -> None:
        try:
            handler = getattr(self, f"_op_{op}", None)
            if handler is None:
                self._reply_error(conn, f"unknown op {op!r}")
                return
            handler(conn, fields, arrays)
        except wire.WireError as exc:
            self._reply_error(conn, str(exc))
        except Exception as exc:  # noqa: BLE001 - one unit's failure, not the duel's
            # Counted and reported. The host stays up: the units after this one are still owed a
            # fair run, and tearing the server down would void all of them for one model error.
            self.errors += 1
            log.warning("policy host: %s raised", op, exc_info=True)
            self._reply_error(conn, f"{type(exc).__name__}: {exc}")

    # -- operations -----------------------------------------------------------------

    def _op_hello(self, conn: Any, fields: dict[str, Any], arrays: dict[str, Any]) -> None:
        wire.send(conn, "ok", {"protocol": wire.PROTOCOL_VERSION})

    def _op_seed(self, conn: Any, fields: dict[str, Any], arrays: dict[str, Any]) -> None:
        seed = fields.get("seed")
        if not isinstance(seed, int):
            raise wire.WireError("seed: expected an integer")
        self.policy.seed(seed)
        wire.send(conn, "ok")

    def _op_reset(self, conn: Any, fields: dict[str, Any], arrays: dict[str, Any]) -> None:
        self.policy.reset()
        wire.send(conn, "ok")

    def _op_prompt(self, conn: Any, fields: dict[str, Any], arrays: dict[str, Any]) -> None:
        """The demonstration, exactly as the field's view allows it to be seen.

        The arrays arrive as the benchmark read them from the materialized prompt; what the view
        withholds was removed before this point, by `demoview.apply`, and was never sent.
        """
        info = self.policy.set_prompt(dict(arrays))
        out = {
            "prompt_steps": getattr(info, "steps", None),
            "prompt_chunks": getattr(info, "chunks", None),
        }
        wire.send(conn, "ok", {k: v for k, v in out.items() if v is not None})

    def _op_act(self, conn: Any, fields: dict[str, Any], arrays: dict[str, Any]) -> None:
        history = wire.unpack_history(fields, arrays)
        action = self.policy.act(history)
        wire.send(conn, "action", arrays={"action": action})

    def _reply_error(self, conn: Any, message: str) -> None:
        with suppress(OSError, ConnectionError):
            wire.send(conn, "error", {"message": message})


def free_tcp_address() -> str:
    """A loopback address nothing is listening on, for a benchmark that cannot use a Unix socket."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{s.getsockname()[1]}"
