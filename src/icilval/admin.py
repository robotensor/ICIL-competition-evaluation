"""Submission intake: a loopback HTTP server the organizer's dashboard form talks to.

POST /admin/submissions {repo, revision|null, track?, duel_size?, skip_model_config_check?, source?}
  -> 200 {ok, track, key, repo, revision, entry, position, accepted_at, message}
  -> 4xx {ok: false, error, detail?}
GET  /admin/health -> 200 {ok, validator_key, tracks: {<track>: {queue_len, in_progress, block}}}

`track` is optional while the competition has one field and required once it has more: queueing
against the wrong ladder is not something the organizer can see from the reply.
"""

from __future__ import annotations

import hmac
import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .ids import is_repo, is_sha_revision
from .queue import Queues
from .spec import Spec

RevisionResolver = Callable[[str, str | None], str]


def hf_revision_resolver(repo: str, revision: str | None) -> str:
    """Pin a repo (optionally at a branch/tag/short sha) to its full commit sha."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, revision=revision or None, timeout=20)
    if not info.sha:
        raise ValueError("Hugging Face returned no commit sha")
    return str(info.sha)


class AdminServer:
    def __init__(
        self,
        spec: Spec,
        queues: Queues,
        token: str,
        validator_key: str,
        *,
        bind: str | None = None,
        port: int | None = None,
        resolver: RevisionResolver = hf_revision_resolver,
        lock: threading.Lock | None = None,
    ):
        if not token:
            raise ValueError("admin token must be set")
        self.spec = spec
        self.queues = queues
        self.token = token
        self.validator_key = validator_key
        self.resolver = resolver
        self.lock = lock or threading.Lock()
        self.bind = bind or str(spec.admin["bind"])
        self.port = int(port if port is not None else spec.admin["port"])
        self.max_body = int(spec.admin["max_body_bytes"])
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:  # quiet
                return

            def _reply(self, status: int, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def _authorized(self) -> bool:
                header = self.headers.get("Authorization", "")
                presented = header[7:].strip() if header.lower().startswith("bearer ") else ""
                return bool(presented) and hmac.compare_digest(
                    presented.encode(), server.token.encode()
                )

            def do_GET(self) -> None:
                if not self._authorized():
                    return self._reply(
                        401, {"ok": False, "error": "Bearer token missing or wrong."}
                    )
                if self.path != "/admin/health":
                    return self._reply(404, {"ok": False, "error": "Not found."})
                with server.lock:
                    tracks = {}
                    for track, queue in server.queues.items():
                        st = queue.state
                        tracks[track] = {
                            "queue_len": len(st.entries),
                            "in_progress": st.in_progress.event_id if st.in_progress else None,
                            "block": st.block,
                        }
                    self._reply(
                        200,
                        {
                            "ok": True,
                            "validator_key": server.validator_key,
                            "tracks": tracks,
                        },
                    )

            def do_POST(self) -> None:
                if not self._authorized():
                    return self._reply(
                        401, {"ok": False, "error": "Bearer token missing or wrong."}
                    )
                if self.path != "/admin/submissions":
                    return self._reply(404, {"ok": False, "error": "Not found."})
                length = int(self.headers.get("Content-Length") or 0)
                if length > server.max_body:
                    return self._reply(413, {"ok": False, "error": "Body too large."})
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw.decode("utf-8") or "{}")
                except ValueError:
                    return self._reply(400, {"ok": False, "error": "Body is not valid JSON."})
                status, reply = server.submit(body)
                self._reply(status, reply)

        self.httpd = ThreadingHTTPServer((self.bind, self.port), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]

    # ---------------------------------------------------------------- logic
    def submit(self, body: Any) -> tuple[int, dict[str, Any]]:
        if not isinstance(body, dict):
            return 422, {"ok": False, "error": "Expected a JSON object."}
        repo = body.get("repo")
        if not isinstance(repo, str) or not is_repo(repo):
            return 422, {"ok": False, "error": "repo must look like owner/name."}
        revision = body.get("revision")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            return 422, {"ok": False, "error": "revision must be a string or null."}
        track = body.get("track")
        fields = self.spec.tracks
        if track is None:
            # One field needs no saying; more than one must be said, because queueing against
            # the wrong ladder is not something the organizer can see from the reply.
            if len(fields) != 1:
                return 422, {
                    "ok": False,
                    "error": f"track is required; the fields are: {', '.join(fields)}.",
                }
            track = fields[0]
        if track not in fields:
            return 422, {"ok": False, "error": f"track must be one of: {', '.join(fields)}."}
        duel_size = body.get("duel_size")
        if duel_size is not None and duel_size not in self.spec.sizes(track):
            return 422, {
                "ok": False,
                "error": f"duel_size must be one of: {', '.join(self.spec.sizes(track))}.",
            }
        skip = bool(body.get("skip_model_config_check", False))
        source = str(body.get("source") or "")[:64]
        try:
            pinned = self.resolver(repo, revision.strip() if isinstance(revision, str) else None)
        except Exception as exc:  # noqa: BLE001 - reported to the organizer
            return 422, {
                "ok": False,
                "error": "Could not resolve the repository revision.",
                "detail": str(exc)[:300],
            }
        if not is_sha_revision(pinned):
            return 422, {
                "ok": False,
                "error": "Resolved revision is not a commit sha.",
                "detail": pinned[:80],
            }
        with self.lock:
            entry, position = self.queues[track].add(
                repo, pinned, duel_size=duel_size, skip_model_config_check=skip, source=source
            )
        return 200, {
            "ok": True,
            "track": track,
            "key": entry.key,
            "repo": entry.repo,
            "revision": entry.revision,
            "entry": entry.ref.entry,
            "position": position,
            "accepted_at": entry.accepted_at,
            "message": "Queued. The validator picks it up on its next cycle and duels it against the reigning model.",
        }

    # ---------------------------------------------------------------- lifecycle
    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def start_background(self) -> threading.Thread:
        t = threading.Thread(target=self.httpd.serve_forever, name="icilval-admin", daemon=True)
        t.start()
        return t

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
