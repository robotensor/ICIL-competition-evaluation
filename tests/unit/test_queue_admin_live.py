import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from icilval.admin import AdminServer
from icilval.ids import ModelRef
from icilval.live import LiveReporter, build_frame
from icilval.queue import Queue


def test_queue_replace_moves_to_back_and_persists(tmp_path):
    q = Queue(tmp_path / "q.json")
    e1, p1 = q.add("a/x", "1" * 40)
    e2, p2 = q.add("b/y", "2" * 40, duel_size="light")
    assert (p1, p2) == (1, 2)
    e1b, p1b = q.add("a/x", "1" * 40)
    assert p1b == 2 and e1b.key == e1.key and [e.key for e in q.entries()] == [e2.key, e1.key]
    q2 = Queue(tmp_path / "q.json")
    assert [e.key for e in q2.entries()] == [e2.key, e1.key]
    assert q2.pop().key == e2.key
    q2.start("e" * 64, ModelRef.make("b/y", "2" * 40))
    snap = q2.snapshot("icil_1demo", None, 1)
    assert snap["in_progress"]["event_id"] == "e" * 64 and snap["entries"][0]["position"] == 1
    q2.finish()
    assert Queue(tmp_path / "q.json").state.in_progress is None
    assert q2.remove(e1.key) and not q2.remove(e1.key)


def _call(url, token, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_admin_server_contract(spec, tmp_path):
    def resolver(repo, revision):
        if repo == "bad/repo":
            raise RuntimeError("no such repo")
        return (revision or "c") * 40 if len(revision or "c") == 1 else "d" * 40

    q = Queue(tmp_path / "q.json")
    server = AdminServer(spec, q, "secret", "k" * 64, bind="127.0.0.1", port=0, resolver=resolver)
    server.start_background()
    base = f"http://127.0.0.1:{server.port}"
    try:
        assert _call(base + "/admin/health", "wrong")[0] == 401
        status, body = _call(base + "/admin/health", "secret")
        assert (
            status == 200
            and body["ok"]
            and body["queue_len"] == 0
            and body["validator_key"] == "k" * 64
        )
        assert _call(base + "/nope", "secret")[0] == 404
        status, body = _call(
            base + "/admin/submissions",
            "secret",
            "POST",
            {
                "repo": "org/model",
                "revision": None,
                "track": spec.sole_track,
                "duel_size": "smoke",
                "skip_model_config_check": False,
                "source": "dashboard-dev-mode",
            },
        )
        assert (
            status == 200
            and body["ok"]
            and body["revision"] == "c" * 40
            and body["position"] == 1
            and body["entry"] == "org/model@" + "c" * 40
        )
        assert set(body) >= {
            "key",
            "repo",
            "revision",
            "entry",
            "position",
            "accepted_at",
            "message",
        }
        assert (
            _call(base + "/admin/submissions", "secret", "POST", {"repo": "not a repo"})[0] == 422
        )
        assert (
            _call(
                base + "/admin/submissions",
                "secret",
                "POST",
                {"repo": "org/model", "duel_size": "huge"},
            )[0]
            == 422
        )
        assert (
            _call(
                base + "/admin/submissions",
                "secret",
                "POST",
                {"repo": "org/model", "track": "other"},
            )[0]
            == 422
        )
        status, body = _call(base + "/admin/submissions", "secret", "POST", {"repo": "bad/repo"})
        assert status == 422 and "detail" in body
        req = urllib.request.Request(
            base + "/admin/submissions",
            data=b"{",
            method="POST",
            headers={"Authorization": "Bearer secret"},
        )
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(req, timeout=5)
        assert ei.value.code == 400
        big = urllib.request.Request(
            base + "/admin/submissions",
            data=b"x" * 20000,
            method="POST",
            headers={"Authorization": "Bearer secret"},
        )
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(big, timeout=5)
        assert ei.value.code == 413
    finally:
        server.shutdown()


class _Sink(BaseHTTPRequestHandler):
    frames = []

    def log_message(self, *a):
        return

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        _Sink.frames.append(
            (self.path, self.headers.get("Authorization"), json.loads(self.rfile.read(n)))
        )
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")


def test_live_frame_and_reporter(spec):
    units = [
        {
            "unit_id": "pp-000",
            "skill": "pick_and_place",
            "task": "t",
            "instance": 0,
            "king_success": True,
            "challenger_success": None,
            "outcome": "tie",
            "demo_video": "a" * 64,
        },
        {
            "unit_id": "da-000",
            "skill": "draw_anything",
            "task": "t2",
            "instance": 1,
            "king_success": None,
            "challenger_success": None,
            "outcome": "tie",
        },
    ]
    frame = build_frame(
        spec,
        validator_key="k",
        event_id="e" * 64,
        kind="duel",
        duel_size="smoke",
        king=None,
        challenger=None,
        phase="evaluating",
        side="king",
        units=units,
        current={"unit_id": "da-000", "skill": "draw_anything"},
        recent_media=None,
        message="m" * 400,
        started_at="2026-01-01T00:00:00Z",
    )
    assert (
        frame["done"] == 1
        and frame["total"] == 2
        and frame["skill_progress"]["king"]["pick_and_place"] == {"done": 1, "total": 1}
    )
    assert frame["skill_progress"]["challenger"]["pick_and_place"]["done"] == 0
    assert (
        frame["units"][0]["skill"] == "pick_and_place"
        and frame["units"][1]["skill"] == "draw_anything"
    )
    assert (
        len(frame["message"]) == 300
        and frame["units"][0]["outcome"] is None
        and frame["schema"] == spec.live["schema"]
    )
    with pytest.raises(ValueError):
        build_frame(
            spec,
            validator_key="k",
            event_id="e",
            kind="duel",
            duel_size=None,
            king=None,
            challenger=None,
            phase="bogus",
            side=None,
            units=[],
            current=None,
            recent_media=None,
            message="",
            started_at="",
        )

    httpd = HTTPServer(("127.0.0.1", 0), _Sink)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        rep = LiveReporter(spec, f"http://127.0.0.1:{httpd.server_address[1]}", "tok")
        assert rep.enabled and rep.post(frame, force=True)
        assert not rep.post(frame)  # rate limited
        assert rep.post(frame, force=True)
        path, auth, got = _Sink.frames[0]
        assert path == spec.live["path"] and auth == "Bearer tok" and got["event_id"] == "e" * 64
        off = LiveReporter(spec, None, None)
        assert not off.enabled and not off.post(frame, force=True)
        dead = LiveReporter(spec, "http://127.0.0.1:9", "tok", timeout_s=0.5)
        assert not dead.post(frame, force=True) and dead.failed == 1
    finally:
        httpd.shutdown()
