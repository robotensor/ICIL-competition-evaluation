"""Generate a drawing target and its demonstration for a unit, in process.

A port of BPP's `procedural_generate_drawings.py` (vendored) driven by one seeded NumPy generator:
a target is a list of *parts* (straight lines, Bezier curves, ovals and pen-up movements in BPP's
`bpp` family; the edges of a closed polygon in `polygon`; one polyline per skeleton stroke of a
font glyph in `glyph`), turned into 10 Hz pen actions at a sampled speed with BPP's noise, delays and hold,
rotated onto a board at a sampled angle and executed on `DrawEnv` while its frames, pen state and
actions are recorded. The result is one npz in the catalogue's demo format (`image`, `agent_pos`,
`pen_down`, `actions`, `boundary_angle`, `drawing`), with the family, seed and part count in `meta`.

Everything is a pure function of the seed; pygame runs headless (`SDL_VIDEODRIVER=dummy`).
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...canon import sha256_file
from ...spec import Spec
from .demos import stroke_mask

log = logging.getLogger(__name__)

CANVAS = 512
CENTER = CANVAS / 2.0
MARGIN_PX = 30.0
MIN_DISTANCE_PX = 50.0
CONNECTION_PROBABILITY = 0.3
OVAL_AXIS_PX = (30.0, 80.0)
SPEED_PX_S = (150.0, 400.0)
NOISE_STD_PX = 1.0
NOISE_BOUNDS_PX = 20.0
PART_DELAY_STEPS = (0, 10)
FINAL_HOLD_STEPS = 10
POSITIONING_STEPS = (10, 30)
OFFSET_FACTOR, OFFSET_MAX_PX = 0.3, 50.0
GLYPH_FILL = 0.55  # the glyph's longer side, as a fraction of the board
GLYPH_RENDER_PX = 256
GLYPH_STROKE_POINTS = 24  # points kept per skeleton stroke after simplification
GLYPH_MIN_STROKE = 0.04  # strokes shorter than this fraction of the glyph are dropped
DEFAULT_FONT = "DejaVuSans"


# ---------------------------------------------------------------- geometry
def _board_position(rng: np.random.Generator, board: float, margin: float) -> np.ndarray:
    half = board / 2
    return rng.uniform(CENTER - half + margin, CENTER + half - margin, size=2)


def _board_position_away(rng, start: np.ndarray, board: float, margin: float, min_d: float):
    for _ in range(100):
        pos = _board_position(rng, board, margin)
        if np.linalg.norm(pos - start) >= min_d:
            return pos
    return pos


def _bezier(p0, c1, c2, p3, t: float) -> np.ndarray:
    u = 1 - t
    return u**3 * p0 + 3 * u**2 * t * c1 + 3 * u * t**2 * c2 + t**3 * p3


def _movement_controls(rng, start, end, board, margin):
    d = np.linalg.norm(end - start)
    mag = min(d * OFFSET_FACTOR, OFFSET_MAX_PX)
    c1 = start + (end - start) / 3 + rng.uniform(-mag, mag, size=2)
    c2 = start + 2 * (end - start) / 3 + rng.uniform(-mag, mag, size=2)
    half = board / 2
    lo, hi = CENTER - half + margin, CENTER + half - margin
    return np.clip(c1, lo, hi), np.clip(c2, lo, hi)


def _inside(p: np.ndarray, board: float, margin: float) -> bool:
    half = board / 2
    return bool(np.all(p >= CENTER - half + margin) and np.all(p <= CENTER + half - margin))


def _oval_points(start, major, minor, angle, start_angle, angular_range, n: int):
    ts = start_angle + np.linspace(0, 1, n) * angular_range
    c, s = math.cos(angle), math.sin(angle)
    xs, ys = major * np.cos(ts), minor * np.sin(ts)
    xr, yr = xs * c - ys * s, xs * s + ys * c
    x0, y0 = major * math.cos(start_angle), minor * math.sin(start_angle)
    center = start - np.array([x0 * c - y0 * s, x0 * s + y0 * c])
    return np.stack([center[0] + xr, center[1] + yr], axis=1)


# ---------------------------------------------------------------- families -> parts
def bpp_parts(rng, cfg: dict[str, Any], board: float, margin: float) -> list[dict[str, Any]]:
    """BPP's procedural target: `min_parts..max_parts` parts, the first a stroke, no two
    movements in a row, a part sometimes connecting to an earlier endpoint."""
    n = int(rng.integers(int(cfg["min_parts"]), int(cfg["max_parts"]) + 1))
    kinds = ["straight", "curve", "oval", "movement"]
    pos = _board_position(rng, board, margin)
    parts: list[dict[str, Any]] = []
    endpoints: list[np.ndarray] = []
    for i in range(n):
        strokes = [k for k in kinds if k != "movement"]
        if i == 0 or (parts and parts[-1]["type"] == "movement"):
            kind = strokes[int(rng.integers(len(strokes)))]
        else:
            kind = kinds[int(rng.integers(len(kinds)))]
        if i > 0 and endpoints and rng.random() < CONNECTION_PROBABILITY:
            end = endpoints[int(rng.integers(len(endpoints)))]
        else:
            end = _board_position_away(rng, pos, board, margin, MIN_DISTANCE_PX)
        if kind == "oval":
            part = _oval_part(rng, pos, board, margin)
            if part is None:
                kind = "straight"
        if kind == "straight":
            part = {"type": "straight", "start": pos, "end": end, "pen": 1.0}
        elif kind == "curve":
            part = {
                "type": "curve",
                "start": pos,
                "end": end,
                "c1": _board_position(rng, board, margin),
                "c2": _board_position(rng, board, margin),
                "pen": 1.0,
            }
        elif kind == "movement":
            part = {"type": "movement", "start": pos, "end": end, "pen": 0.0}
        parts.append(part)
        pos = part["end"]
        endpoints.append(pos)
    return parts


def _oval_part(rng, start, board, margin) -> dict[str, Any] | None:
    for _ in range(100):
        major = rng.uniform(*OVAL_AXIS_PX)
        minor = major * rng.uniform(0.2, 3.0)
        angle = rng.uniform(0, 2 * math.pi)
        rng_dir = -2 * math.pi if rng.random() < 0.5 else 2 * math.pi
        pts = _oval_points(start, major, minor, angle, 0.0, rng_dir, 32)
        if all(_inside(p, board, margin) for p in pts):
            return {
                "type": "oval",
                "start": start,
                "end": start,
                "major": major,
                "minor": minor,
                "angle": angle,
                "start_angle": 0.0,
                "angular_range": rng_dir,
                "pen": 1.0,
            }
    return None


def polygon_parts(rng, cfg: dict[str, Any], board: float, margin: float) -> list[dict[str, Any]]:
    """One closed polygon: `vertices` corners around a random centre, drawn edge by edge."""
    lo, hi = (int(x) for x in cfg["vertices"])
    k = int(rng.integers(lo, hi + 1))
    half = board / 2
    radius = rng.uniform(0.15, 0.4) * board
    center = rng.uniform(CENTER - half + margin + radius, CENTER + half - margin - radius, size=2)
    angles = np.sort(rng.uniform(0, 2 * math.pi, size=k))
    radii = radius * rng.uniform(0.6, 1.0, size=k)
    corners = [
        center + r * np.array([math.cos(a), math.sin(a)])
        for a, r in zip(angles, radii, strict=True)
    ]
    corners.append(corners[0])
    return [
        {"type": "straight", "start": a, "end": b, "pen": 1.0}
        for a, b in zip(corners[:-1], corners[1:], strict=True)
    ]


def glyph_strokes(char: str, font: str, size_px: int = GLYPH_RENDER_PX) -> list[np.ndarray]:
    """The skeleton of a rendered character as ordered polylines, in a unit square."""
    from PIL import Image, ImageDraw, ImageFont
    from skimage.morphology import skeletonize

    font_path = _font_path(font)
    image = Image.new("L", (size_px, size_px), 0)
    fnt = ImageFont.truetype(font_path, int(size_px * 0.8))
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), char, font=fnt)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size_px - w) / 2 - bbox[0], (size_px - h) / 2 - bbox[1]), char, fill=255, font=fnt)
    mask = np.asarray(image) > 128
    skeleton = skeletonize(mask)
    strokes = _trace(skeleton)
    ys, xs = np.nonzero(skeleton)
    if not strokes or xs.size == 0:
        raise ValueError(f"glyph {char!r} has no skeleton")
    lo = np.array([xs.min(), ys.min()], dtype=np.float64)
    span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1.0)
    return [(np.asarray(s, dtype=np.float64)[:, ::-1] - lo) / span for s in strokes]


def _font_path(font: str) -> str:
    try:
        from matplotlib import font_manager

        return font_manager.findfont(font.replace("Sans", " Sans"), fallback_to_default=False)
    except Exception:  # noqa: BLE001 - fall back to a file name PIL can find
        return f"{font}.ttf"


def _trace(skeleton: np.ndarray) -> list[list[tuple[int, int]]]:
    """Walk a skeleton into strokes: from each endpoint (then any leftover pixel) along
    8-neighbours until a junction or a visited pixel."""
    pts = {(int(r), int(c)) for r, c in zip(*np.nonzero(skeleton), strict=True)}

    def nbrs(p):
        r, c = p
        return [
            q
            for q in ((r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
            if q in pts
        ]

    degree = {p: len(nbrs(p)) for p in pts}
    visited: set[tuple[int, int]] = set()
    strokes: list[list[tuple[int, int]]] = []
    order = sorted(pts, key=lambda p: (degree[p] != 1, p))
    for start in order:
        if start in visited:
            continue
        path = [start]
        visited.add(start)
        cur = start
        while True:
            nxt = [q for q in nbrs(cur) if q not in visited]
            if not nxt:
                break
            cur = nxt[0]
            path.append(cur)
            visited.add(cur)
            if degree[cur] > 2:
                break
        if len(path) >= 2:
            strokes.append(path)
    return strokes


def _length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1)))


def _along(points: np.ndarray, n: int) -> np.ndarray:
    """`n` points spaced evenly by arc length along a polyline."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return np.repeat(points[:1], n, axis=0)
    targets = np.linspace(0.0, cum[-1], n)
    xs = np.interp(targets, cum, points[:, 0])
    ys = np.interp(targets, cum, points[:, 1])
    return np.stack([xs, ys], axis=1)


def _simplify(points: np.ndarray, n: int) -> np.ndarray:
    if len(points) <= n:
        return points
    idx = np.linspace(0, len(points) - 1, n).round().astype(int)
    return points[idx]


def glyph_parts(rng, cfg: dict[str, Any], board: float, margin: float) -> list[dict[str, Any]]:
    """One character of the font: its skeleton strokes scaled onto the board, pen-up movements
    between them; drawn in a random stroke order and direction."""
    chars = str(cfg["characters"])
    char = chars[int(rng.integers(len(chars)))]
    strokes = glyph_strokes(char, str(cfg.get("font", DEFAULT_FONT)))
    scale = GLYPH_FILL * board * rng.uniform(0.7, 1.0)
    half = board / 2
    origin = rng.uniform(CENTER - half + margin, CENTER + half - margin - scale, size=2)
    strokes = [s for s in strokes if _length(s) >= GLYPH_MIN_STROKE]
    order = rng.permutation(len(strokes))
    parts: list[dict[str, Any]] = []
    pos: np.ndarray | None = None
    for i in order:
        pts = _simplify(strokes[i], GLYPH_STROKE_POINTS) * scale + origin
        if rng.random() < 0.5:
            pts = pts[::-1]
        if pos is not None and np.linalg.norm(pts[0] - pos) > 2.0:
            parts.append({"type": "movement", "start": pos, "end": pts[0], "pen": 0.0})
        parts.append(
            {"type": "polyline", "start": pts[0], "end": pts[-1], "points": pts, "pen": 1.0}
        )
        pos = pts[-1]
    if not parts:
        raise ValueError(f"glyph {char!r} produced no strokes")
    parts[0]["char"] = char
    return parts


FAMILIES = {"bpp": bpp_parts, "polygon": polygon_parts, "glyph": glyph_parts}


# ---------------------------------------------------------------- parts -> actions
def actions_from_parts(parts, rng, control_hz: int, speed: float, board: float, margin: float):
    """BPP's conversion: per-part step counts from the speed, Bezier / ellipse sampling, random
    delays between parts, a final hold, Gaussian noise except on each part's last action."""
    actions: list[np.ndarray] = []
    finals: list[int] = []
    for i, part in enumerate(parts):
        start, end, pen = part["start"], part["end"], part["pen"]
        if part["type"] == "straight":
            steps = max(int(np.linalg.norm(end - start) / speed * control_hz), 5)
            pts = [start + (end - start) * (k / (steps - 1)) for k in range(steps)]
        elif part["type"] == "polyline":
            steps = max(int(_length(part["points"]) / speed * control_hz), 8)
            pts = list(_along(np.asarray(part["points"], dtype=np.float64), steps))
        elif part["type"] in ("curve", "movement"):
            if part["type"] == "movement":
                c1, c2 = _movement_controls(rng, start, end, board, margin)
            else:
                c1, c2 = part["c1"], part["c2"]
            steps = max(
                int(np.linalg.norm(end - start) * 1.5 / speed * control_hz),
                5 if part["type"] == "movement" else 8,
            )
            pts = [_bezier(start, c1, c2, end, k / (steps - 1)) for k in range(steps)]
        else:  # oval
            a, b = part["major"], part["minor"]
            h = ((a - b) / (a + b)) ** 2
            perimeter = math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
            steps = max(int(perimeter / speed * control_hz), 16)
            pts = list(
                _oval_points(
                    start, a, b, part["angle"], part["start_angle"], part["angular_range"], steps
                )
            )
        actions.extend(np.array([p[0], p[1], pen], dtype=np.float32) for p in pts)
        finals.append(len(actions) - 1)
        if i < len(parts) - 1:
            for _ in range(int(rng.integers(PART_DELAY_STEPS[0], PART_DELAY_STEPS[1] + 1))):
                actions.append(np.array([end[0], end[1], pen], dtype=np.float32))
    last = parts[-1]
    for _ in range(FINAL_HOLD_STEPS):
        actions.append(np.array([last["end"][0], last["end"][1], last["pen"]], dtype=np.float32))
    out = np.array(actions, dtype=np.float32)
    noise_mask = np.ones(len(out), dtype=bool)
    noise_mask[finals] = False
    noise = np.clip(
        rng.normal(0.0, NOISE_STD_PX, size=(int(noise_mask.sum()), 2)),
        -NOISE_BOUNDS_PX,
        NOISE_BOUNDS_PX,
    )
    out[noise_mask, :2] += noise.astype(np.float32)
    return out


def rotate_actions(actions: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    out = actions.copy()
    x, y = actions[:, 0] - CENTER, actions[:, 1] - CENTER
    out[:, 0] = x * c - y * s + CENTER
    out[:, 1] = x * s + y * c + CENTER
    return out


# ---------------------------------------------------------------- recording
@dataclass
class DrawingResult:
    success: bool
    family: str
    seed: int
    npz: Path | None = None
    steps: int = 0
    sha256: str | None = None
    boundary_angle: float = 0.0
    parts: int = 0
    error: str | None = None


def record_demo(actions: np.ndarray, angle: float, cursor: np.ndarray, spec: Spec, skill: str, rng):
    """Execute the (rotated) actions on a board at `angle`: first BPP's pen-up positioning from
    the cursor start to the first action, then the trajectory; frames, pen state and actions
    recorded per step."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from behavior_prompting.train_network.env.draw.draw_env import DrawEnv

    env_cfg = spec.env(skill)
    env = DrawEnv(
        boundary_angle=float(angle),
        render_size=int(env_cfg["policy_image_resolution"]),
        overlay_reward=False,
        overlay_action_cross=False,
        overlay_target_drawing=False,
        render_mode="rgb_array",
    )
    env.seed(int(rng.integers(1 << 31)))
    env.reset_to_state = np.array([float(cursor[0]), float(cursor[1]), 0.0])
    obs = env.reset(no_rotation=False)
    images, agent_pos, pen_down, acts = [], [], [], []

    def step(act: np.ndarray):
        nonlocal obs
        images.append((np.transpose(obs["image"], (1, 2, 0)) * 255).astype(np.uint8))
        agent_pos.append(np.asarray(obs["agent_pos"], dtype=np.float32))
        pen_down.append(np.asarray(obs["pen_down"], dtype=np.float32).reshape(1))
        acts.append(np.asarray(act, dtype=np.float32))
        obs, _, _, _ = env.step(act)

    start = actions[0, :2]
    here = np.asarray(obs["agent_pos"], dtype=np.float32)
    c1, c2 = _movement_controls(rng, here, start, float(env_cfg["board_length"]), MARGIN_PX)
    n_pos = int(rng.integers(POSITIONING_STEPS[0], POSITIONING_STEPS[1] + 1))
    for k in range(n_pos):
        if np.linalg.norm(np.asarray(obs["agent_pos"]) - start) < 5.0:
            break
        target = _bezier(here, c1, c2, start, k / (n_pos - 1))
        step(np.array([target[0], target[1], 0.0], dtype=np.float32))
    for _ in range(100):
        if np.linalg.norm(np.asarray(obs["agent_pos"]) - start) < 5.0:
            break
        step(np.array([start[0], start[1], 0.0], dtype=np.float32))
    else:
        env.close()
        raise RuntimeError("the pen did not reach the first action")
    for act in actions:
        step(act)
    drawing = stroke_mask(env.get_drawing_image())
    env.close()
    return {
        "image": np.stack(images),
        "agent_pos": np.stack(agent_pos),
        "pen_down": np.stack(pen_down),
        "actions": np.stack(acts),
        "boundary_angle": np.asarray(float(angle), dtype=np.float64),
        "drawing": drawing,
    }


def generate_drawing(
    spec: Spec, skill: str, family: str, seed: int, out_npz: Path, demo_id: str
) -> DrawingResult:
    """A target of `family` and one demonstration of it at a random board angle, from `seed`."""
    env_cfg = spec.env(skill)
    families = spec.skill_generation(skill).get("families") or {}
    if family not in FAMILIES or family not in families:
        raise ValueError(f"unknown drawing family {family!r}")
    rng = np.random.default_rng(int(seed))
    board = float(env_cfg["board_length"])
    lo, hi = (float(x) for x in env_cfg["board_angle_range_rad"])
    c_lo, c_hi = (int(x) for x in env_cfg["cursor_start_range_px"])
    try:
        parts = FAMILIES[family](
            rng,
            {k: v for k, v in families[family].items() if not k.startswith("_")},
            board,
            MARGIN_PX,
        )
        speed = rng.uniform(*SPEED_PX_S)
        upright = actions_from_parts(
            parts, rng, int(env_cfg["control_freq"]), speed, board, MARGIN_PX
        )
        angle = float(rng.uniform(lo, hi))
        cursor = rng.integers(c_lo, c_hi + 1, size=2)
        arrays = record_demo(rotate_actions(upright, angle), angle, cursor, spec, skill, rng)
    except Exception as exc:  # noqa: BLE001 - reported, the caller retries with another seed
        log.warning("drawing %s seed %d failed: %s", family, seed, exc)
        return DrawingResult(False, family, int(seed), error=f"{type(exc).__name__}: {exc}"[:300])
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "demo_id": demo_id,
        "source": "generated",
        "family": family,
        "seed": int(seed),
        "parts": len(parts),
        "steps": int(arrays["actions"].shape[0]),
        "boundary_angle": angle,
        "stroke_pixels": int(arrays["drawing"].sum()),
        "character": parts[0].get("char") if family == "glyph" else None,
    }
    np.savez_compressed(out_npz, meta=json.dumps(meta), **arrays)
    return DrawingResult(
        True, family, int(seed), out_npz, meta["steps"], sha256_file(out_npz), angle, len(parts)
    )
