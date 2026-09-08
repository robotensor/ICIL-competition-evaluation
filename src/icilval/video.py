"""Frames to mp4 through ffmpeg (bundled with imageio-ffmpeg). Settings come from spec.media."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .canon import sha256_file


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


class VideoWriter:
    """Streams raw RGB frames into an H.264 mp4. All frames must share one shape."""

    def __init__(self, path: str | Path, fps: int, video_cfg: dict[str, Any]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.cfg = video_cfg
        self._proc: subprocess.Popen | None = None
        self._shape: tuple[int, int] | None = None
        self.frames = 0

    def _open(self, h: int, w: int) -> None:
        cmd = [
            ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            str(self.cfg.get("codec", "libx264")),
            "-pix_fmt",
            str(self.cfg.get("pix_fmt", "yuv420p")),
            "-crf",
            str(self.cfg.get("crf", 20)),
            "-preset",
            str(self.cfg.get("preset", "veryfast")),
        ]
        if self.cfg.get("faststart", True):
            cmd += ["-movflags", "+faststart"]
        cmd.append(str(self.path))
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._shape = (h, w)

    def write(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be HxWx3 uint8")
        h, w = frame.shape[:2]
        if h % 2 or w % 2:  # yuv420p needs even dimensions
            frame = frame[: h - h % 2, : w - w % 2]
            h, w = frame.shape[:2]
        if self._proc is None:
            self._open(h, w)
        elif (h, w) != self._shape:
            raise ValueError("frame shape changed mid-video")
        assert self._proc.stdin is not None
        self._proc.stdin.write(frame.tobytes())
        self.frames += 1

    def close(self) -> Path:
        if self._proc is None:
            # zero frames: write nothing, leave no file
            return self.path
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        err = self._proc.stderr.read() if self._proc.stderr else b""
        code = self._proc.wait()
        self._proc = None
        if code != 0:
            raise RuntimeError(f"ffmpeg failed ({code}): {err.decode(errors='replace')[:400]}")
        return self.path

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc: Any) -> None:
        if exc[0] is None:
            self.close()
        elif self._proc is not None:
            self._proc.kill()


def encode_frames(
    frames: np.ndarray | list[np.ndarray], path: str | Path, fps: int, video_cfg: dict[str, Any]
) -> str:
    with VideoWriter(path, fps, video_cfg) as w:
        for f in frames:
            w.write(f)
    return sha256_file(path)


def side_by_side(*frames: np.ndarray, gap: int = 4) -> np.ndarray:
    h = max(f.shape[0] for f in frames)
    parts = []
    for i, f in enumerate(frames):
        if f.shape[0] != h:
            pad = np.zeros((h - f.shape[0], f.shape[1], 3), np.uint8)
            f = np.concatenate([f, pad], axis=0)
        parts.append(f)
        if i < len(frames) - 1:
            parts.append(np.zeros((h, gap, 3), np.uint8))
    return np.concatenate(parts, axis=1)


def upscale(frame: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(frame, factor, axis=0), factor, axis=1)


def is_faststart(path: str | Path) -> bool:
    """`moov` before `mdat` means the player can start before the download finishes."""
    data = Path(path).read_bytes()[: 1 << 20]
    return 0 <= data.find(b"moov") < max(data.find(b"mdat"), len(data))
