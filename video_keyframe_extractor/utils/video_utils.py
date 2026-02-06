import json
import math
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class VideoMetadata:
    duration_seconds: float
    has_audio: bool


def _run(cmd: List[str]) -> str:
    proc = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}"
        )
    return proc.stdout


def probe_video(video_path: str) -> VideoMetadata:
    """Return duration (seconds) and whether audio stream exists."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    out = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
    )
    data: Dict[str, Any] = json.loads(out)

    duration_str = (data.get("format") or {}).get("duration")
    if not duration_str:
        raise ValueError(f"Unable to determine video duration for: {video_path}")

    duration_seconds = float(duration_str)
    has_audio = any(
        (s or {}).get("codec_type") == "audio" for s in data.get("streams", [])
    )

    return VideoMetadata(duration_seconds=duration_seconds, has_audio=has_audio)


def split_video_into_chunks(
    video_path: str,
    output_dir: str,
    *,
    max_chunk_seconds: int,
    overlap_seconds: int,
) -> List[Dict[str, Any]]:
    """Split a local video into chunks using ffmpeg (stream copy).

    Returns list of dicts: {"path": ..., "start": ..., "end": ...}.
    """
    meta = probe_video(video_path)

    os.makedirs(output_dir, exist_ok=True)

    if meta.duration_seconds <= max_chunk_seconds:
        return [
            {
                "path": video_path,
                "start": 0.0,
                "end": meta.duration_seconds,
            }
        ]

    chunk_step = max(1, max_chunk_seconds - overlap_seconds)
    total_chunks = int(math.ceil(meta.duration_seconds / chunk_step))

    chunks: List[Dict[str, Any]] = []
    for i in range(total_chunks):
        start = float(i * chunk_step)
        if start >= meta.duration_seconds:
            break
        end = min(meta.duration_seconds, start + max_chunk_seconds)
        duration = max(0.0, end - start)

        out_path = os.path.join(
            output_dir, f"chunk_{i:03d}_{int(start)}_{int(end)}.mp4"
        )

        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(start),
                "-i",
                video_path,
                "-t",
                str(duration),
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                out_path,
            ]
        )

        chunks.append({"path": out_path, "start": start, "end": end})

    if not chunks:
        raise RuntimeError("Chunking produced no output chunks.")

    return chunks
