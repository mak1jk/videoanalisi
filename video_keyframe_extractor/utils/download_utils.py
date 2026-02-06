import os
import subprocess
import uuid
from typing import Optional


def download_video_url(
    url: str, output_dir: str, *, filename_stem: Optional[str] = None
) -> str:
    """Download a video URL (YouTube or direct) locally using yt-dlp.

    Returns the downloaded file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    stem = filename_stem or str(uuid.uuid4())
    outtmpl = os.path.join(output_dir, f"{stem}.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f",
        "bv*+ba/best",
        "--merge-output-format",
        "mp4",
        "-o",
        outtmpl,
        url,
    ]

    proc = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"yt-dlp download failed ({proc.returncode}).\n{proc.stderr.strip()}"
        )

    # yt-dlp will pick the extension; after merge-output-format it's typically mp4
    expected_mp4 = os.path.join(output_dir, f"{stem}.mp4")
    if os.path.exists(expected_mp4):
        return expected_mp4

    # Fallback: find any file with matching stem
    for name in os.listdir(output_dir):
        if name.startswith(stem + "."):
            return os.path.join(output_dir, name)

    raise FileNotFoundError("yt-dlp completed but output file was not found.")
