import os
import uuid
from dataclasses import dataclass
from typing import List

from ..config import Config
from ..utils.video_utils import probe_video, split_video_into_chunks


@dataclass(frozen=True)
class PreparedVideoChunk:
    local_path: str
    gcs_uri: str
    start_seconds: float
    end_seconds: float


def _upload_to_gcs(local_path: str, *, bucket_name: str, prefix: str) -> str:
    """Upload a local file to GCS and return gs:// URI."""
    try:
        from google.cloud import storage
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "google-cloud-storage is required for uploading to GCS. "
            "Install it with: pip install google-cloud-storage"
        ) from e

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blob_name = f"{prefix}/{os.path.basename(local_path)}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)

    return f"gs://{bucket_name}/{blob_name}"


def prepare_video_for_vertex(video_path: str) -> List[PreparedVideoChunk]:
    """Prepare a local video for Vertex video understanding.

    - Probes duration/audio
    - Splits into <=45m chunks if needed
    - Uploads each chunk to GCS
    """
    if not Config.GCS_BUCKET:
        raise ValueError(
            "VIDEOANALISI_GCS_BUCKET is required for Vertex video understanding with local videos."
        )

    meta = probe_video(video_path)
    max_seconds = (
        Config.VIDEO_MAX_CHUNK_SECONDS_WITH_AUDIO
        if meta.has_audio
        else Config.VIDEO_MAX_CHUNK_SECONDS_NO_AUDIO
    )

    chunk_dir = os.path.join(Config.TEMP_DIR, "video_chunks", str(uuid.uuid4()))
    os.makedirs(chunk_dir, exist_ok=True)

    chunks = split_video_into_chunks(
        video_path,
        chunk_dir,
        max_chunk_seconds=max_seconds,
        overlap_seconds=Config.VIDEO_CHUNK_OVERLAP_SECONDS,
    )

    upload_prefix = f"videoanalisi/{os.path.splitext(os.path.basename(video_path))[0]}/{uuid.uuid4()}"

    prepared: List[PreparedVideoChunk] = []
    for chunk in chunks:
        local_chunk_path = chunk["path"]
        gcs_uri = _upload_to_gcs(
            local_chunk_path, bucket_name=Config.GCS_BUCKET, prefix=upload_prefix
        )
        prepared.append(
            PreparedVideoChunk(
                local_path=local_chunk_path,
                gcs_uri=gcs_uri,
                start_seconds=float(chunk["start"]),
                end_seconds=float(chunk["end"]),
            )
        )

    return prepared
