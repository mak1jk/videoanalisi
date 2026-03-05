import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Vertex AI / Google Cloud
    USE_VERTEX_AI = os.getenv("USE_VERTEX_AI", "false").lower() in {"1", "true", "yes"}
    GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
    GCP_LOCATION = (
        os.getenv("GOOGLE_CLOUD_LOCATION")
        or os.getenv("GOOGLE_CLOUD_REGION")
        or "us-central1"
    )
    GCS_BUCKET = os.getenv("VIDEOANALISI_GCS_BUCKET") or os.getenv("GCS_BUCKET")

    # Model selection
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

    # Video understanding / chunking
    VIDEO_NATIVE_ENABLED = os.getenv("VIDEO_NATIVE_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    VIDEO_MAX_CHUNK_SECONDS_WITH_AUDIO = int(
        os.getenv("VIDEO_MAX_CHUNK_SECONDS_WITH_AUDIO", "2700")
    )  # 45m
    VIDEO_MAX_CHUNK_SECONDS_NO_AUDIO = int(
        os.getenv("VIDEO_MAX_CHUNK_SECONDS_NO_AUDIO", "3600")
    )  # 60m
    VIDEO_CHUNK_OVERLAP_SECONDS = int(os.getenv("VIDEO_CHUNK_OVERLAP_SECONDS", "15"))
    VIDEO_TIMELINE_MAX_EVENTS = int(os.getenv("VIDEO_TIMELINE_MAX_EVENTS", "60"))
    VIDEO_MEDIA_RESOLUTION = os.getenv(
        "VIDEO_MEDIA_RESOLUTION", "LOW"
    )  # LOW|MEDIUM|HIGH|ULTRA_HIGH

    # Processing Settings
    FRAME_SAMPLE_RATE = 1  # frames per second (base)
    CLIP_SIMILARITY_THRESHOLD = 0.90
    MIN_SEGMENT_DURATION = 30.0  # seconds (for VLM context)

    # Paths
    OUTPUT_DIR = "output_documents"
    TEMP_DIR = "temp_processing"

    @staticmethod
    def validate():
        if Config.USE_VERTEX_AI:
            if not Config.GCP_PROJECT:
                print(
                    "Warning: USE_VERTEX_AI enabled but GOOGLE_CLOUD_PROJECT is not set."
                )
            if not Config.GCP_LOCATION:
                print(
                    "Warning: USE_VERTEX_AI enabled but GOOGLE_CLOUD_LOCATION is not set."
                )
            if not Config.GCS_BUCKET:
                print(
                    "Warning: VIDEOANALISI_GCS_BUCKET is not set. "
                    "Local videos will need a bucket to be analyzed via Vertex AI video understanding."
                )
        else:
            if (
                not Config.GOOGLE_API_KEY
                and not Config.OPENAI_API_KEY
                and not Config.ANTHROPIC_API_KEY
            ):
                print("Warning: No VLM API keys found in .env file.")
