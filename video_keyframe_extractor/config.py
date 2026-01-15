import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    # Processing Settings
    FRAME_SAMPLE_RATE = 1  # frames per second (base)
    CLIP_SIMILARITY_THRESHOLD = 0.90
    MIN_SEGMENT_DURATION = 30.0 # seconds (for VLM context)

    # Paths
    OUTPUT_DIR = "output_documents"
    TEMP_DIR = "temp_processing"

    @staticmethod
    def validate():
        if not Config.GOOGLE_API_KEY and not Config.OPENAI_API_KEY and not Config.ANTHROPIC_API_KEY:
            print("Warning: No VLM API keys found in .env file.")
