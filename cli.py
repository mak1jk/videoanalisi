import argparse
import os

from video_keyframe_extractor.config import Config
from video_keyframe_extractor.core.orchestrator import FrameSelectorOrchestrator
from video_keyframe_extractor.output.html_generator import HTMLGenerator
from video_keyframe_extractor.utils.download_utils import download_video_url
from video_keyframe_extractor.vlm_providers.gemini_provider import GeminiProvider


def _apply_runtime_config(args: argparse.Namespace) -> None:
    if args.use_vertex_ai:
        Config.USE_VERTEX_AI = True
        os.environ["USE_VERTEX_AI"] = "true"

    if args.gcp_project:
        Config.GCP_PROJECT = args.gcp_project
        os.environ["GOOGLE_CLOUD_PROJECT"] = args.gcp_project

    if args.gcp_location:
        Config.GCP_LOCATION = args.gcp_location
        os.environ["GOOGLE_CLOUD_LOCATION"] = args.gcp_location

    if args.gcs_bucket:
        Config.GCS_BUCKET = args.gcs_bucket
        os.environ["VIDEOANALISI_GCS_BUCKET"] = args.gcs_bucket

    if args.gemini_model:
        Config.GEMINI_MODEL = args.gemini_model
        os.environ["GEMINI_MODEL"] = args.gemini_model

    if args.video_native is not None:
        Config.VIDEO_NATIVE_ENABLED = args.video_native
        os.environ["VIDEO_NATIVE_ENABLED"] = "true" if args.video_native else "false"


def main() -> None:
    parser = argparse.ArgumentParser(description="Video Keyframe Extractor CLI")
    parser.add_argument("video_path", nargs="?", help="Path to the input video file")
    parser.add_argument("--video-url", help="Video URL (YouTube or direct)")

    parser.add_argument(
        "--use-vertex-ai",
        action="store_true",
        help="Use Vertex AI (ADC auth). Required for video-native chunking.",
    )
    parser.add_argument("--gcp-project", help="Google Cloud project id")
    parser.add_argument(
        "--gcp-location", help="Google Cloud location (default: us-central1)"
    )
    parser.add_argument("--gcs-bucket", help="GCS bucket for uploads (local videos)")

    parser.add_argument(
        "--gemini-model",
        help="Gemini model id (default: GEMINI_MODEL env or gemini-3-flash-preview)",
    )

    parser.add_argument(
        "--video-native",
        dest="video_native",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable video-native keyframe selection",
    )

    args = parser.parse_args()

    if not args.video_path and not args.video_url:
        parser.error("Provide either video_path or --video-url")

    _apply_runtime_config(args)
    Config.validate()

    video_path = args.video_path
    if args.video_url:
        downloads_dir = os.path.join(Config.TEMP_DIR, "downloads")
        print(f"Downloading video from URL: {args.video_url}")
        video_path = download_video_url(args.video_url, downloads_dir)
        print(f"Downloaded to: {video_path}")

    if not video_path or not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    print(f"Processing video: {video_path}")

    try:
        vlm_provider = GeminiProvider(model_name=Config.GEMINI_MODEL)
        orchestrator = FrameSelectorOrchestrator(vlm_provider)
        html_generator = HTMLGenerator()

        result_data = orchestrator.process_video(video_path)

        project_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = html_generator.generate_document(result_data, project_name)

        print("\n✅ Processing Complete! Open the result here:")
        print(f"file://{os.path.abspath(output_path)}")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
