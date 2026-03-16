import argparse
import http.server
import json
import os
import functools
import threading
import webbrowser

from video_keyframe_extractor.config import Config
from video_keyframe_extractor.core.orchestrator import FrameSelectorOrchestrator
from video_keyframe_extractor.output.html_generator import HTMLGenerator
from video_keyframe_extractor.utils.download_utils import download_video_url
from video_keyframe_extractor.vlm_providers.gemini_provider import GeminiProvider


class _RangeHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with Range request support for video seeking."""

    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()

        range_header = self.headers.get("Range")
        if range_header is None:
            return super().send_head()

        # Parse Range: bytes=START-END
        try:
            range_spec = range_header.replace("bytes=", "")
            parts = range_spec.split("-")
            file_size = os.path.getsize(path)
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1
        except (ValueError, IndexError):
            return super().send_head()

        ctype = self.guess_type(path)
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404)
            return None

        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        # Wrap to limit bytes sent
        return _LimitedFile(f, length)

    def log_message(self, format, *args):
        # Suppress noisy request logs
        pass


class _LimitedFile:
    """File wrapper that limits read to a specific number of bytes."""

    def __init__(self, f, limit):
        self._f = f
        self._remaining = limit

    def read(self, size=-1):
        if self._remaining <= 0:
            return b""
        if size < 0 or size > self._remaining:
            size = self._remaining
        data = self._f.read(size)
        self._remaining -= len(data)
        return data

    def close(self):
        self._f.close()


def _serve_directory(directory: str, port: int) -> None:
    """Start a local HTTP server with Range support for video seeking."""
    handler = functools.partial(_RangeHTTPRequestHandler, directory=directory)
    server = http.server.HTTPServer(("0.0.0.0", port), handler)
    url = f"http://localhost:{port}/index.html"
    print(f"\nServer HTTP avviato: {url}")
    print("Premi Ctrl+C per terminare.\n")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer terminato.")
        server.server_close()


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

    if hasattr(args, "transcriber") and args.transcriber:
        Config.TRANSCRIBER = args.transcriber


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

    parser.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help="Fast mode: upload video to Gemini Files API, skip CLIP/Whisper/scene detection (default: on)",
    )
    parser.add_argument(
        "--no-fast",
        dest="fast",
        action="store_false",
        help="Disable fast mode, use full local pipeline (Whisper + CLIP)",
    )

    parser.add_argument(
        "--transcriber",
        choices=["groq", "local"],
        default=None,
        help=(
            "Transcription backend: 'groq' uses Groq Whisper API (requires GROQ_API_KEY), "
            "'local' uses faster-whisper offline (requires: pip install faster-whisper). "
            "Default: groq if GROQ_API_KEY is set, else local."
        ),
    )

    parser.add_argument(
        "--regenerate",
        metavar="PROJECT_DIR",
        help="Regenerate HTML from saved data.json (no API calls). Pass the output project directory.",
    )

    parser.add_argument(
        "--serve",
        nargs="?",
        const=8080,
        type=int,
        metavar="PORT",
        help="Start a local HTTP server to view the report (default port: 8080). Solves browser video playback issues.",
    )

    args = parser.parse_args()

    # Regenerate mode: re-render from cached data.json
    if args.regenerate:
        project_dir = args.regenerate
        json_path = os.path.join(project_dir, "data.json")
        if not os.path.exists(json_path):
            print(f"Error: {json_path} not found. Run full processing first.")
            return
        with open(json_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        project_name = os.path.basename(os.path.normpath(project_dir))
        html_gen = HTMLGenerator()
        output_path = html_gen.generate_from_cache(cached, project_name)
        print(f"\nRegenerated: file://{os.path.abspath(output_path)}")
        if args.serve:
            _serve_directory(os.path.dirname(os.path.abspath(output_path)), args.serve)
        return

    # Serve-only mode: just start HTTP server on existing output
    if args.serve and not args.video_path and not args.video_url and not args.regenerate:
        # Try to serve the output_documents directory
        output_base = "output_documents"
        if not os.path.isdir(output_base):
            print(f"Error: {output_base} not found.")
            return
        _serve_directory(os.path.abspath(output_base), args.serve)
        return

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
    mode = "FAST (Gemini Files API)" if args.fast else "FULL (local pipeline)"
    print(f"Mode: {mode}")

    try:
        vlm_provider = GeminiProvider(model_name=Config.GEMINI_MODEL)
        orchestrator = FrameSelectorOrchestrator(vlm_provider, fast_mode=args.fast)
        html_generator = HTMLGenerator()

        if args.fast:
            result_data = orchestrator.process_video_fast(video_path)
        else:
            result_data = orchestrator.process_video(video_path)

        project_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = html_generator.generate_document(result_data, project_name)

        print("\n✅ Processing Complete! Open the result here:")
        print(f"file://{os.path.abspath(output_path)}")

        if args.serve:
            _serve_directory(os.path.dirname(os.path.abspath(output_path)), args.serve)

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
