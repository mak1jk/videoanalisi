import gradio as gr
import os
import shutil

from video_keyframe_extractor.config import Config
from video_keyframe_extractor.core.orchestrator import FrameSelectorOrchestrator
from video_keyframe_extractor.output.html_generator import HTMLGenerator
from video_keyframe_extractor.utils.download_utils import download_video_url
from video_keyframe_extractor.vlm_providers.gemini_provider import GeminiProvider
from video_keyframe_extractor.vlm_providers.openrouter_provider import (
    OpenRouterProvider,
)


def get_vlm_models(provider_type: str, use_vertex_ai: bool) -> list[str]:
    """Fetch available models based on selected provider."""
    try:
        if provider_type == "Gemini":
            Config.USE_VERTEX_AI = use_vertex_ai
            provider = GeminiProvider()
        else:
            provider = OpenRouterProvider()
        return provider.get_available_models()
    except Exception as e:
        print(f"Error fetching models: {e}")
        if provider_type == "Gemini":
            return [Config.GEMINI_MODEL]
        return ["google/gemini-flash-1.5"]


def process_video_pipeline(
    video_file,
    video_url: str,
    provider_type: str,
    use_vertex_ai: bool,
    gcp_project: str,
    gcp_location: str,
    gcs_bucket: str,
    model_name: str,
    api_key_input: str,
    progress=gr.Progress(),
):
    if not video_file and not (video_url and video_url.strip()):
        return None, None, "Please upload a video file or provide a URL."

    # Apply runtime config
    Config.USE_VERTEX_AI = bool(use_vertex_ai)
    os.environ["USE_VERTEX_AI"] = "true" if Config.USE_VERTEX_AI else "false"

    if gcp_project:
        Config.GCP_PROJECT = gcp_project
        os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_project

    if gcp_location:
        Config.GCP_LOCATION = gcp_location
        os.environ["GOOGLE_CLOUD_LOCATION"] = gcp_location

    if gcs_bucket:
        Config.GCS_BUCKET = gcs_bucket
        os.environ["VIDEOANALISI_GCS_BUCKET"] = gcs_bucket

    if model_name and model_name.strip():
        Config.GEMINI_MODEL = model_name.strip()
        os.environ["GEMINI_MODEL"] = Config.GEMINI_MODEL

    if provider_type == "Gemini" and not Config.USE_VERTEX_AI:
        if api_key_input:
            Config.GOOGLE_API_KEY = api_key_input
            os.environ["GOOGLE_API_KEY"] = api_key_input
        if not Config.GOOGLE_API_KEY:
            return None, None, "Error: Google API Key is missing (or enable Vertex AI)."

    if provider_type == "OpenRouter":
        if api_key_input:
            Config.OPENROUTER_API_KEY = api_key_input
            os.environ["OPENROUTER_API_KEY"] = api_key_input
        if not Config.OPENROUTER_API_KEY:
            return None, None, "Error: OpenRouter API Key is missing."

    Config.validate()

    # Resolve video input
    progress(0, desc="Preparing video input...")

    if video_url and video_url.strip():
        downloads_dir = os.path.join(Config.TEMP_DIR, "downloads")
        video_path = download_video_url(video_url.strip(), downloads_dir)
    else:
        video_path = video_file

    progress(0.05, desc="Initializing pipeline...")

    try:
        if provider_type == "Gemini":
            vlm_provider = GeminiProvider(model_name=Config.GEMINI_MODEL)
        else:
            vlm_provider = OpenRouterProvider(model_name=model_name)

        orchestrator = FrameSelectorOrchestrator(vlm_provider)
        html_generator = HTMLGenerator()

        progress(0.1, desc="Step 1: Audio Extraction & Transcription (Whisper)...")
        result_data = orchestrator.process_video(video_path)

        progress(0.85, desc="Step 2: Generating HTML Report...")
        project_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = html_generator.generate_document(result_data, project_name)

        progress(1.0, desc="Done!")

        # Create a zip for download
        output_dir = os.path.dirname(output_path)
        zip_path = shutil.make_archive(output_dir, "zip", output_dir)

        return output_path, zip_path, f"Successfully processed! Output at {output_path}"

    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, None, f"Error occurred: {str(e)}"


def build_app():
    with gr.Blocks(title="Video Keyframe Extractor") as demo:
        gr.Markdown("# Video Keyframe Extractor")

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="Upload Video (MP4, AVI, MOV, MKV)", sources=["upload"]
                )
                video_url = gr.Textbox(
                    label="...or Video URL (YouTube / direct)",
                    placeholder="https://www.youtube.com/watch?v=... or https://example.com/video.mp4",
                )

                with gr.Accordion("Provider Settings", open=True):
                    provider_radio = gr.Radio(
                        ["Gemini", "OpenRouter"],
                        label="Select Provider",
                        value="Gemini",
                    )

                    use_vertex_ai = gr.Checkbox(
                        label="Use Vertex AI (recommended for video-native)", value=True
                    )

                    gcp_project = gr.Textbox(
                        label="Google Cloud Project (GOOGLE_CLOUD_PROJECT)",
                        placeholder="my-gcp-project",
                    )
                    gcp_location = gr.Textbox(
                        label="Google Cloud Location (GOOGLE_CLOUD_LOCATION)",
                        value=Config.GCP_LOCATION,
                    )
                    gcs_bucket = gr.Textbox(
                        label="GCS Bucket for uploads (VIDEOANALISI_GCS_BUCKET)",
                        placeholder="my-bucket",
                    )

                    api_key_input = gr.Textbox(
                        label="API Key (only for non-Vertex Gemini or OpenRouter)",
                        placeholder="Paste your API Key here",
                        type="password",
                    )

                    model_dropdown = gr.Dropdown(
                        choices=[Config.GEMINI_MODEL],
                        value=Config.GEMINI_MODEL,
                        label="Model",
                        interactive=True,
                        allow_custom_value=True,
                    )

                    refresh_btn = gr.Button("Refresh Models", size="sm")

                process_btn = gr.Button("Process Video", variant="primary")

            with gr.Column(scale=1):
                status_output = gr.Textbox(label="Status Log", lines=6)
                zip_file = gr.File(label="Download HTML + Images Package")

        def refresh_models_with_key(provider_type, use_vertex):
            models = get_vlm_models(provider_type, use_vertex)
            if models:
                return gr.update(
                    choices=models, value=models[0]
                ), f"Fetched {len(models)} models"
            return (
                gr.update(choices=["No models available"], value="No models available"),
                "No models available",
            )

        refresh_btn.click(
            fn=refresh_models_with_key,
            inputs=[provider_radio, use_vertex_ai],
            outputs=[model_dropdown, status_output],
        )

        process_btn.click(
            fn=process_video_pipeline,
            inputs=[
                video_input,
                video_url,
                provider_radio,
                use_vertex_ai,
                gcp_project,
                gcp_location,
                gcs_bucket,
                model_dropdown,
                api_key_input,
            ],
            outputs=[gr.File(visible=False), zip_file, status_output],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
