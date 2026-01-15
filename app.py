import gradio as gr
import os
import shutil
import time
from video_keyframe_extractor.core.orchestrator import FrameSelectorOrchestrator
from video_keyframe_extractor.vlm_providers.gemini_provider import GeminiProvider
from video_keyframe_extractor.vlm_providers.openrouter_provider import OpenRouterProvider
from video_keyframe_extractor.config import Config

# --- Backend Logic Wrappers ---

def get_vlm_models(provider_type):
    """Fetches available models based on selected provider."""
    try:
        if provider_type == "Gemini (Direct)":
            provider = GeminiProvider()
        else:
            provider = OpenRouterProvider()
        return provider.get_available_models()
    except Exception as e:
        print(f"Error fetching models: {e}")
        return ["gemini-1.5-flash"] if provider_type == "Gemini (Direct)" else ["google/gemini-flash-1.5"]

def process_video_pipeline(video_file, provider_type, model_name, api_key_input, progress=gr.Progress()):
    """
    Main processing function for Gradio.
    """
    if video_file is None:
        return None, "Please upload a video file first."
    
    video_path = video_file
    
    # Update Config with API Key provided in UI
    if api_key_input:
        if provider_type == "Gemini (Direct)":
            Config.GOOGLE_API_KEY = api_key_input
            os.environ["GOOGLE_API_KEY"] = api_key_input
        else:
            Config.OPENROUTER_API_KEY = api_key_input
            os.environ["OPENROUTER_API_KEY"] = api_key_input
    
    # Check if we have the needed key
    if provider_type == "Gemini (Direct)" and not Config.GOOGLE_API_KEY:
        return None, "Error: Google API Key is missing."
    if provider_type == "OpenRouter" and not Config.OPENROUTER_API_KEY:
        return None, "Error: OpenRouter API Key is missing."

    progress(0, desc="Initializing pipeline...")
    
    try:
        # Initialize selected Provider
        if provider_type == "Gemini (Direct)":
            vlm_provider = GeminiProvider(model_name=f"models/{model_name}" if "models/" not in model_name else model_name)
        else:
            vlm_provider = OpenRouterProvider(model_name=model_name)
            
        orchestrator = FrameSelectorOrchestrator(vlm_provider)
        html_generator = HTMLGenerator()
        
        progress(0.1, desc="Step 1: Audio Extraction & Transcription (Whisper)...")
        result_data = orchestrator.process_video(video_path)
        
        progress(0.8, desc="Step 2: Generating HTML Report...")
        project_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = html_generator.generate_document(result_data, project_name)
        
        progress(1.0, desc="Done!")
        
        # Create a zip for download
        output_dir = os.path.dirname(output_path)
        zip_path = shutil.make_archive(output_dir, 'zip', output_dir)
        
        return output_path, zip_path, f"Successfully processed! Output at {output_path}"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"Error occurred: {str(e)}"

# --- UI Construction ---

def build_app():
    with gr.Blocks(title="Video Keyframe Extractor") as demo:
        gr.Markdown("# 🎬 Video Keyframe Extractor")
        
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Video", sources=["upload"])
                
                with gr.Accordion("⚙️ Provider Settings", open=True):
                    provider_radio = gr.Radio(
                        ["Gemini (Direct)", "OpenRouter"], 
                        label="Select Provider", 
                        value="Gemini (Direct)"
                    )
                    
                    model_dropdown = gr.Dropdown(
                        choices=get_vlm_models("Gemini (Direct)"), 
                        value="gemini-1.5-flash", 
                        label="Model", 
                        interactive=True
                    )
                    
                    api_key_input = gr.Textbox(
                        label="API Key", 
                        placeholder="Paste your API Key here", 
                        type="password"
                    )
                
                process_btn = gr.Button("🚀 Process Video", variant="primary")
            
            with gr.Column(scale=1):
                status_output = gr.Textbox(label="Status Log", lines=5)
                zip_file = gr.File(label="Download HTML + Images Package")

        # Dynamic model update based on provider
        def update_models(p):
            models = get_vlm_models(p)
            return gr.update(choices=models, value=models[0])

        provider_radio.change(fn=update_models, inputs=provider_radio, outputs=model_dropdown)
        
        process_btn.click(
            fn=process_video_pipeline,
            inputs=[video_input, provider_radio, model_dropdown, api_key_input],
            outputs=[gr.File(visible=False), zip_file, status_output]
        )

    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch()

if __name__ == "__main__":
    app = build_app()
    app.launch()
