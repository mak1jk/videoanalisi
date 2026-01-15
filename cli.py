import argparse
import os
from video_keyframe_extractor.core.orchestrator import FrameSelectorOrchestrator
from video_keyframe_extractor.vlm_providers.gemini_provider import GeminiProvider
from video_keyframe_extractor.output.html_generator import HTMLGenerator
from video_keyframe_extractor.config import Config

def main():
    parser = argparse.ArgumentParser(description="Video Keyframe Extractor CLI")
    parser.add_argument("video_path", help="Path to the input video file")
    
    args = parser.parse_args()
    
    video_path = args.video_path
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Check API Key
    if not Config.GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found. Please set it in .env file.")
        return

    print(f"Processing video: {video_path}")
    
    # Initialize components
    try:
        vlm_provider = GeminiProvider()
        orchestrator = FrameSelectorOrchestrator(vlm_provider)
        html_generator = HTMLGenerator()
        
        # Run Pipeline
        result_data = orchestrator.process_video(video_path)
        
        # Generate Output
        project_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = html_generator.generate_document(result_data, project_name)
        
        print(f"\n✅ Processing Complete! Open the result here:")
        print(f"file://{os.path.abspath(output_path)}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
