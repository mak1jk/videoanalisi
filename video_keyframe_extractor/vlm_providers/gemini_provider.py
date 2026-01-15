import google.generativeai as genai
import os
import json
from typing import List, Optional
from .base import VLMProvider
from ..config import Config

class GeminiProvider(VLMProvider):
    def __init__(self, model_name="models/gemini-1.5-flash-latest"):
        api_key = Config.GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def get_available_models(self) -> List[str]:
        try:
            models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    name = m.name.replace("models/", "")
                    if "flash" in name or "pro" in name:
                        models.append(name)
            return models
        except Exception as e:
            print(f"Error fetching models: {e}")
            return ["gemini-1.5-flash"]

    def analyze_image_text(self, text_segment: str, image_paths: List[str], prompt: str = None) -> Optional[int]:
        """
        Analyzes text and images to select the best matching image.
        Returns the index of the best image in image_paths, or None if no image is relevant.
        """
        if not image_paths:
            return None

        # Prepare images for Gemini
        try:
            pil_images = []
            for path in image_paths:
                if os.path.exists(path):
                    pil_images.append(genai.upload_file(path))
                else:
                    print(f"Warning: Image not found at {path}")
                    # Maintain index alignment by appending None or handling carefully
                    # For MVP, we skip, but this affects index return. 
                    # Better: Filter paths before passing here.
            
            if not pil_images:
                return None
                
            default_prompt = f"""
            Task: Select the best image that visually represents the following text segment from a video presentation.
            
            Text Segment: "{text_segment}"
            
            Candidate Images are provided in order.
            
            Instructions:
            1. Analyze the text to understand the key visual topic (e.g., a specific slide title, a diagram, a code snippet).
            2. Look at each image and determine which one matches the text best.
            3. If an image is a slide, check if the text content matches the slide title or bullets.
            4. If no image is relevant (e.g., just a talking head when the text is about code), return -1.
            5. If multiple images are similar, pick the sharpest/clearest one.
            
            Output strictly a JSON object:
            {{
                "selected_index": <integer index of the best image, or -1 if none>,
                "reasoning": "<short explanation>"
            }}
            """
            
            final_prompt = prompt if prompt else default_prompt
            
            # Send to model
            response = self.model.generate_content([final_prompt] + pil_images)
            response_text = response.text
            
            # Parse JSON
            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned_text)
            selected_idx = data.get("selected_index", -1)
            
            if selected_idx >= 0 and selected_idx < len(image_paths):
                return selected_idx
            return None
            
        except Exception as e:
            print(f"Error in analyze_image_text: {e}")
            return None
    def segment_text(self, full_transcript_json: str) -> str:
        """
        Segments the transcript using Gemini.
        full_transcript_json: JSON string of Whisper segments.
        """
        prompt = """
        You are an expert video editor and content creator.
        I will provide you with a raw transcript of a video, including word-level timestamps.
        Your task is to segment this transcript into logical sections (e.g., slides, distinct topics, or scenes).
        
        Rules:
        1. Group sentences that belong to the same topic or visual context.
        2. Provide a short title for each section.
        3. Precise start and end times are critical. Use the 'start' of the first word and 'end' of the last word in the section.
        4. Output MUST be valid JSON in the following format:
        {
            "sections": [
                {
                    "title": "Introduction",
                    "start_time": 0.0,
                    "end_time": 15.5,
                    "text": "Hello everyone..."
                },
                ...
            ]
        }
        
        Do not add any markdown formatting (like ```json). Just the raw JSON string.
        """
        
        # We might need to chunk this if it's too long, but for MVP we send it all.
        # Ensure we don't exceed token limits.
        
        try:
            response = self.model.generate_content([prompt, full_transcript_json])
            return response.text
        except Exception as e:
            print(f"Error during segmentation: {e}")
            return "{\"sections\": []}"
