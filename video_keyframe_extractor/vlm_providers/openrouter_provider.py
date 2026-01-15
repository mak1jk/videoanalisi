import os
import json
import base64
import requests
from typing import List, Optional
from .base import VLMProvider
from ..config import Config

class OpenRouterProvider(VLMProvider):
    def __init__(self, model_name="google/gemini-flash-1.5"):
        self.api_key = Config.OPENROUTER_API_KEY
        self.model_name = model_name
        self.base_url = "https://openrouter.ai/api/v1"

    def get_available_models(self) -> List[str]:
        # OpenRouter ha troppi modelli, mettiamo i principali vision come default
        # ma potremmo fetchare la lista completa se necessario
        return [
            "google/gemini-flash-1.5",
            "google/gemini-pro-1.5",
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o-mini",
            "meta-llama/llama-3.2-11b-vision-instruct"
        ]

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_image_text(self, text_segment: str, image_paths: List[str], prompt: str = None) -> Optional[int]:
        if not self.api_key:
            print("Error: OPENROUTER_API_KEY not set")
            return None

        content = [
            {
                "type": "text",
                "text": prompt if prompt else f"""
                Task: Select the best image that visually represents: "{text_segment}"
                Return ONLY a JSON object with "selected_index" (0 to N-1, or -1 if none).
                """
            }
        ]

        for i, path in enumerate(image_paths):
            if os.path.exists(path):
                base64_image = self._encode_image(path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })

        try:
            response = requests.post(
                url=f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://github.com/wonderwall/videoanalisi", # Optional
                },
                data=json.dumps({
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": content}],
                    "response_format": {"type": "json_object"}
                })
            )
            
            result = response.json()
            response_text = result['choices'][0]['message']['content']
            data = json.loads(response_text)
            selected_idx = data.get("selected_index", -1)
            
            if 0 <= selected_idx < len(image_paths):
                return selected_idx
            return None
            
        except Exception as e:
            print(f"OpenRouter Error: {e}")
            return None

    def segment_text(self, full_transcript_json: str) -> str:
        if not self.api_key:
            return "{\"sections\": []}"

        prompt = """
        Analyze this transcript with word timestamps and divide it into semantic sections (topics/slides).
        Return JSON format: {"sections": [{"title": "...", "start_time": 0.0, "end_time": 10.0, "text": "..."}]}
        """

        try:
            response = requests.post(
                url=f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                data=json.dumps({
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "You are a video analysis expert. Output ONLY raw JSON."},
                        {"role": "user", "content": f"{prompt}\n\nTranscript: {full_transcript_json}"}
                    ]
                })
            )
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"OpenRouter Segmentation Error: {e}")
            return "{\"sections\": []}"
