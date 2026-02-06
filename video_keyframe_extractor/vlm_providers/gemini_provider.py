import json
import os
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from .base import VLMProvider
from ..config import Config


class GeminiProvider(VLMProvider):
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or Config.GEMINI_MODEL

        if Config.USE_VERTEX_AI:
            if not Config.GCP_PROJECT:
                raise ValueError(
                    "GOOGLE_CLOUD_PROJECT is required when USE_VERTEX_AI=true"
                )
            self.client = genai.Client(
                vertexai=True,
                project=Config.GCP_PROJECT,
                location=Config.GCP_LOCATION,
            )
        else:
            api_key = Config.GOOGLE_API_KEY
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables.")
            self.client = genai.Client(api_key=api_key)

    def get_available_models(self) -> List[str]:
        try:
            models = []
            for m in self.client.models.list():
                name = getattr(m, "name", None) or ""
                if not name:
                    continue
                models.append(name)
            return sorted(models)
        except Exception as e:
            print(f"Error fetching models: {e}")
            return [Config.GEMINI_MODEL]

    def analyze_image_text(
        self, text_segment: str, image_paths: List[str], prompt: Optional[str] = None
    ) -> Optional[int]:
        if not image_paths:
            return None

        image_parts: List[types.Part] = []
        for path in image_paths:
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                image_parts.append(
                    types.Part.from_bytes(data=f.read(), mime_type="image/jpeg")
                )

        if not image_parts:
            return None

        default_prompt = f"""
Task: Select the best image that visually represents the following text segment from a video presentation.

Text Segment: "{text_segment}"

Candidate Images are provided in order.

Instructions:
1. Analyze the text to understand the key visual topic (e.g., a slide title, a diagram, a code snippet).
2. Pick the single best matching image.
3. If no image is relevant, return -1.

Output strictly a JSON object:
{{
  "selected_index": <integer index of the best image, or -1 if none>,
  "reasoning": "<short explanation>"
}}
""".strip()

        final_prompt = prompt or default_prompt

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[final_prompt, *image_parts],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "selected_index": {"type": "INTEGER"},
                            "reasoning": {"type": "STRING"},
                        },
                        "required": ["selected_index"],
                    },
                ),
            )

            data = (
                response.parsed
                if response.parsed is not None
                else json.loads(response.text)
            )
            selected_idx = int(data.get("selected_index", -1))

            if 0 <= selected_idx < len(image_paths):
                return selected_idx
            return None
        except Exception as e:
            print(f"Error in analyze_image_text: {e}")
            return None

    def segment_text(self, full_transcript: str) -> str:
        prompt = """
You are an expert video editor and content creator.
I will provide you with a raw transcript of a video, including timestamps.
Your task is to segment this transcript into logical sections (slides/topics).

Rules:
1. Group sentences that belong to the same topic or visual context.
2. Provide a short title for each section.
3. Use the provided timestamps; start_time must be the first segment start, end_time the last segment end.
4. Output MUST be valid JSON in the following format:
{
  "sections": [
    {
      "title": "Introduction",
      "start_time": 0.0,
      "end_time": 15.5,
      "text": "..."
    }
  ]
}
Return ONLY raw JSON (no markdown).
""".strip()

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, full_transcript],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "sections": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "title": {"type": "STRING"},
                                        "start_time": {"type": "NUMBER"},
                                        "end_time": {"type": "NUMBER"},
                                        "text": {"type": "STRING"},
                                    },
                                    "required": [
                                        "title",
                                        "start_time",
                                        "end_time",
                                        "text",
                                    ],
                                },
                            }
                        },
                        "required": ["sections"],
                    },
                ),
            )

            if response.text:
                return response.text
            return json.dumps(response.parsed or {"sections": []})
        except Exception as e:
            print(f"Error during segmentation: {e}")
            return '{"sections": []}'

    def supports_video_timeline(self) -> bool:
        return True

    def analyze_video_timeline(
        self,
        video_uri: str,
        *,
        max_events: int,
        mime_type: str = "video/mp4",
        prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Analyze a video and return a timeline of key visual moments."""

        schema: Dict[str, Any] = {
            "type": "OBJECT",
            "properties": {
                "events": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "timestamp_s": {"type": "NUMBER"},
                            "frame_type": {
                                "type": "STRING",
                                "enum": [
                                    "slide",
                                    "speaker",
                                    "demo",
                                    "transition",
                                    "other",
                                ],
                            },
                            "slide_text": {"type": "STRING"},
                            "description": {"type": "STRING"},
                            "confidence": {"type": "NUMBER"},
                        },
                        "required": ["timestamp_s", "frame_type"],
                    },
                }
            },
            "required": ["events"],
        }

        default_prompt = f"""
You are a video analyst.
Scan the video and return up to {max_events} key visual events that would be useful as screenshots for a video-to-document workflow.

Guidelines:
- Prefer frames where a slide changes, a diagram appears, code is shown, or a demo screen is readable.
- Include a best-effort extraction of on-screen text in `slide_text` (keep it short, key phrases only).
- Use `frame_type`: slide | speaker | demo | transition | other.
- `timestamp_s` must be in seconds from the start of THIS video file.
- Include `confidence` in [0, 1].

Return ONLY JSON.
""".strip()

        resolution = (Config.VIDEO_MEDIA_RESOLUTION or "LOW").upper()
        try:
            media_resolution = getattr(types.PartMediaResolutionLevel, resolution)
        except Exception:
            media_resolution = types.PartMediaResolutionLevel.LOW

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_uri(
                        file_uri=video_uri,
                        mime_type=mime_type,
                        media_resolution=media_resolution,
                    ),
                    prompt or default_prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=schema,
                    max_output_tokens=8192,
                ),
            )

            parsed = (
                response.parsed
                if response.parsed is not None
                else json.loads(response.text)
            )
            events = parsed.get("events") or []
            if not isinstance(events, list):
                return []
            return events
        except Exception as e:
            print(f"Error in analyze_video_timeline: {e}")
            return []
