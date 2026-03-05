import json
import math
import os
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from .base import VLMProvider
from ..config import Config

# Gemini Files API: ~263 tokens/sec at default resolution.
# 1M token limit → max ~3800 sec safe. We use 1700s chunks with margin.
_MAX_CHUNK_SECONDS = 1700


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
Sei un esperto video editor e creatore di contenuti.
Ti fornisco una trascrizione grezza di un video, con timestamp.
Il tuo compito e segmentare questa trascrizione in sezioni logiche (slide/argomenti).

Regole:
1. Raggruppa le frasi che appartengono allo stesso argomento o contesto visivo.
2. Fornisci un titolo breve per ogni sezione (in italiano).
3. Usa i timestamp forniti; start_time deve essere l'inizio del primo segmento, end_time la fine dell'ultimo.
4. Il testo riassuntivo deve essere in italiano.
5. L'output DEVE essere JSON valido nel seguente formato:
{
  "sections": [
    {
      "title": "Introduzione",
      "start_time": 0.0,
      "end_time": 15.5,
      "text": "..."
    }
  ]
}
Restituisci SOLO JSON grezzo (no markdown).
""".strip()

        for attempt in range(5):
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
                msg = str(e)
                wait = 60
                m = re.search(r'retry in (\d+)s', msg)
                if m:
                    wait = int(m.group(1)) + 5
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    print(f"  Gemini rate limit. Waiting {wait}s before retry ({attempt+1}/5)...")
                    time.sleep(wait)
                else:
                    print(f"Error during segmentation: {e}")
                    return '{"sections": []}'
        print("Gemini rate limit exceeded after 5 retries.")
        return '{"sections": []}'

    def _get_video_duration(self, video_path: str) -> float:
        result = subprocess.check_output([
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", video_path
        ])
        return float(result.decode().strip())

    def _upload_and_wait(self, video_path: str, label: str) -> types.File:
        size_mb = os.path.getsize(video_path) / 1024 / 1024
        print(f"Uploading {label} ({size_mb:.0f} MB)...")
        file = self.client.files.upload(
            file=video_path,
            config=types.UploadFileConfig(
                mime_type="video/mp4",
                display_name=os.path.basename(video_path),
            ),
        )
        print(f"Waiting for processing...", end="", flush=True)
        while file.state.name == "PROCESSING":
            time.sleep(5)
            print(".", end="", flush=True)
            file = self.client.files.get(name=file.name)
        print()
        if file.state.name != "ACTIVE":
            raise RuntimeError(f"Video processing failed: {file.state.name}")
        return file

    def _analyze_chunk(self, file: types.File, offset: float, max_sections: int) -> List[Dict]:
        schema: Dict[str, Any] = {
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
                            "keyframe_time": {"type": "NUMBER"},
                            "tasks": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "description": {"type": "STRING"},
                                        "assignee": {"type": "STRING"},
                                        "priority": {"type": "STRING"},
                                    },
                                    "required": ["description"],
                                },
                            },
                        },
                        "required": ["title", "start_time", "end_time", "text", "keyframe_time"],
                    },
                }
            },
            "required": ["sections"],
        }

        offset_note = f"NOTA: i timestamp in questo clip partono da 0s ma corrispondono a {offset:.0f}s nel video completo. Aggiungi {offset:.0f} a tutti i timestamp." if offset > 0 else ""
        prompt = f"""Sei un esperto analista di video educativi e riunioni stand-up.
Analizza questo video clip e identifica fino a {max_sections} sezioni o argomenti logici.
{offset_note}

Per ogni sezione fornisci:
- title: titolo breve e descrittivo (max 8 parole, in italiano)
- start_time: timestamp in secondi NEL VIDEO COMPLETO (aggiungi {offset:.0f}s di offset)
- end_time: timestamp in secondi NEL VIDEO COMPLETO (aggiungi {offset:.0f}s di offset)
- text: riassunto del contenuto in 2-3 frasi in italiano
- keyframe_time: miglior timestamp NEL VIDEO COMPLETO per uno screenshot
- tasks: array di task/azioni menzionate nella sezione. Per ogni task:
  - description: descrizione dell'attivita o azione da svolgere (in italiano)
  - assignee: persona a cui e assegnata (se menzionata, altrimenti stringa vuota)
  - priority: "alta", "media", "bassa" (se deducibile dal contesto, altrimenti "media")
  Se non ci sono task nella sezione, usa un array vuoto [].

Presta particolare attenzione a:
- Assegnazioni di lavoro ("io faccio...", "tu ti occupi di...", "bisogna fare...")
- Scadenze e deadline menzionate
- Problemi da risolvere e chi se ne occupa
- Decisioni prese durante la riunione

Restituisci SOLO JSON valido."""

        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        types.Part.from_uri(file_uri=file.uri, mime_type="video/mp4"),
                        prompt,
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
                return parsed.get("sections", [])
            except json.JSONDecodeError as e:
                print(f"  JSON parse error (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    time.sleep(5)
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    wait = 60
                    m = re.search(r'retry in (\d+)s', msg)
                    if m:
                        wait = int(m.group(1)) + 5
                    print(f"  Rate limit (attempt {attempt+1}/3). Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        return []

    def _count_tokens_for_file(self, file: types.File) -> int:
        """Count tokens for an uploaded file. Returns 0 on error."""
        try:
            resp = self.client.models.count_tokens(
                model=self.model_name,
                contents=[
                    types.Part.from_uri(file_uri=file.uri, mime_type="video/mp4"),
                    "Analizza questo video.",
                ],
            )
            return resp.total_tokens
        except Exception as e:
            print(f"  Token count unavailable: {e}")
            return 0

    def upload_and_analyze_video(self, video_path: str, max_sections: int = 40) -> dict:
        """Upload video to Gemini Files API. Auto-chunks videos > 28 min to stay under 1M token limit.
        Returns dict with 'sections' list and 'cost_info' dict."""
        duration = self._get_video_duration(video_path)
        print(f"Video duration: {duration/60:.1f} min")
        total_tokens = 0
        _PRICE_PER_M = 0.075  # USD per 1M input tokens (Gemini Flash)

        if duration <= _MAX_CHUNK_SECONDS:
            # Single chunk
            file = self._upload_and_wait(video_path, os.path.basename(video_path))
            tokens = self._count_tokens_for_file(file)
            total_tokens += tokens
            if tokens:
                cost = (tokens / 1_000_000) * _PRICE_PER_M
                print(f"  Token stimati: {tokens:,} (~${cost:.4f} USD)")
            print(f"Analyzing with {self.model_name}...")
            try:
                sections = self._analyze_chunk(file, offset=0.0, max_sections=max_sections)
                print(f"Gemini returned {len(sections)} sections.")
                cost_info = {"total_tokens": total_tokens, "estimated_cost_usd": (total_tokens / 1_000_000) * _PRICE_PER_M, "model": self.model_name}
                return {"sections": sections, "cost_info": cost_info}
            except Exception as e:
                print(f"Error analyzing video: {e}")
                return {"sections": [], "cost_info": {}}
            finally:
                try:
                    self.client.files.delete(name=file.name)
                    print("Cleaned up uploaded file.")
                except Exception:
                    pass
        else:
            # Multi-chunk: split with ffmpeg
            n_chunks = math.ceil(duration / _MAX_CHUNK_SECONDS)
            chunk_dur = duration / n_chunks
            sections_per_chunk = max(10, max_sections // n_chunks)
            print(f"Video too long ({duration/60:.1f} min). Splitting into {n_chunks} chunks of ~{chunk_dur/60:.1f} min each.")

            all_sections: List[Dict] = []

            with tempfile.TemporaryDirectory() as tmpdir:
                for i in range(n_chunks):
                    start = i * chunk_dur
                    chunk_path = os.path.join(tmpdir, f"chunk_{i:02d}.mp4")
                    subprocess.run([
                        "ffmpeg", "-y", "-ss", str(start), "-t", str(chunk_dur),
                        "-i", video_path, "-c", "copy", chunk_path
                    ], check=True, capture_output=True)

                    print(f"\n--- Chunk {i+1}/{n_chunks} ({start/60:.1f}min - {(start+chunk_dur)/60:.1f}min) ---")
                    file = self._upload_and_wait(chunk_path, f"chunk {i+1}/{n_chunks}")
                    tokens = self._count_tokens_for_file(file)
                    total_tokens += tokens
                    if tokens:
                        cost = (tokens / 1_000_000) * _PRICE_PER_M
                        print(f"  Token stimati chunk {i+1}: {tokens:,} (~${cost:.4f} USD)")
                    print(f"Analyzing chunk {i+1} with {self.model_name}...")
                    try:
                        chunk_sections = self._analyze_chunk(file, offset=start, max_sections=sections_per_chunk)
                        print(f"  Got {len(chunk_sections)} sections from chunk {i+1}.")
                        all_sections.extend(chunk_sections)
                    except Exception as e:
                        print(f"  Error analyzing chunk {i+1}: {e}")
                    finally:
                        try:
                            self.client.files.delete(name=file.name)
                        except Exception:
                            pass

            # Sort by start_time
            all_sections.sort(key=lambda s: s.get("start_time", 0))
            total_cost = (total_tokens / 1_000_000) * _PRICE_PER_M
            print(f"\nTotal sections: {len(all_sections)} | Token totali: {total_tokens:,} | Costo stimato: ${total_cost:.4f} USD")
            cost_info = {"total_tokens": total_tokens, "estimated_cost_usd": total_cost, "model": self.model_name}
            return {"sections": all_sections, "cost_info": cost_info}

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
            media_resolution = types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW

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
