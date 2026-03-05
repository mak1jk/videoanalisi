import os
import math
import time
import re
from groq import Groq, RateLimitError


MAX_FILE_BYTES = 25 * 1024 * 1024  # Groq limit: 25 MB per request


class GroqTranscriber:
    """
    Transcriber that uses Groq's Whisper large-v3 API.
    Supports automatic chunking for audio files > 25 MB and rate-limit retry.
    """

    def __init__(self, api_key: str = None, model: str = "whisper-large-v3"):
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.model = model

    def transcribe(self, audio_path: str, language: str = None) -> dict:
        file_size = os.path.getsize(audio_path)
        print(f"Audio size: {file_size / 1024 / 1024:.1f} MB")

        if file_size <= MAX_FILE_BYTES:
            return self._transcribe_file(audio_path, language)

        return self._transcribe_chunked(audio_path, language, file_size)

    def _transcribe_file(self, audio_path: str, language: str = None, retry: int = 5) -> dict:
        print(f"Transcribing with Groq Whisper ({self.model})...")
        for attempt in range(retry):
            try:
                with open(audio_path, "rb") as f:
                    response = self.client.audio.transcriptions.create(
                        file=(os.path.basename(audio_path), f),
                        model=self.model,
                        language=language,
                        response_format="verbose_json",
                        timestamp_granularities=["segment", "word"],
                    )
                return self._normalize_response(response)
            except RateLimitError as e:
                wait = self._parse_wait(str(e))
                print(f"  Rate limit hit. Waiting {wait}s before retry ({attempt+1}/{retry})...")
                time.sleep(wait + 5)
        raise RuntimeError(f"Groq rate limit exceeded after {retry} retries for {audio_path}")

    def _parse_wait(self, error_msg: str) -> int:
        """Extract wait seconds from Groq rate limit error message."""
        m = re.search(r'try again in (\d+)m(\d+)s', error_msg)
        if m:
            return int(m.group(1)) * 60 + int(m.group(2))
        m = re.search(r'try again in (\d+)s', error_msg)
        if m:
            return int(m.group(1))
        return 90  # default fallback

    def _transcribe_chunked(self, audio_path: str, language: str, file_size: int) -> dict:
        """Split large audio via ffmpeg and merge transcriptions."""
        import subprocess, tempfile

        duration_cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", audio_path
        ]
        duration = float(subprocess.check_output(duration_cmd).decode().strip())
        n_chunks = math.ceil(file_size / MAX_FILE_BYTES)
        chunk_dur = duration / n_chunks

        print(f"Audio too large ({file_size/1024/1024:.1f} MB). Splitting into {n_chunks} chunks of ~{chunk_dur:.0f}s")

        all_segments = []
        all_words = []
        full_text_parts = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(n_chunks):
                start = i * chunk_dur
                chunk_path = os.path.join(tmpdir, f"chunk_{i:03d}.wav")
                subprocess.run([
                    "ffmpeg", "-y", "-ss", str(start), "-t", str(chunk_dur),
                    "-i", audio_path, "-ar", "16000", "-ac", "1", chunk_path
                ], check=True, capture_output=True)

                print(f"  Chunk {i+1}/{n_chunks} ({start:.0f}s - {start+chunk_dur:.0f}s)...")
                result = self._transcribe_file(chunk_path, language)

                for seg in result.get("segments", []):
                    seg["start"] += start
                    seg["end"] += start
                    all_segments.append(seg)
                for w in result.get("words", []):
                    w["start"] += start
                    w["end"] += start
                    all_words.append(w)
                full_text_parts.append(result.get("text", ""))

        return {
            "text": " ".join(full_text_parts),
            "segments": all_segments,
            "words": all_words,
        }

    def _normalize_response(self, response) -> dict:
        """Convert Groq response to same format as openai-whisper."""
        segments = []
        words = []

        for seg in getattr(response, "segments", []) or []:
            segments.append({
                "id": getattr(seg, "id", 0),
                "start": getattr(seg, "start", 0.0),
                "end": getattr(seg, "end", 0.0),
                "text": getattr(seg, "text", ""),
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in (getattr(seg, "words", []) or [])
                ],
            })
            for w in getattr(seg, "words", []) or []:
                words.append({"word": w.word, "start": w.start, "end": w.end})

        return {
            "text": getattr(response, "text", ""),
            "segments": segments,
            "words": words,
        }
