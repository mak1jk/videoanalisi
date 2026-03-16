"""
Local transcription backend using faster-whisper.

Optional dependency: pip install faster-whisper
Falls back gracefully if not installed.
"""

import gc


class LocalWhisperTranscriber:
    """
    Transcriber that uses faster-whisper for fully local/offline transcription.
    No API key required.

    Install the optional dependency with:
        pip install faster-whisper
    """

    def __init__(self, model_size: str = "base", device: str = None, compute_type: str = None):
        try:
            from faster_whisper import WhisperModel  # noqa: F401
        except ImportError:
            raise ImportError(
                "faster-whisper is not installed. "
                "Install it with: pip install faster-whisper"
            )

        import torch

        self.model_size = model_size
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type if compute_type else (
            "float16" if self.device == "cuda" else "int8"
        )
        print(f"LocalWhisperTranscriber initialized: device={self.device}, compute_type={self.compute_type}")

    def transcribe(self, audio_path: str, language: str = None) -> dict:
        """
        Transcribe audio using faster-whisper.

        Returns a dict compatible with GroqTranscriber output:
            {
                "text": str,
                "segments": [{"start", "end", "text", "words": [...]}],
                "words": [{"word", "start", "end"}],
            }
        """
        from faster_whisper import WhisperModel

        print(f"Loading faster-whisper model '{self.model_size}'...")
        model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

        print("Transcribing audio locally...")
        kwargs = {"word_timestamps": True}
        if language:
            kwargs["language"] = language

        segments_iter, info = model.transcribe(audio_path, **kwargs)

        all_segments = []
        all_words = []
        full_text_parts = []

        for seg in segments_iter:
            words = []
            for w in (seg.words or []):
                word_entry = {"word": w.word, "start": w.start, "end": w.end}
                words.append(word_entry)
                all_words.append(word_entry)

            all_segments.append({
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": words,
            })
            full_text_parts.append(seg.text)

        # Free GPU memory
        del model
        gc.collect()
        try:
            import torch
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

        return {
            "text": "".join(full_text_parts),
            "segments": all_segments,
            "words": all_words,
        }
