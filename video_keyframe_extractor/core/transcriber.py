import whisper
import torch
import gc

class Transcriber:
    def __init__(self, device=None, compute_type="float16"):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type
        print(f"Transcriber initialized on device: {self.device}")

    def transcribe(self, audio_path: str, model_size="base", language=None):
        """
        Transcribes audio using Whisper with word-level timestamps.
        """
        print(f"Loading Whisper model '{model_size}'...")
        model = whisper.load_model(model_size, device=self.device)
        
        print("Transcribing audio...")
        result = model.transcribe(audio_path, language=language, word_timestamps=True)
        
        # Free memory
        del model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return result

    def format_to_text_chunks(self, transcription_result):
        """
        Helper to convert result to specific chunks if needed.
        Currently returns the raw segments.
        """
        return transcription_result["segments"]
