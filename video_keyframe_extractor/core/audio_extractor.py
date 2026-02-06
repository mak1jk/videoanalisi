import ffmpeg
import os

class AudioExtractor:
    def __init__(self, output_dir="temp_processing"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_audio(self, video_path: str) -> str:
        """
        Extracts audio from video and saves it as WAV (16kHz, mono) for optimal Whisper usage.
        Returns the path to the extracted audio file.
        """
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(self.output_dir, f"{base_name}.wav")

        print(f"Extracting audio to {output_path}...")
        
        try:
            (
                ffmpeg
                .input(video_path)
                .output(output_path, acodec='pcm_s16le', ac=1, ar='16000')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            return output_path
        except ffmpeg.Error as e:
            stderr = e.stderr.decode("utf8", errors="replace") if e.stderr else str(e)
            raise RuntimeError(f"FFmpeg audio extraction failed: {stderr}") from e
