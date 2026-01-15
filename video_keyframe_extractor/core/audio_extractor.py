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
        output_path = os.path.join(self.output_dir, from_video_name=f"{base_name}.wav")
        # Removing from_video_name= part since os.path.join doesn't take keyword args, fixing it below
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
            print("FFmpeg error:", e.stderr.decode('utf8'))
            raise e
