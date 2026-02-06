import cv2
import os
from typing import Optional

class FrameSampler:
    def __init__(self, output_dir="temp_processing/frames"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def sample_frames_uniform(self, video_path: str, interval_sec: int = 1):
        """
        Extracts frames at a fixed time interval.
        Returns a list of dicts: {'timestamp': float, 'path': str, 'frame_idx': int}
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            cap.release()
            raise ValueError(f"Invalid FPS ({fps}) for video: {video_path}")

        interval_frames = max(1, int(fps * interval_sec))
        
        frames_data = []
        frame_idx = 0
        
        print(f"Sampling frames every {interval_sec} second(s)...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval_frames == 0:
                timestamp = frame_idx / fps
                frame_name = f"frame_{frame_idx}_{timestamp:.2f}.jpg"
                frame_path = os.path.join(self.output_dir, frame_name)
                
                cv2.imwrite(frame_path, frame)
                frames_data.append({
                    'timestamp': timestamp,
                    'path': frame_path,
                    'frame_idx': frame_idx
                })

            frame_idx += 1

        cap.release()
        return frames_data

    def get_frame_at_timestamp(self, video_path: str, timestamp: float) -> Optional[str]:
        """
        Extracts a single frame at a specific timestamp.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            cap.release()
            return None
        frame_no = int(fps * timestamp)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_name = f"keyframe_{timestamp:.2f}.jpg"
            frame_path = os.path.join(self.output_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            return frame_path
        else:
            return None
