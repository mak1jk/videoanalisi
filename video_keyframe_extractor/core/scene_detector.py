from scenedetect import open_video, SceneManager, ContentDetector
import os

class SceneDetector:
    def __init__(self, threshold=30.0):
        self.threshold = threshold

    def detect_scenes(self, video_path: str):
        """
        Detects distinct scenes in the video.
        Returns a list of tuples: (start_time_sec, end_time_sec)
        """
        print(f"Detecting scenes in {video_path}...")
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))
        
        scene_manager.detect_scenes(video, show_progress=True)
        scene_list = scene_manager.get_scene_list()
        
        # Convert FrameTimecodes to seconds
        scenes_seconds = []
        for scene in scene_list:
            start, end = scene
            scenes_seconds.append((start.get_seconds(), end.get_seconds()))
            
        return scenes_seconds
