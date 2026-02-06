import unittest
from unittest.mock import patch

from video_keyframe_extractor.core.frame_sampler import FrameSampler


class _FakeCapture:
    def __init__(self, fps: float, opened: bool = True):
        self._fps = fps
        self._opened = opened
        self.released = False

    def isOpened(self):
        return self._opened

    def get(self, _):
        return self._fps

    def release(self):
        self.released = True


class FrameSamplerTests(unittest.TestCase):
    @patch("video_keyframe_extractor.core.frame_sampler.cv2.VideoCapture")
    def test_sample_frames_uniform_raises_on_invalid_fps(self, video_capture_mock):
        fake = _FakeCapture(fps=0.0, opened=True)
        video_capture_mock.return_value = fake
        sampler = FrameSampler(output_dir="temp_processing/test_frames")

        with self.assertRaises(ValueError):
            sampler.sample_frames_uniform("fake.mp4", interval_sec=1)

        self.assertTrue(fake.released)

    @patch("video_keyframe_extractor.core.frame_sampler.cv2.VideoCapture")
    def test_get_frame_at_timestamp_returns_none_when_fps_invalid(self, video_capture_mock):
        fake = _FakeCapture(fps=0.0, opened=True)
        video_capture_mock.return_value = fake
        sampler = FrameSampler(output_dir="temp_processing/test_frames")

        out = sampler.get_frame_at_timestamp("fake.mp4", timestamp=1.0)
        self.assertIsNone(out)
        self.assertTrue(fake.released)


if __name__ == "__main__":
    unittest.main()
