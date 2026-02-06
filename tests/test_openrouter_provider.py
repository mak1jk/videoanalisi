import unittest
from unittest.mock import Mock, patch

from video_keyframe_extractor.config import Config
from video_keyframe_extractor.vlm_providers.openrouter_provider import OpenRouterProvider


class OpenRouterProviderTests(unittest.TestCase):
    def setUp(self):
        Config.OPENROUTER_API_KEY = "test-key"

    @patch("video_keyframe_extractor.vlm_providers.openrouter_provider.requests.get")
    def test_get_available_models_uses_timeout_and_raise_for_status(self, get_mock):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {"data": [{"id": "openai/gpt-4o-mini"}]}
        get_mock.return_value = response

        provider = OpenRouterProvider()
        models = provider.get_available_models()

        self.assertIn("openai/gpt-4o-mini", models)
        get_mock.assert_called_once()
        self.assertEqual(get_mock.call_args.kwargs["timeout"], provider.timeout)
        response.raise_for_status.assert_called_once()

    @patch("video_keyframe_extractor.vlm_providers.openrouter_provider.requests.post")
    def test_segment_text_uses_timeout_and_raise_for_status(self, post_mock):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": '{"sections": []}'}}]
        }
        post_mock.return_value = response

        provider = OpenRouterProvider()
        out = provider.segment_text("hello")

        self.assertEqual(out, '{"sections": []}')
        post_mock.assert_called_once()
        self.assertEqual(post_mock.call_args.kwargs["timeout"], provider.timeout)
        response.raise_for_status.assert_called_once()


if __name__ == "__main__":
    unittest.main()
