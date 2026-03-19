import importlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_cli_module():
    sys.modules.pop("cli", None)

    config_module = types.ModuleType("video_keyframe_extractor.config")

    class _Config:
        pass

    config_module.Config = _Config

    orchestrator_module = types.ModuleType(
        "video_keyframe_extractor.core.orchestrator"
    )
    orchestrator_module.FrameSelectorOrchestrator = object

    html_generator_module = types.ModuleType(
        "video_keyframe_extractor.output.html_generator"
    )
    html_generator_module.HTMLGenerator = object

    semantic_search_module = types.ModuleType(
        "video_keyframe_extractor.core.semantic_search"
    )
    semantic_search_module.semantic_search_sections = lambda *args, **kwargs: []

    download_utils_module = types.ModuleType(
        "video_keyframe_extractor.utils.download_utils"
    )
    download_utils_module.download_video_url = lambda *args, **kwargs: None

    gemini_provider_module = types.ModuleType(
        "video_keyframe_extractor.vlm_providers.gemini_provider"
    )
    gemini_provider_module.GeminiProvider = object

    with patch.dict(
        sys.modules,
        {
            "video_keyframe_extractor.config": config_module,
            "video_keyframe_extractor.core.orchestrator": orchestrator_module,
            "video_keyframe_extractor.output.html_generator": html_generator_module,
            "video_keyframe_extractor.core.semantic_search": semantic_search_module,
            "video_keyframe_extractor.utils.download_utils": download_utils_module,
            "video_keyframe_extractor.vlm_providers.gemini_provider": gemini_provider_module,
        },
    ):
        return importlib.import_module("cli")


cli = _load_cli_module()


class RangeParserTests(unittest.TestCase):
    def test_parse_range_header_supports_explicit_start_end(self):
        self.assertEqual(cli._parse_range_header("bytes=0-99", 1000), (0, 99))

    def test_parse_range_header_supports_open_ended_ranges(self):
        self.assertEqual(cli._parse_range_header("bytes=100-", 1000), (100, 999))

    def test_parse_range_header_supports_suffix_ranges(self):
        self.assertEqual(cli._parse_range_header("bytes=-100", 1000), (900, 999))

    def test_parse_range_header_rejects_invalid_ranges(self):
        invalid_ranges = [
            "bytes=abc-def",
            "bytes=100-99",
            "bytes=1000-1001",
            "bytes=-0",
            "bytes=-",
            "bytes=0-99,200-299",
            "items=0-10",
        ]

        for header in invalid_ranges:
            with self.subTest(header=header):
                self.assertIsNone(cli._parse_range_header(header, 1000))


class _HandlerHarness(cli._RangeHTTPRequestHandler):
    def __init__(self, file_path, range_header):
        self.file_path = str(file_path)
        self.path = "/video.mp4"
        self.headers = {"Range": range_header} if range_header is not None else {}
        self.responses = []
        self.sent_headers = {}
        self.end_headers_called = False
        self.fallback_called = False

    def translate_path(self, _path):
        return self.file_path

    def guess_type(self, _path):
        return "video/mp4"

    def send_response(self, status_code):
        self.responses.append(status_code)

    def send_header(self, name, value):
        self.sent_headers[name] = value

    def end_headers(self):
        self.end_headers_called = True

    def send_error(self, code):
        raise AssertionError(f"unexpected send_error({code})")

    def call_send_head(self):
        with patch.object(
            cli.http.server.SimpleHTTPRequestHandler,
            "send_head",
            autospec=True,
            side_effect=self._fallback_send_head,
        ):
            return self.send_head()

    def _fallback_send_head(self, _handler):
        self.fallback_called = True
        return "fallback-result"


class RangeHandlerTests(unittest.TestCase):
    def test_send_head_returns_206_and_expected_headers_for_explicit_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "video.mp4"
            payload = bytes(range(256)) * 4
            video_path.write_bytes(payload)

            handler = _HandlerHarness(video_path, "bytes=0-99")
            limited_file = handler.call_send_head()

            self.assertEqual(handler.responses, [206])
            self.assertEqual(
                handler.sent_headers["Content-Range"], f"bytes 0-99/{len(payload)}"
            )
            self.assertEqual(handler.sent_headers["Content-Length"], "100")
            self.assertEqual(handler.sent_headers["Accept-Ranges"], "bytes")
            self.assertTrue(handler.end_headers_called)
            self.assertFalse(handler.fallback_called)
            self.assertEqual(limited_file.read(), payload[:100])
            self.assertEqual(limited_file.read(), b"")
            limited_file.close()

    def test_send_head_supports_open_ended_ranges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "video.mp4"
            payload = b"0123456789"
            video_path.write_bytes(payload)

            handler = _HandlerHarness(video_path, "bytes=4-")
            limited_file = handler.call_send_head()

            self.assertEqual(handler.responses, [206])
            self.assertEqual(
                handler.sent_headers["Content-Range"], f"bytes 4-9/{len(payload)}"
            )
            self.assertEqual(handler.sent_headers["Content-Length"], "6")
            self.assertEqual(limited_file.read(), payload[4:])
            limited_file.close()

    def test_send_head_supports_suffix_ranges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "video.mp4"
            payload = b"abcdefghij"
            video_path.write_bytes(payload)

            handler = _HandlerHarness(video_path, "bytes=-4")
            limited_file = handler.call_send_head()

            # Suffix ranges are explicitly supported so browser seek behavior stays
            # consistent if a client asks for the tail of the asset.
            self.assertEqual(handler.responses, [206])
            self.assertEqual(
                handler.sent_headers["Content-Range"], f"bytes 6-9/{len(payload)}"
            )
            self.assertEqual(handler.sent_headers["Content-Length"], "4")
            self.assertEqual(limited_file.read(), b"ghij")
            limited_file.close()

    def test_send_head_falls_back_when_range_header_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "video.mp4"
            video_path.write_bytes(b"video-bytes")

            handler = _HandlerHarness(video_path, None)
            result = handler.call_send_head()

            self.assertEqual(result, "fallback-result")
            self.assertTrue(handler.fallback_called)
            self.assertEqual(handler.responses, [])
            self.assertEqual(handler.sent_headers, {})

    def test_send_head_falls_back_for_invalid_range_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "video.mp4"
            video_path.write_bytes(b"video-bytes")

            handler = _HandlerHarness(video_path, "bytes=8-3")
            result = handler.call_send_head()

            self.assertEqual(result, "fallback-result")
            self.assertTrue(handler.fallback_called)
            self.assertEqual(handler.responses, [])
            self.assertEqual(handler.sent_headers, {})


if __name__ == "__main__":
    unittest.main()
