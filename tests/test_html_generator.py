import tempfile
import unittest
from pathlib import Path

from video_keyframe_extractor.output.html_generator import HTMLGenerator


class HTMLGeneratorTests(unittest.TestCase):
    def test_generate_document_escapes_html_in_text(self):
        project_root = Path(__file__).resolve().parents[1]
        template_dir = project_root / "video_keyframe_extractor" / "output" / "templates"

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = HTMLGenerator(
                template_dir=str(template_dir),
                output_dir=tmpdir,
            )
            data = {
                "video_path": "video.mp4",
                "segments": [
                    {
                        "title": "Intro",
                        "start": 0.0,
                        "end": 10.0,
                        "text": '<script>alert("x")</script>',
                        "keyframe": None,
                    }
                ],
            }

            output = generator.generate_document(data, "project")
            html = Path(output).read_text(encoding="utf-8")

            self.assertIn("&lt;script&gt;alert", html)
            self.assertNotIn('<script>alert("x")</script>', html)


if __name__ == "__main__":
    unittest.main()
