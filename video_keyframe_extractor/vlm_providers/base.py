from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VLMProvider(ABC):
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Returns a list of available model names."""

    @abstractmethod
    def analyze_image_text(
        self, text_segment: str, image_paths: List[str], prompt: Optional[str] = None
    ) -> Optional[int]:
        """Selects the best matching image index for a text segment."""

    @abstractmethod
    def segment_text(self, full_transcript: str) -> str:
        """Segments a transcript into logical sections and returns JSON (string)."""

    def supports_video_timeline(self) -> bool:
        return False

    def analyze_video_timeline(
        self,
        video_uri: str,
        *,
        max_events: int,
        mime_type: str = "video/mp4",
        prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "This provider does not support video timeline analysis."
        )
