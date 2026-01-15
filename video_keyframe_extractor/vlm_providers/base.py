from abc import ABC, abstractmethod
from typing import List, Optional

class VLMProvider(ABC):
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Returns a list of available model names."""
        pass

    @abstractmethod
    def analyze_image_text(self, text_segment: str, image_paths: List[str], prompt: str) -> Optional[int]:
        """
        Analyzes text and images to select the best matching image.
        Returns the index of the best image in image_paths, or None if no image is relevant.
        """
        pass

    @abstractmethod
    def segment_text(self, full_transcript: str) -> str:
        """
        Uses the LLM to segment the raw transcript into logical sections.
        Returns a JSON string structure.
        """
        pass
