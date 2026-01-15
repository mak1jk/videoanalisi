import json
from ..vlm_providers.base import VLMProvider

class TextSegmenter:
    def __init__(self, provider: VLMProvider):
        self.provider = provider

    def segment_transcript(self, transcript_segments: list) -> dict:
        """
        Takes raw Whisper segments and reorganizes them via LLM.
        """
        print("Segmenting transcript with LLM...")
        
        # Prepare data for LLM - simplify to save tokens if needed
        # We keep words for precise timestamp recovery if the LLM hallucinates timestamps,
        # but for now let's trust the LLM or give it the segment-level data.
        
        # Minimizing payload: Just start, end, text.
        simplified_segments = []
        for s in transcript_segments:
            simplified_segments.append({
                "start": s.get("start"),
                "end": s.get("end"),
                "text": s.get("text").strip()
            })
            
        json_payload = json.dumps(simplified_segments)
        
        # Call LLM
        response_text = self.provider.segment_text(json_payload)
        
        # Parse response
        try:
            # Clean potential markdown
            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            print(f"Failed to parse LLM response: {response_text[:100]}...")
            return {"sections": []}
