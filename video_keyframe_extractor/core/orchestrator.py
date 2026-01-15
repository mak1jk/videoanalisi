from typing import List, Dict
import os
from .audio_extractor import AudioExtractor
from .transcriber import Transcriber
from .frame_sampler import FrameSampler
from .scene_detector import SceneDetector
from .text_segmenter import TextSegmenter
from .candidate_selector import CandidateSelector
from .embeddings import EmbeddingsModel
from ..vlm_providers.base import VLMProvider
from ..config import Config

class FrameSelectorOrchestrator:
    def __init__(self, vlm_provider: VLMProvider):
        self.vlm_provider = vlm_provider
        self.audio_extractor = AudioExtractor()
        # Initialize transcriber lazily or here? Here is fine.
        self.transcriber = Transcriber() 
        self.frame_sampler = FrameSampler()
        self.scene_detector = SceneDetector()
        
        # Initialize Embeddings/Candidate Selector
        self.embeddings_model = EmbeddingsModel()
        self.candidate_selector = CandidateSelector(self.embeddings_model, similarity_threshold=Config.CLIP_SIMILARITY_THRESHOLD)
        
        self.text_segmenter = TextSegmenter(self.vlm_provider)

    def process_video(self, video_path: str) -> Dict:
        """
        Full pipeline execution.
        """
        results = {
            "video_path": video_path,
            "segments": []
        }
        
        # 1. Extract & Transcribe
        print("--- Step 1: Audio & Transcription ---")
        audio_path = self.audio_extractor.extract_audio(video_path)
        raw_transcript = self.transcriber.transcribe(audio_path)
        
        # 2. Segment Text
        print("--- Step 2: Semantic Segmentation ---")
        transcript_segments = raw_transcript["segments"]
        segmented_data = self.text_segmenter.segment_transcript(transcript_segments)
        sections = segmented_data.get("sections", [])
        
        if not sections:
            print("Warning: No sections found via LLM. Falling back to raw chunks?")
            # Fallback logic could go here
            return results

        # 3. Frame Sampling & Scene Detection
        print("--- Step 3: Frame Extraction & Analysis ---")
        # Sample uniformly
        uniform_frames = self.frame_sampler.sample_frames_uniform(video_path, interval_sec=1)
        
        # Detect scenes (optional optimization: add keyframes from scene changes)
        scenes = self.scene_detector.detect_scenes(video_path)
        # Add scene boundary frames to candidates? 
        # For now, let's rely on uniform sampling + CLIP filtering which is robust enough.
        
        # Filter duplicates globally once? Or per segment?
        # Globally filtering first is more efficient for embedding calculation.
        unique_frames = self.candidate_selector.select_unique_frames(uniform_frames)
        
        # 4. Match Segments to Frames
        print(f"--- Step 4: VLM Matching ({len(sections)} sections) ---")
        
        final_segments = []
        
        for section in sections:
            start = section.get("start_time", 0)
            end = section.get("end_time", 0)
            text = section.get("text", "")
            title = section.get("title", "Untitled")
            
            print(f"Processing section: '{title}' ({start:.1f}s - {end:.1f}s)")
            
            # Get candidates in this time range
            candidates = self.candidate_selector.get_candidates_for_segment(unique_frames, start, end)
            
            selected_frame = None
            
            if candidates:
                # Limit candidates for VLM (top 5-10 to save tokens/cost)
                # If we had a 'visual score' we would sort by that. 
                # For now, take up to 5 evenly spaced in the interval?
                # Or just the first 5 unique ones.
                shortlisted = candidates[:5] 
                image_paths = [c['path'] for c in shortlisted]
                
                print(f"  - Sending {len(shortlisted)} images to VLM...")
                idx = self.vlm_provider.analyze_image_text(text, image_paths)
                
                if idx is not None and idx != -1:
                    selected_frame = shortlisted[idx]
                    print(f"  -> Selected frame: {selected_frame['path']}")
                else:
                    print("  -> No frame selected by VLM.")
            else:
                print("  -> No candidate frames in this time range.")
            
            final_segments.append({
                "title": title,
                "start": start,
                "end": end,
                "text": text,
                "keyframe": selected_frame # {'path': ..., 'timestamp': ...} or None
            })
            
        results["segments"] = final_segments
        return results
