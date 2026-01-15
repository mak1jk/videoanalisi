from typing import List, Dict
import numpy as np
from .embeddings import EmbeddingsModel
from .frame_sampler import FrameSampler

class CandidateSelector:
    def __init__(self, embeddings_model: EmbeddingsModel, similarity_threshold: float = 0.90):
        self.embeddings_model = embeddings_model
        self.similarity_threshold = similarity_threshold

    def select_unique_frames(self, frames_data: List[Dict]) -> List[Dict]:
        """
        Filters frames that are visually too similar using CLIP embeddings.
        frames_data: List of dicts {'path': str, 'timestamp': float, ...}
        """
        if not frames_data:
            return []

        print(f"Filtering {len(frames_data)} frames (threshold: {self.similarity_threshold})...")
        
        # 1. Compute embeddings for all frames
        frame_paths = [f['path'] for f in frames_data]
        embeddings = self.embeddings_model.encode_images(frame_paths)
        
        unique_frames = []
        if len(frames_data) > 0:
            # Always keep the first frame
            unique_frames.append(frames_data[0])
            last_kept_emb = embeddings[0]

            for i in range(1, len(frames_data)):
                current_emb = embeddings[i]
                sim = self.embeddings_model.compute_similarity(last_kept_emb, current_emb).item()
                
                if sim < self.similarity_threshold:
                    unique_frames.append(frames_data[i])
                    last_kept_emb = current_emb
        
        print(f"Retained {len(unique_frames)} unique frames.")
        return unique_frames

    def get_candidates_for_segment(self, all_frames: List[Dict], start: float, end: float) -> List[Dict]:
        """
        Returns frames that fall within the specific time segment.
        """
        candidates = [f for f in all_frames if start <= f['timestamp'] <= end]
        return candidates
