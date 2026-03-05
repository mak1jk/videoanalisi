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
        Processes one frame at a time to avoid OOM on large videos.
        frames_data: List of dicts {'path': str, 'timestamp': float, ...}
        """
        if not frames_data:
            return []

        total = len(frames_data)
        print(f"Filtering {total} frames (threshold: {self.similarity_threshold})...")

        unique_frames = []
        last_kept_emb = None

        for i, frame in enumerate(frames_data):
            if i % 200 == 0:
                print(f"  CLIP dedup: {i}/{total} processed, {len(unique_frames)} kept...")
            current_emb = self.embeddings_model.encode_image(frame['path'])
            if last_kept_emb is None:
                unique_frames.append(frame)
                last_kept_emb = current_emb
                continue
            sim = self.embeddings_model.compute_similarity(last_kept_emb, current_emb).item()
            if sim < self.similarity_threshold:
                unique_frames.append(frame)
                last_kept_emb = current_emb

        print(f"Retained {len(unique_frames)} unique frames.")
        return unique_frames

    def get_candidates_for_segment(self, all_frames: List[Dict], start: float, end: float) -> List[Dict]:
        """
        Returns frames that fall within the specific time segment.
        """
        candidates = [f for f in all_frames if start <= f['timestamp'] <= end]
        return candidates
