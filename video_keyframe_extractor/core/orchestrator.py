from typing import Dict
import os
from .audio_extractor import AudioExtractor
from .transcriber import Transcriber
from .frame_sampler import FrameSampler
from .scene_detector import SceneDetector
from .text_segmenter import TextSegmenter
from .candidate_selector import CandidateSelector
from .embeddings import EmbeddingsModel
from .vertex_video_preparer import prepare_video_for_vertex
from .video_timeline_selector import parse_events, pick_best_event_for_section
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
        self.candidate_selector = CandidateSelector(
            self.embeddings_model, similarity_threshold=Config.CLIP_SIMILARITY_THRESHOLD
        )

        self.text_segmenter = TextSegmenter(self.vlm_provider)

    def process_video(self, video_path: str) -> Dict:
        """
        Full pipeline execution.
        """
        results = {"video_path": video_path, "segments": []}

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

        # 3. Keyframe selection
        if (
            Config.USE_VERTEX_AI
            and Config.VIDEO_NATIVE_ENABLED
            and self.vlm_provider.supports_video_timeline()
        ):
            print("--- Step 3: Video-Native Timeline Analysis (Vertex AI) ---")

            prepared_chunks = prepare_video_for_vertex(video_path)
            all_events = []

            for chunk in prepared_chunks:
                print(
                    f"Analyzing chunk {chunk.start_seconds:.0f}s-{chunk.end_seconds:.0f}s via {chunk.gcs_uri}"
                )
                raw_events = self.vlm_provider.analyze_video_timeline(
                    chunk.gcs_uri,
                    max_events=Config.VIDEO_TIMELINE_MAX_EVENTS,
                    mime_type="video/mp4",
                )
                all_events.extend(
                    parse_events(raw_events, global_offset=chunk.start_seconds)
                )

            print(f"--- Step 4: Selecting Keyframes ({len(sections)} sections) ---")

            final_segments = []
            for section in sections:
                start = float(section.get("start_time", 0))
                end = float(section.get("end_time", 0))
                text = section.get("text", "")
                title = section.get("title", "Untitled")

                print(f"Processing section: '{title}' ({start:.1f}s - {end:.1f}s)")

                best_event = pick_best_event_for_section(
                    all_events, start=start, end=end, section_text=text
                )

                selected_frame = None
                if best_event is not None:
                    frame_path = self.frame_sampler.get_frame_at_timestamp(
                        video_path, best_event.timestamp
                    )
                    selected_frame = {
                        "path": frame_path,
                        "timestamp": best_event.timestamp,
                        "frame_type": best_event.frame_type,
                        "slide_text": best_event.slide_text,
                        "description": best_event.description,
                        "confidence": best_event.confidence,
                    }
                else:
                    # Fallback: sample the middle of the segment
                    fallback_ts = start + max(0.0, (end - start) / 2.0)
                    try:
                        frame_path = self.frame_sampler.get_frame_at_timestamp(
                            video_path, fallback_ts
                        )
                        selected_frame = {"path": frame_path, "timestamp": fallback_ts}
                    except Exception:
                        selected_frame = None

                final_segments.append(
                    {
                        "title": title,
                        "start": start,
                        "end": end,
                        "text": text,
                        "keyframe": selected_frame,
                    }
                )

            results["segments"] = final_segments
            return results

        # 3. Frame Sampling & Scene Detection (legacy fallback)
        print("--- Step 3: Frame Extraction & Analysis ---")
        uniform_frames = self.frame_sampler.sample_frames_uniform(
            video_path, interval_sec=1
        )

        # Detect scenes (optional optimization)
        _ = self.scene_detector.detect_scenes(video_path)

        # Filter duplicates globally once
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

            candidates = self.candidate_selector.get_candidates_for_segment(
                unique_frames, start, end
            )
            selected_frame = None

            if candidates:
                shortlisted = candidates[:5]
                image_paths = [c["path"] for c in shortlisted]

                print(f"  - Sending {len(shortlisted)} images to VLM...")
                idx = self.vlm_provider.analyze_image_text(text, image_paths)

                if idx is not None and idx != -1:
                    selected_frame = shortlisted[idx]
                    print(f"  -> Selected frame: {selected_frame['path']}")
                else:
                    print("  -> No frame selected by VLM.")
            else:
                print("  -> No candidate frames in this time range.")

            final_segments.append(
                {
                    "title": title,
                    "start": start,
                    "end": end,
                    "text": text,
                    "keyframe": selected_frame,
                }
            )

        results["segments"] = final_segments
        return results
