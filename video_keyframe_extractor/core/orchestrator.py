from typing import Dict
import os
from .audio_extractor import AudioExtractor
from .transcriber import Transcriber
from .groq_transcriber import GroqTranscriber
from .frame_sampler import FrameSampler
from .scene_detector import SceneDetector
from .text_segmenter import TextSegmenter
from .candidate_selector import CandidateSelector
from .embeddings import EmbeddingsModel
from .vertex_video_preparer import prepare_video_for_vertex
from .video_timeline_selector import parse_events, pick_best_event_for_section
from ..vlm_providers.base import VLMProvider
from ..config import Config


def _build_transcriber(prefer: str = None):
    """
    Build a transcriber based on preference and available resources.

    prefer: "groq" | "local" | None (auto-detect)
    """
    backend = prefer or Config.TRANSCRIBER or ("groq" if Config.GROQ_API_KEY else "local")

    if backend == "groq":
        if not Config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Cannot use '--transcriber groq'.\n"
                "Set GROQ_API_KEY in your .env file or switch to '--transcriber local'."
            )
        print("Transcriber: Groq Whisper API")
        return GroqTranscriber(api_key=Config.GROQ_API_KEY)

    if backend == "local":
        try:
            from .local_whisper_provider import LocalWhisperTranscriber  # noqa
            print("Transcriber: faster-whisper (local, no API key required)")
            return LocalWhisperTranscriber()
        except ImportError:
            raise ImportError(
                "faster-whisper is not installed.\n"
                "Install it with: pip install faster-whisper\n"
                "Or use '--transcriber groq' with a valid GROQ_API_KEY."
            )

    # Fallback: openai-whisper (legacy)
    print("Transcriber: openai-whisper (legacy local)")
    return Transcriber()


class FrameSelectorOrchestrator:
    def __init__(self, vlm_provider: VLMProvider, fast_mode: bool = False):
        self.vlm_provider = vlm_provider
        self.fast_mode = fast_mode
        self.frame_sampler = FrameSampler()
        self.audio_extractor = AudioExtractor()

        # Groq transcriber available in both modes (for fast path)
        self.groq_transcriber = None
        if Config.GROQ_API_KEY and (not Config.TRANSCRIBER or Config.TRANSCRIBER == "groq"):
            self.groq_transcriber = GroqTranscriber(api_key=Config.GROQ_API_KEY)

        if not fast_mode:
            self.transcriber = _build_transcriber()
            self.scene_detector = SceneDetector()
            self.embeddings_model = EmbeddingsModel()
            self.candidate_selector = CandidateSelector(
                self.embeddings_model, similarity_threshold=Config.CLIP_SIMILARITY_THRESHOLD
            )
            self.text_segmenter = TextSegmenter(self.vlm_provider)

    def process_video_fast(self, video_path: str) -> Dict:
        """
        Fast pipeline: upload video to Gemini Files API, get sections + keyframes in one call.
        Also runs Groq Whisper for verbatim transcript per section.
        """
        results = {"video_path": video_path, "segments": []}

        print("--- Fast Mode: Gemini Files API ---")
        gemini_result = self.vlm_provider.upload_and_analyze_video(video_path)
        sections = gemini_result.get("sections", [])
        results["cost_info"] = gemini_result.get("cost_info", {})

        if not sections:
            print("Warning: No sections returned from Gemini.")
            return results

        # Groq Whisper transcription for verbatim text
        whisper_segments = []
        full_transcript = ""
        if self.groq_transcriber:
            print("--- Trascrizione audio con Groq Whisper ---")
            audio_path = self.audio_extractor.extract_audio(video_path)
            raw_transcript = self.groq_transcriber.transcribe(audio_path, language="it")
            whisper_segments = [
                s for s in raw_transcript.get("segments", [])
                if s.get("text", "").strip()
            ]
            full_transcript = raw_transcript.get("text", "").strip()
            if whisper_segments:
                print(f"  {len(whisper_segments)} segmenti trascritti con timestamp.")
            elif full_transcript:
                print(f"  Trascrizione completa ottenuta ({len(full_transcript)} caratteri, senza timestamp segmento).")

        print(f"--- Extracting {len(sections)} keyframes ---")

        # Compute total video duration for proportional text split
        total_duration = max(
            (float(s.get("end_time", 0)) for s in sections), default=1.0
        )

        final_segments = []
        for i, section in enumerate(sections):
            keyframe_time = float(section.get("keyframe_time") or section.get("start_time", 0))
            start = float(section.get("start_time", 0))
            end = float(section.get("end_time", 0))

            # Filter whisper segments overlapping this section
            transcript_text = ""
            if whisper_segments:
                matching = [
                    ws["text"].strip()
                    for ws in whisper_segments
                    if ws.get("end", 0) > start and ws.get("start", 0) < end
                ]
                transcript_text = " ".join(matching)
            elif full_transcript and total_duration > 0:
                # Fallback: proportional split of full text by section duration
                chars_per_sec = len(full_transcript) / total_duration
                char_start = int(start * chars_per_sec)
                char_end = int(end * chars_per_sec)
                transcript_text = full_transcript[char_start:char_end].strip()

            print(f"  [{i+1}/{len(sections)}] '{section.get('title', '')}' @ {keyframe_time:.1f}s")
            frame_path = self.frame_sampler.get_frame_at_timestamp(video_path, keyframe_time)
            final_segments.append({
                "title": section.get("title", "Untitled"),
                "start": start,
                "end": end,
                "text": section.get("text", ""),
                "transcript": transcript_text,
                "tasks": section.get("tasks") or [],
                "keyframe": {"path": frame_path, "timestamp": keyframe_time} if frame_path else None,
            })

        results["segments"] = final_segments
        return results

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
