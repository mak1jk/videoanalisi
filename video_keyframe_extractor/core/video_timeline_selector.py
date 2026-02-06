import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TimelineEvent:
    timestamp: float
    frame_type: str
    slide_text: str
    description: str
    confidence: float


def _normalize_text(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if len(t) >= 3]
    return tokens


def _token_overlap_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ta = set(_normalize_text(a))
    tb = set(_normalize_text(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def pick_best_event_for_section(
    events: List[TimelineEvent], *, start: float, end: float, section_text: str
) -> Optional[TimelineEvent]:
    candidates = [e for e in events if start <= e.timestamp <= end]
    if not candidates:
        return None

    best: Optional[TimelineEvent] = None
    best_score = -1.0

    for e in candidates:
        score = float(e.confidence)
        if e.frame_type == "slide":
            score += 0.25
        elif e.frame_type == "demo":
            score += 0.10

        score += 0.75 * _token_overlap_score(section_text, e.slide_text)

        if score > best_score:
            best_score = score
            best = e

    return best


def parse_events(
    raw_events: List[Dict[str, Any]], *, global_offset: float
) -> List[TimelineEvent]:
    parsed: List[TimelineEvent] = []
    for item in raw_events or []:
        try:
            ts = float(item.get("timestamp_s")) + float(global_offset)
        except Exception:
            continue
        parsed.append(
            TimelineEvent(
                timestamp=ts,
                frame_type=str(item.get("frame_type") or "unknown"),
                slide_text=str(item.get("slide_text") or ""),
                description=str(item.get("description") or ""),
                confidence=float(item.get("confidence") or 0.0),
            )
        )

    return parsed
