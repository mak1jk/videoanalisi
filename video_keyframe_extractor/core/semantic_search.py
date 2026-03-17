from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .embeddings import EmbeddingsModel


def _section_text(section: Dict) -> str:
    parts = [
        section.get("title", ""),
        section.get("transcript", ""),
        section.get("text", ""),
    ]
    return "\n".join(
        part.strip() for part in parts if isinstance(part, str) and part.strip()
    )


def load_sections_from_json(json_path: str) -> List[Dict]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    sections = data.get("segments", [])
    if not sections:
        raise ValueError(f"No segments found in {json_path}")
    return sections


def semantic_search_sections(json_path: str, query: str, top_k: int = 3) -> List[Dict]:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    sections = load_sections_from_json(json_path)
    section_payload = []
    for idx, section in enumerate(sections):
        text = _section_text(section)
        if not text:
            continue
        section_payload.append({"index": idx, "section": section, "text": text})

    if not section_payload:
        raise ValueError("No searchable text found in the provided JSON")

    embeddings = EmbeddingsModel()
    query_embedding = embeddings.encode_text(query.strip())
    section_embeddings = [
        embeddings.encode_text(item["text"]) for item in section_payload
    ]

    results = []
    for item, section_embedding in zip(section_payload, section_embeddings):
        score = float(
            embeddings.compute_similarity(query_embedding, section_embedding).item()
        )
        section = item["section"]
        results.append(
            {
                "index": item["index"],
                "title": section.get("title", "Untitled"),
                "start": float(section.get("start", 0) or 0),
                "end": float(section.get("end", 0) or 0),
                "score": score,
                "transcript": section.get("transcript") or section.get("text") or "",
                "text": section.get("text", ""),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]
