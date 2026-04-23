from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DocumentContent:
    source_path: Path
    file_type: str
    text: str
    image_summaries: list[str] = field(default_factory=list)


@dataclass
class TopicBundle:
    topic: str
    evidence: str
    notes: str
    mcq: list[dict]
    short_answer: list[dict]


@dataclass
class AnalysisResult:
    topics: list[TopicBundle]
    raw_documents: list[DocumentContent]
    warnings: list[str] = field(default_factory=list)
