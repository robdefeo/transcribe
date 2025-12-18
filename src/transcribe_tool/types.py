"""Shared data structures used across modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class WordEntry:
    """Represents a single word with precise timestamps."""

    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    mapped_speaker: Optional[str] = None


@dataclass(slots=True)
class SegmentEntry:
    """High-level utterance produced by WhisperX."""

    id: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    mapped_speaker: Optional[str] = None
    emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None
    emotion_scores: Optional[dict[str, float]] = None
    words: list[WordEntry] = field(default_factory=list)


@dataclass(slots=True)
class SpeakerEmbedding:
    """Aggregate embedding for a diarized speaker."""

    speaker_id: str
    vector: list[float]
    similarity_to_user: Optional[float] = None
    mapped_label: Optional[str] = None


@dataclass(slots=True)
class AudioDocument:
    """Paths and metadata tracked across the pipeline."""

    path: Path
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
