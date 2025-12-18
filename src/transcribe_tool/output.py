"""Output helpers for transcripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from pydantic import BaseModel, Field

from .types import SegmentEntry, SpeakerEmbedding
from .utils.timecode import format_timestamp


class SpeakerInfo(BaseModel):
    speaker_id: str
    mapped_label: Optional[str] = None
    similarity_to_user: Optional[float] = Field(default=None, ge=-1.0, le=1.0)


class WordModel(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    mapped_speaker: Optional[str] = None


class SegmentModel(BaseModel):
    id: int
    start: float
    end: float
    speaker: Optional[str] = None
    mapped_speaker: Optional[str] = None
    emotion: Optional[str] = None
    emotion_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    emotion_scores: Optional[dict[str, float]] = None
    text: str
    words: List[WordModel]


class TranscriptionResult(BaseModel):
    audio_path: Path
    language: Optional[str]
    date_recorded: Optional[str] = None
    date_transcribed: Optional[str] = None
    speakers: List[SpeakerInfo]
    segments: List[SegmentModel]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)


def build_result(
    audio_path: Path,
    language: Optional[str],
    segments: Iterable[SegmentEntry],
    speaker_embeddings: Optional[dict[str, SpeakerEmbedding]] = None,
    date_recorded: Optional[str] = None,
    date_transcribed: Optional[str] = None,
) -> TranscriptionResult:
    speakers = []
    if speaker_embeddings:
        for emb in speaker_embeddings.values():
            speakers.append(
                SpeakerInfo(
                    speaker_id=emb.speaker_id,
                    mapped_label=emb.mapped_label,
                    similarity_to_user=emb.similarity_to_user,
                )
            )

    segment_models: List[SegmentModel] = []
    for segment in segments:
        words = [
            WordModel(
                start=word.start,
                end=word.end,
                text=word.text,
                speaker=word.speaker,
                mapped_speaker=word.mapped_speaker,
            )
            for word in segment.words
        ]
        segment_models.append(
            SegmentModel(
                id=segment.id,
                start=segment.start,
                end=segment.end,
                speaker=segment.speaker,
                mapped_speaker=segment.mapped_speaker,
                emotion=segment.emotion,
                emotion_confidence=segment.emotion_confidence,
                emotion_scores=segment.emotion_scores,
                text=segment.text,
                words=words,
            )
        )

    return TranscriptionResult(
        audio_path=audio_path,
        language=language,
        date_recorded=date_recorded,
        date_transcribed=date_transcribed,
        speakers=speakers,
        segments=segment_models,
    )


def save_json(result: TranscriptionResult, path: Path) -> None:
    path.write_text(result.to_json())


def format_text_output(result: TranscriptionResult) -> str:
    lines = []
    for segment in result.segments:
        label = segment.mapped_speaker or segment.speaker or "SPEAKER"
        timestamp = f"[{format_timestamp(segment.start)} - {format_timestamp(segment.end)}]"
        emotion_suffix = f" ({segment.emotion})" if segment.emotion else ""
        lines.append(f"{timestamp} {label}{emotion_suffix}: {segment.text}")
    return "\n".join(lines)


def save_text(result: TranscriptionResult, path: Path) -> None:
    path.write_text(format_text_output(result))
