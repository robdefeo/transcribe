"""Voice enrollment and recognition helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from . import audio_io
from .config import AppConfig
from .runtime import RuntimeContext
from .types import SegmentEntry, SpeakerEmbedding

try:  # Heavy dependency, import lazily in functions when possible.
    from resemblyzer import VoiceEncoder, preprocess_wav
except ImportError:  # pragma: no cover - optional during lightweight tests.
    VoiceEncoder = None  # type: ignore[assignment]
    preprocess_wav = None  # type: ignore[assignment]


@dataclass(slots=True)
class EnrollmentRecord:
    """Metadata saved for each enrolled user."""

    user_id: str
    embedding: List[float]
    created_at: str
    sample_rate: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "user_id": self.user_id,
                "embedding": self.embedding,
                "created_at": self.created_at,
                "sample_rate": self.sample_rate,
            },
            indent=2,
        )

    @staticmethod
    def from_json(data: str) -> "EnrollmentRecord":
        payload = json.loads(data)
        return EnrollmentRecord(
            user_id=payload["user_id"],
            embedding=list(payload["embedding"]),
            created_at=payload["created_at"],
            sample_rate=payload.get("sample_rate", 16000),
        )


def enroll_user(
    audio_path: Path,
    user_id: str,
    config: AppConfig,
    *,
    runtime: RuntimeContext | None = None,
) -> EnrollmentRecord:
    """Create and persist an embedding for the given user."""

    if preprocess_wav is None or VoiceEncoder is None:
        raise RuntimeError("Resemblyzer is required for enrollment but is missing.")

    wav = preprocess_wav(str(audio_path))
    runtime = runtime or RuntimeContext.from_config(config)
    device = runtime.device
    encoder = VoiceEncoder(device=device)
    embedding = encoder.embed_utterance(wav).tolist()
    record = EnrollmentRecord(
        user_id=user_id,
        embedding=embedding,
        created_at=datetime.now(timezone.utc).isoformat(),
        sample_rate=16000,
    )
    path = config.enrollment_path_for(user_id)
    path.write_text(record.to_json())
    return record


def load_enrollment(path: Path) -> EnrollmentRecord:
    """Load an enrollment record from disk."""

    return EnrollmentRecord.from_json(path.read_text())


def load_enrollment_by_user(user_id: str, config: AppConfig) -> EnrollmentRecord:
    """Resolve the default path for a user and load the embedding."""

    path = config.enrollment_path_for(user_id)
    if not path.exists():
        raise FileNotFoundError(f"No enrollment found for user '{user_id}'.")
    return load_enrollment(path)


def build_speaker_embeddings(
    audio_path: Path,
    segments: List[SegmentEntry],
    config: AppConfig,
    *,
    runtime: RuntimeContext | None = None,
) -> Dict[str, SpeakerEmbedding]:
    """Aggregate embeddings for each diarized speaker."""

    if VoiceEncoder is None:
        raise RuntimeError("Resemblyzer is required for speaker mapping but is missing.")

    waveform, sample_rate = audio_io.load_audio(audio_path, target_sample_rate=16000)
    runtime = runtime or RuntimeContext.from_config(config)
    device = runtime.device
    encoder = VoiceEncoder(device=device)

    speaker_vectors: Dict[str, List[np.ndarray]] = {}
    for segment in segments:
        if not segment.speaker:
            continue
        clip = audio_io.slice_audio(waveform, sample_rate, segment.start, segment.end)
        if clip.size == 0:
            continue
        emb = encoder.embed_utterance(clip).astype(np.float32)
        speaker_vectors.setdefault(segment.speaker, []).append(emb)

    embeddings: Dict[str, SpeakerEmbedding] = {}
    for speaker_id, vectors in speaker_vectors.items():
        if not vectors:
            continue
        stacked = np.stack(vectors, axis=0)
        averaged = stacked.mean(axis=0)
        embeddings[speaker_id] = SpeakerEmbedding(
            speaker_id=speaker_id,
            vector=averaged.tolist(),
        )
    return embeddings


def map_speakers_to_user(
    speaker_embeddings: Dict[str, SpeakerEmbedding],
    enrollment: EnrollmentRecord,
    thresholds: AppConfig | None = None,
    *,
    match_threshold: Optional[float] = None,
    maybe_threshold: Optional[float] = None,
) -> Dict[str, SpeakerEmbedding]:
    """Annotate speakers with similarity scores to the enrolled user."""

    if thresholds is not None:
        match_threshold = match_threshold or thresholds.thresholds.match_threshold
        maybe_threshold = maybe_threshold or thresholds.thresholds.maybe_threshold
    else:
        match_threshold = match_threshold or 0.75
        maybe_threshold = maybe_threshold or 0.65

    ref = np.array(enrollment.embedding, dtype=np.float32)
    ref_norm = np.linalg.norm(ref)
    if ref_norm == 0:
        raise ValueError("Enrollment embedding has zero norm.")

    for speaker_id, embedding in speaker_embeddings.items():
        vec = np.array(embedding.vector, dtype=np.float32)
        denom = (np.linalg.norm(vec) * ref_norm)
        similarity = float(vec.dot(ref) / denom) if denom else 0.0
        embedding.similarity_to_user = similarity
        if similarity >= match_threshold:
            embedding.mapped_label = f"USER:{enrollment.user_id}"
        elif similarity >= maybe_threshold:
            embedding.mapped_label = f"MAYBE:{enrollment.user_id}"
        else:
            embedding.mapped_label = None
    return speaker_embeddings


def apply_mapped_labels(
    segments: Iterable[SegmentEntry],
    speaker_embeddings: Dict[str, SpeakerEmbedding],
) -> None:
    """Update segment and word labels with mapped speaker IDs."""

    mapping = {
        speaker_id: emb.mapped_label or speaker_id
        for speaker_id, emb in speaker_embeddings.items()
    }
    for segment in segments:
        if segment.speaker and segment.speaker in mapping:
            segment.mapped_speaker = mapping[segment.speaker]
        for word in segment.words:
            if word.speaker and word.speaker in mapping:
                word.mapped_speaker = mapping[word.speaker]
