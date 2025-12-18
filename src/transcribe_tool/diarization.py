"""Speaker diarization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from soundfile import LibsndfileError

from .config import AppConfig
from .audio_io import transcode_to_temp_wav
from .runtime import RuntimeContext
from .prefetch import (
    _allow_omegaconf_safe_globals,
    _diarization_access_error,
    _is_auth_error,
)
from .types import SegmentEntry, WordEntry


@dataclass(slots=True)
class DiarizedSegment:
    """Represents a diarized chunk with a speaker label."""

    start: float
    end: float
    speaker: str


class DiarizationEngine:
    """Wrapper around pyannote audio pipelines."""

    def __init__(self, config: AppConfig, *, runtime: RuntimeContext | None = None):
        self.config = config
        self.runtime = runtime or RuntimeContext.from_config(config)
        self._pipeline = None

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        from pyannote.audio import Pipeline  # Local import

        token = self.runtime.hf_token
        if not token:
            raise RuntimeError(
                "Hugging Face token is required for the community diarization pipeline."
            )

        _allow_omegaconf_safe_globals()
        try:
            pipeline = Pipeline.from_pretrained(
                self.config.model.diarization_model,
                use_auth_token=token,
            )
        except AttributeError as exc:
            raise _diarization_access_error(self.config.model.diarization_model) from exc
        except Exception as exc:  # pragma: no cover - defensive, surfacing HF auth errors
            if _is_auth_error(exc):
                raise _diarization_access_error(self.config.model.diarization_model) from exc
            raise
        if pipeline is None:
            raise _diarization_access_error(self.config.model.diarization_model)
        device = self.runtime.device
        if device != "cpu":
            pipeline.to(device)
        self._pipeline = pipeline

    def diarize(self, audio_path: Path) -> List[DiarizedSegment]:
        """Run diarization and return structured segments."""

        self._ensure_pipeline()
        assert self._pipeline is not None
        diarization = self._run_with_transcode_fallback(audio_path)
        segments: List[DiarizedSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                DiarizedSegment(start=turn.start, end=turn.end, speaker=str(speaker))
            )
        return segments

    def _run_with_transcode_fallback(self, audio_path: Path):
        """Call the pipeline, transcoding to WAV if the backend cannot read the input."""

        try:
            return self._pipeline(str(audio_path))
        except LibsndfileError:
            transcoded = transcode_to_temp_wav(audio_path)
            try:
                return self._pipeline(str(transcoded))
            finally:
                transcoded.unlink(missing_ok=True)

    @staticmethod
    def apply_speakers(
        segments: List[SegmentEntry],
        diarized_segments: List[DiarizedSegment],
    ) -> List[SegmentEntry]:
        """Assign speaker labels to each segment and word."""

        for segment in segments:
            speaker_counts: dict[str, float] = {}
            for word in segment.words:
                speaker = _select_speaker(word.start, word.end, diarized_segments)
                word.speaker = speaker
                if speaker:
                    speaker_counts[speaker] = speaker_counts.get(speaker, 0.0) + (
                        word.end - word.start
                    )
            if speaker_counts:
                segment.speaker = max(speaker_counts, key=speaker_counts.get)
        return segments


def _select_speaker(
    start: float,
    end: float,
    diarized_segments: List[DiarizedSegment],
) -> Optional[str]:
    best_speaker: Optional[str] = None
    best_overlap = 0.0
    for diar in diarized_segments:
        overlap = _overlap(start, end, diar.start, diar.end)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = diar.speaker
    return best_speaker


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    left = max(a_start, b_start)
    right = min(a_end, b_end)
    duration = right - left
    return duration if duration > 0 else 0.0
