"""WhisperX transcription helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import AppConfig
from .prefetch import _allow_omegaconf_safe_globals
from .runtime import RuntimeContext
from .types import SegmentEntry, WordEntry


@dataclass(slots=True)
class TranscriptionOutput:
    """Structured result of a transcription pass."""

    segments: List[SegmentEntry]
    language: Optional[str]
    raw_response: Dict[str, Any]


class TranscriptionEngine:
    """Thin wrapper around WhisperX APIs."""

    def __init__(self, config: AppConfig, *, runtime: RuntimeContext | None = None):
        self.config = config
        self.runtime = runtime or RuntimeContext.from_config(config)
        self._model = None
        self._align_model = None
        self._align_metadata = None
        self._language = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import whisperx  # Local import to avoid heavy dependency at import time.

        device = self.runtime.device
        _allow_omegaconf_safe_globals()
        self._model = whisperx.load_model(
            self.config.model.asr_model,
            device=device,
            compute_type=self.config.model.compute_type,
        )

    def _ensure_alignment(self, language: Optional[str]) -> None:
        if self._align_model is not None:
            return
        import whisperx

        device = self.runtime.device
        _allow_omegaconf_safe_globals()
        align_model, metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
        )
        self._align_model = align_model
        self._align_metadata = metadata

    def transcribe(self, audio_path: Path) -> TranscriptionOutput:
        """Run ASR + alignment and return structured segments."""

        import whisperx

        self._ensure_model()
        assert self._model is not None
        device = self.runtime.device
        result = self._model.transcribe(
            str(audio_path),
            batch_size=self.config.model.batch_size,
            language=self.config.model.language,
        )
        segments = result.get("segments", [])
        language = result.get("language") or self.config.model.language
        self._language = language

        self._ensure_alignment(language)
        aligned = whisperx.align(
            segments,
            self._align_model,
            self._align_metadata,
            str(audio_path),
            device=device,
        )

        structured_segments = self._build_segments(aligned["segments"])
        return TranscriptionOutput(
            segments=structured_segments,
            language=language,
            raw_response=aligned,
        )

    def _build_segments(self, segments: List[Dict[str, Any]]) -> List[SegmentEntry]:
        structured: List[SegmentEntry] = []
        for idx, segment in enumerate(segments):
            words = [
                WordEntry(
                    start=word.get("start", 0.0),
                    end=word.get("end", 0.0),
                    text=word.get("word", word.get("text", "")),
                )
                for word in segment.get("words", [])
            ]
            structured.append(
                SegmentEntry(
                    id=idx,
                    start=segment.get("start", 0.0),
                    end=segment.get("end", 0.0),
                    text=segment.get("text", ""),
                    words=words,
                )
            )
        return structured
