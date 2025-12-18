"""Emotion detection helpers using a Hugging Face audio classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch
import torchaudio
from torchaudio.functional import resample
from soundfile import LibsndfileError

from .config import AppConfig
from .audio_io import transcode_to_temp_wav
from .constants import EMOTION_MODEL_ID
from .runtime import RuntimeContext
from .types import SegmentEntry


class EmotionDetector:
    """Annotates segments with emotion labels."""

    MIN_DURATION_SECONDS = 0.25

    def __init__(
        self,
        config: AppConfig,
        top_k: int = 5,
        *,
        runtime: RuntimeContext | None = None,
    ):
        self.config = config
        self.top_k = top_k
        self._classifier = None
        self._target_rate: Optional[int] = None
        self.runtime = runtime or RuntimeContext.from_config(config)

    def annotate_segments(self, audio_path: Path, segments: Iterable[SegmentEntry]) -> None:
        """Update each segment with an estimated emotion."""

        segments = list(segments)
        if not segments:
            return
        classifier = self._ensure_classifier()
        temp_wav: Optional[Path] = None
        try:
            try:
                waveform, sample_rate = torchaudio.load(str(audio_path))
            except LibsndfileError:
                temp_wav = transcode_to_temp_wav(
                    audio_path,
                    sample_rate=self._target_rate,
                    channels=1,
                )
                waveform, sample_rate = torchaudio.load(str(temp_wav))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform, sample_rate = self._maybe_resample(waveform, sample_rate)
            total_samples = waveform.shape[-1]

            for segment in segments:
                start_sample = max(0, int(segment.start * sample_rate))
                end_sample = min(total_samples, int(segment.end * sample_rate))
                if end_sample <= start_sample:
                    continue
                if (end_sample - start_sample) < int(self.MIN_DURATION_SECONDS * sample_rate):
                    continue
                chunk = waveform[:, start_sample:end_sample]
                array = chunk.mean(dim=0).cpu().numpy()
                predictions = classifier(
                    {"array": array, "sampling_rate": sample_rate},
                    top_k=self.top_k,
                )
                if not predictions:
                    continue
                if isinstance(predictions, dict):
                    predictions = [predictions]
                top_prediction = predictions[0]
                segment.emotion = top_prediction.get("label")
                score = top_prediction.get("score")
                if score is not None:
                    segment.emotion_confidence = float(score)
                segment.emotion_scores = {
                    pred.get("label"): float(pred.get("score") or 0.0)
                    for pred in predictions
                    if pred.get("label") is not None
                }
        finally:
            if temp_wav:
                temp_wav.unlink(missing_ok=True)

    def _maybe_resample(self, waveform: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
        target_rate = self._target_rate or sample_rate
        if target_rate and sample_rate != target_rate:
            waveform = resample(waveform, sample_rate, target_rate)
            sample_rate = target_rate
        return waveform, sample_rate

    def _ensure_classifier(self):
        if self._classifier is not None:
            return self._classifier
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - surface friendly error
            raise RuntimeError(
                "transformers is required for emotion detection. Install it via `uv pip install transformers`."
            ) from exc

        pipeline_kwargs = self.runtime.transformer_pipeline_kwargs()
        self._classifier = pipeline(
            task="audio-classification",
            model=EMOTION_MODEL_ID,
            **pipeline_kwargs,
        )
        extractor = getattr(self._classifier, "feature_extractor", None)
        sampling_rate = getattr(extractor, "sampling_rate", None)
        if sampling_rate:
            self._target_rate = int(sampling_rate)
        return self._classifier
