"""Typer-based command line interface for the transcribe tool."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from .config import CONFIG_PATH, build_config_from_cli
from .diarization import DiarizationEngine
from .emotion import EmotionDetector
from .output import build_result, format_text_output, save_json, save_text
from .prefetch import Prefetcher
from .runtime import RuntimeContext
from .speaker_id import (
    apply_mapped_labels,
    build_speaker_embeddings,
    enroll_user,
    load_enrollment,
    load_enrollment_by_user,
    map_speakers_to_user,
)
from .transcription import TranscriptionEngine


app = typer.Typer(help="Transcribe audio with WhisperX and diarization")


@app.command()
def init(
    device: Optional[str] = typer.Option(None, help="Execution device (cpu, cuda, mps)"),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", help="Hugging Face token used to download diarization models"
    ),
    asr_model: Optional[str] = typer.Option(None, "--asr-model", help="WhisperX model to cache"),
    compute_type: Optional[str] = typer.Option(
        None, "--compute-type", help="Precision/quantization for WhisperX"
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        help="Language code for the alignment model (defaults to 'en' when unspecified)",
    ),
    diarization_model: Optional[str] = typer.Option(
        None,
        "--diarization-model",
        help="pyannote pipeline identifier to download",
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", help="Batch size to store for later use"
    ),
) -> None:
    """Download all required model weights ahead of time."""

    config = build_config_from_cli(
        device=device,
        huggingface_token=hf_token,
        asr_model=asr_model,
        compute_type=compute_type,
        language=language,
        diarization_model=diarization_model,
        batch_size=batch_size,
    )
    runtime = RuntimeContext.from_config(config)
    prefetcher = Prefetcher(
        config,
        alignment_language=config.model.language,
        runtime=runtime,
    )
    summary = prefetcher.prefetch()
    typer.echo(json.dumps(asdict(summary), indent=2))
    config.save()


@app.command()
def enroll(
    audio: Path = typer.Argument(..., exists=True, readable=True, help="Path to enrollment audio"),
    user_id: str = typer.Option(..., "--user-id", help="Identifier for the enrolled user"),
    device: Optional[str] = typer.Option(None, help="Execution device (cpu, cuda, mps)"),
) -> None:
    """Enroll a new user voice profile."""

    config = build_config_from_cli(device=device)
    runtime = RuntimeContext.from_config(config)
    record = enroll_user(audio, user_id=user_id, config=config, runtime=runtime)
    typer.echo(record.to_json())


@app.command()
def transcribe(
    audio: Path = typer.Argument(..., exists=True, readable=True, help="Audio file to transcribe"),
    user_id: Optional[str] = typer.Option(None, "--user-id", help="Enrolled user to match"),
    embedding: Optional[Path] = typer.Option(
        None,
        "--embedding",
        exists=True,
        readable=True,
        help="Path to a specific enrollment file",
    ),
    json_out: Optional[Path] = typer.Option(None, help="Path to write JSON output"),
    text_out: Optional[Path] = typer.Option(None, help="Path to write text transcript"),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        exists=False,
        writable=True,
        help="Directory where outputs will be written using the source filename",
    ),
    device: Optional[str] = typer.Option(None, help="Execution device (cpu, cuda, mps)"),
    hf_token: Optional[str] = typer.Option(
        None, help="Hugging Face token for diarization models"
    ),
) -> None:
    """Transcribe audio with diarization and optional speaker recognition."""

    config = build_config_from_cli(device=device, huggingface_token=hf_token)
    typer.echo(f"Using config at {CONFIG_PATH}")
    runtime = RuntimeContext.from_config(config)
    transcriber = TranscriptionEngine(config, runtime=runtime)
    diarizer = DiarizationEngine(config, runtime=runtime)
    emotion_detector = EmotionDetector(config, runtime=runtime)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        if json_out is None:
            json_out = output_dir / f"{audio.stem}.json"
        if text_out is None:
            text_out = output_dir / f"{audio.stem}.txt"

    transcription = transcriber.transcribe(audio)
    diar_segments = diarizer.diarize(audio)
    segments = DiarizationEngine.apply_speakers(transcription.segments, diar_segments)
    date_recorded = _infer_recorded_date(audio)
    date_transcribed = datetime.now(timezone.utc).isoformat()

    enrollment_record = None
    if user_id:
        enrollment_record = load_enrollment_by_user(user_id, config)
    elif embedding:
        enrollment_record = load_enrollment(embedding)

    speaker_embeddings = None
    if enrollment_record is not None:
        speaker_embeddings = build_speaker_embeddings(
            audio,
            segments,
            config,
            runtime=runtime,
        )
        if speaker_embeddings:
            map_speakers_to_user(
                speaker_embeddings,
                enrollment_record,
                thresholds=config,
            )
            apply_mapped_labels(segments, speaker_embeddings)

    emotion_detector.annotate_segments(audio, segments)

    result = build_result(
        audio_path=audio,
        language=transcription.language,
        segments=segments,
        speaker_embeddings=speaker_embeddings,
        date_recorded=date_recorded,
        date_transcribed=date_transcribed,
    )

    if json_out:
        save_json(result, json_out)
    if text_out:
        save_text(result, text_out)

    typer.echo(format_text_output(result))


if __name__ == "__main__":  # pragma: no cover
    app()


def _infer_recorded_date(audio_path: Path) -> str:
    """Return file modification time in ISO8601 (UTC) as a proxy for recording date."""

    try:
        timestamp = audio_path.stat().st_mtime
    except OSError:
        return datetime.now(timezone.utc).isoformat()
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
