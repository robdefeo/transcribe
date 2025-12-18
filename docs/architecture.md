# Transcribe Tool Architecture

## Overview

A Python CLI package that:
- Enrolls a user's voice with Resemblyzer.
- Transcribes audio via WhisperX with word-level timestamps.
- Performs diarization with pyannote's `speaker-diarization-community-1` pipeline.
- Maps diarized speakers to the enrolled user when confidence is high.
- Outputs structured JSON plus formatted text with timestamps.

## Key Components

```
transcribe/
├─ pyproject.toml
├─ requirements.txt
├─ README.md
├─ src/
│  └─ transcribe_tool/
│     ├─ __init__.py
│     ├─ cli.py
│     ├─ config.py
│     ├─ audio_io.py
│     ├─ transcription.py
│     ├─ diarization.py
│     ├─ speaker_id.py
│     ├─ output.py
│     └─ utils/
│        ├─ __init__.py
│        └─ timecode.py
└─ tests/
   └─ test_config.py (placeholder until real samples exist)
```

### Modules
- `cli.py`: Typer-based entry points for `enroll` and `transcribe` commands.
- `config.py`: Global defaults (model names, devices, paths, thresholds, HF token resolution).
- `audio_io.py`: Audio loading, normalization, slicing helpers compatible with WhisperX and Resemblyzer.
- `transcription.py`: Wraps WhisperX ASR + alignment with device-aware loading and batching controls.
- `diarization.py`: Interfaces with pyannote pipeline (via WhisperX helper or direct) and merges speaker labels onto aligned words.
- `speaker_id.py`: Resemblyzer enrollment, embedding storage, similarity scoring, and speaker-label mapping.
- `output.py`: JSON schema creation via Pydantic models and human-readable text formatter.
- `utils/timecode.py`: Timestamp formatting helpers.

## Data Flow
1. **Enrollment**: load audio → preprocess → Resemblyzer embedding → save metadata+embedding file.
2. **Transcription**: load audio → WhisperX ASR → alignment → diarization → per-word speaker assignment.
3. **Speaker Mapping**: aggregate diarized speaker embeddings → compare with user embedding (cosine similarity).
4. **Output**: produce structured JSON result plus optional text transcript with speaker labels.

## Configuration Notes
- Device auto-detection (CUDA, MPS, CPU) with CLI overrides.
- Similarity thresholds default to 0.75 (match) and 0.65 (maybe).
- HF token read from CLI flag, env var, or config file.
- Embeddings stored under `~/.transcribe_tool/enrollments/<user_id>.json` by default.

## Future Extensions
- Offline diarization fallback when HF token is unavailable.
- Batch transcription mode for folders.
- GUI wrapper (e.g., simple web front-end) if needed later.
