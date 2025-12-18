# Transcribe Tool

Local-first CLI that combines WhisperX transcription, pyannote diarization, and Resemblyzer-based speaker recognition.

## Features
- `transcribe` command: word-level timestamps, diarization, optional user voice identification.
- Per-segment emotion labels using the Hugging Face model `firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3`.
- `enroll` command: generate and persist a voice embedding for later identification.
- JSON and human-readable transcript outputs with timestamped speaker labels.

## Requirements
- Python 3.10+
- FFmpeg installed and on your `PATH` (WhisperX and the diarization WAV fallback shell out to it for audio decoding), e.g., `brew install ffmpeg` on macOS
- PyTorch + torchaudio (CPU, CUDA, or MPS builds supported)
- Hugging Face access token for `pyannote/speaker-diarization-3.1` (accept the model terms at <https://huggingface.co/pyannote/speaker-diarization-3.1> before generating the token). The default diarization pipeline also downloads `pyannote/segmentation-3.0` and `pyannote/wespeaker-voxceleb-resnet34-LM`, so accept those licenses with the same account. The emotion classifier `firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3` is public today, but the same token will be forwarded if the repository ever requires gated access.
- Model downloads require internet the first time.

## Installation
Install [uv](https://github.com/astral-sh/uv) (see the project docs for platform-specific steps). Then synchronize the virtual environment and dependencies:

```bash
uv sync
```

This will create `.venv` (tracked in `.gitignore`). You can activate it with `source .venv/bin/activate`, or run commands directly through uv, e.g. `uv run transcribe --help`.

## Commands
### Prefetch models (recommended)
Download WhisperX, pyannote, and the emotion classifier ahead of time so `transcribe` runs without long downloads:

```bash
uv run transcribe init --hf-token YOUR_HF_TOKEN --language en
```

`--hf-token` is required for the pyannote diarization pipeline. Obtain it from https://huggingface.co/settings/tokens after accepting the model license.

If you previously accepted the community diarization model, you still need to accept `pyannote/speaker-diarization-3.1` before the token will work with the default pipeline.

Running `init` also writes your selected options (device, models, Hugging Face token, etc.) to `~/.transcribe_tool/.transcribe.config`. Future invocations of `transcribe` or `enroll` will automatically reuse those settings, so you only need to pass overrides when you want to change a value.

### Enroll a voice
```bash
uv run transcribe enroll path/to/enroll.wav --user-id alice
```
Stores the embedding at `~/.transcribe_tool/enrollments/alice.json`.

### Transcribe with diarization + recognition
```bash
uv run transcribe transcribe meeting.wav --user-id alice --output-dir transcripts/
```
Passing `--output-dir` creates (if needed) a directory and writes `meeting.json` and `meeting.txt` inside it based on the audio filename. You can still override either output path explicitly via `--json-out` / `--text-out` when you need custom filenames or locations. The CLI falls back to the values stored in `~/.transcribe_tool/.transcribe.config`. Pass `--device` / `--hf-token` (or edit the config file) whenever you need to override them temporarily. You can also keep using the `HUGGINGFACE_TOKEN` environment variable if you prefer not to persist the token on disk.

If you point at a non-WAV source (e.g., `.m4a`, `.mp3`), the tool automatically transcodes it to a temporary 16 kHz mono WAV via FFmpeg before running diarization so torchaudio/libsndfile quirks do not block the pipeline. Each segment in the console output (and JSON) now shows the detected emotion label too, e.g., `[00:00 - 00:05] SPEAKER_00 (happy): ...`.

## Troubleshooting
- `FileNotFoundError: 'ffmpeg'`: ensure FFmpeg is installed and accessible on `PATH` (e.g., `brew install ffmpeg`).
- `LibsndfileError: Format not recognised`: FFmpeg fallback should prevent this now; double-check FFmpeg is installed and retry so the CLI can transcode the input before diarization.
- `Unable to download diarization model`: visit <https://huggingface.co/pyannote/speaker-diarization-3.1>, accept the model terms with the same Hugging Face account as your token, then rerun. You can override `--diarization-model` if you want to use a different pyannote pipeline (e.g., the community build, which currently requires pyannote.audio 4.x).

## Development
- Source lives under `src/transcribe_tool`.
- Tests in `tests/` (currently lightweight, expand once audio fixtures exist). Run them via `uv run pytest`.
- Architecture overview in `docs/architecture.md`.

## Roadmap
1. Add audio fixtures and integration tests.
2. Support multiple enrolled users.
3. Provide offline diarization fallback.
