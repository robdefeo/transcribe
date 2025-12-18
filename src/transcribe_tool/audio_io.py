"""Audio I/O helpers for the transcribe tool."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
from soundfile import LibsndfileError

try:  # Optional dependency until torchaudio is installed.
    import torch
    import torchaudio
except ImportError:  # pragma: no cover - torchaudio not always available during tests.
    torch = None  # type: ignore[assignment]
    torchaudio = None  # type: ignore[assignment]


def load_audio(path: Path, target_sample_rate: int | None = None) -> Tuple[np.ndarray, int]:
    """Load an audio file into a mono numpy array.

    Parameters
    ----------
    path:
        Input audio path.
    target_sample_rate:
        Optional resampling rate.
    """

    data, sample_rate = sf.read(path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if target_sample_rate and sample_rate != target_sample_rate:
        if torchaudio is None or torch is None:
            raise RuntimeError(
                "torchaudio is required for resampling but is not installed."
            )
        waveform = torch.from_numpy(data).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sample_rate
        )
        data = resampler(waveform).squeeze(0).numpy()
        sample_rate = target_sample_rate

    return data.astype(np.float32), sample_rate


def slice_audio(
    waveform: np.ndarray,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
) -> np.ndarray:
    """Return a slice of the waveform covering the requested time window."""

    start_idx = int(start_sec * sample_rate)
    end_idx = int(end_sec * sample_rate)
    start_idx = max(start_idx, 0)
    end_idx = min(end_idx, waveform.shape[0])
    if start_idx >= end_idx:
        return np.array([], dtype=np.float32)
    return waveform[start_idx:end_idx]


def transcode_to_temp_wav(
    audio_path: Path,
    sample_rate: int | None = 16000,
    channels: int | None = 1,
) -> Path:
    """Convert ``audio_path`` to a temporary WAV file via ffmpeg."""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg is required to transcode non-WAV audio but was not found in PATH."
        )
    fd, tmp_name = tempfile.mkstemp(prefix="transcribe_audio_", suffix=".wav")
    os.close(fd)
    tmp_path = Path(tmp_name)
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
    ]
    if channels:
        cmd.extend(["-ac", str(channels)])
    if sample_rate:
        cmd.extend(["-ar", str(sample_rate)])
    cmd.append(str(tmp_path))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to transcode {audio_path.name} to WAV via ffmpeg."
        ) from exc
    return tmp_path
