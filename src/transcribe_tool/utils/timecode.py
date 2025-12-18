"""Timecode helpers."""

from __future__ import annotations

import math


def format_timestamp(value: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""

    total_ms = int(round(value * 1000))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    seconds = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def parse_timestamp(text: str) -> float:
    """Parse HH:MM:SS.mmm string back into seconds."""

    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp: {text}")
    hours, minutes = int(parts[0]), int(parts[1])
    seconds_parts = parts[2].split(".")
    seconds = int(seconds_parts[0])
    if len(seconds_parts) > 1:
        ms_str = seconds_parts[1]
        # Pad or truncate to 3 digits
        ms_str = (ms_str + "000")[:3]
        millis = int(ms_str)
    else:
        millis = 0
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def duration_from_samples(num_samples: int, sample_rate: int) -> float:
    """Return duration in seconds for a waveform."""

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    return num_samples / float(sample_rate)
