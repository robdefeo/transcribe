"""Transcribe Tool package."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("transcribe-tool")
except PackageNotFoundError:  # pragma: no cover - only during local dev
    __version__ = "0.1.0"

__all__ = ["__version__"]
