"""Shared runtime context helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import AppConfig


@dataclass(slots=True)
class RuntimeContext:
    """Captures computed runtime settings such as device and HF token."""

    device: str
    hf_token: Optional[str]

    @classmethod
    def from_config(cls, config: AppConfig) -> "RuntimeContext":
        return cls(
            device=config.resolved_device(),
            hf_token=config.resolved_hf_token(),
        )

    def huggingface_token(self) -> Optional[str]:
        return self.hf_token

    def transformer_device(self) -> Any:
        """Return the device argument expected by Hugging Face pipelines."""

        dev = (self.device or "cpu").lower()
        if dev.startswith("cuda"):
            # Hugging Face expects an integer GPU id for CUDA devices.
            if ":" in dev:
                _, idx = dev.split(":", 1)
                try:
                    return int(idx)
                except ValueError:
                    return 0
            return 0
        if dev.startswith("mps"):
            try:  # Import torch lazily to avoid heavy dependency at import time.
                import torch  # type: ignore[import-not-found]

                if torch.backends.mps.is_available():
                    return torch.device("mps")
            except Exception:  # pragma: no cover - fall back to CPU when torch is missing.
                return "cpu"
        return "cpu"

    def transformer_pipeline_kwargs(self, *, force_cpu: bool = False) -> Dict[str, Any]:
        """Build kwargs for transformers.pipeline with consistent auth/device."""

        kwargs: Dict[str, Any] = {}
        kwargs["device"] = "cpu" if force_cpu else self.transformer_device()
        if self.hf_token:
            kwargs["token"] = self.hf_token
        return kwargs
