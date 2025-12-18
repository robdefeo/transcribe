"""Application configuration helpers."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

from .constants import DIARIZATION_MODEL_ID


DEFAULT_ENROLLMENT_DIR = Path.home() / ".transcribe_tool" / "enrollments"
CONFIG_PATH = DEFAULT_ENROLLMENT_DIR.parent / ".transcribe.config"


@dataclass(slots=True)
class ModelConfig:
    """Models used throughout the pipeline."""

    asr_model: str = "large-v2"
    diarization_model: str = DIARIZATION_MODEL_ID
    compute_type: str = "int8_float16"
    batch_size: int = 8
    language: Optional[str] = None


@dataclass(slots=True)
class ThresholdConfig:
    """Similarity thresholds for speaker recognition."""

    match_threshold: float = 0.75
    maybe_threshold: float = 0.65


@dataclass(slots=True)
class AppPaths:
    """Filesystem layout for generated artifacts."""

    enrollment_dir: Path = field(default_factory=lambda: DEFAULT_ENROLLMENT_DIR)

    def ensure(self) -> None:
        self.enrollment_dir.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class AppConfig:
    """Central configuration for CLI commands."""

    device: str = "auto"
    huggingface_token: Optional[str] = None
    model: ModelConfig = field(default_factory=ModelConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    paths: AppPaths = field(default_factory=AppPaths)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppConfig":
        """Load configuration from disk, falling back to defaults."""

        resolved_path = path or CONFIG_PATH
        if not resolved_path.exists():
            return cls()
        try:
            with resolved_path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except json.JSONDecodeError:
            return cls()

        model_data: Dict[str, Any] = raw.get("model", {})
        thresholds_data: Dict[str, Any] = raw.get("thresholds", {})
        paths_data: Dict[str, Any] = raw.get("paths", {})
        enrollment_dir = Path(paths_data.get("enrollment_dir", DEFAULT_ENROLLMENT_DIR))

        config = cls(
            device=raw.get("device", "auto"),
            huggingface_token=raw.get("huggingface_token"),
            model=ModelConfig(**model_data),
            thresholds=ThresholdConfig(**thresholds_data),
            paths=AppPaths(enrollment_dir=enrollment_dir),
        )
        return config

    def save(self, path: Optional[Path] = None) -> None:
        """Persist the configuration to disk."""

        resolved_path = path or CONFIG_PATH
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "device": self.device,
            "huggingface_token": self.huggingface_token,
            "model": asdict(self.model),
            "thresholds": asdict(self.thresholds),
            "paths": {"enrollment_dir": str(self.paths.enrollment_dir)},
        }
        with resolved_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def apply_overrides(
        self,
        *,
        device: Optional[str] = None,
        huggingface_token: Optional[str] = None,
        model_overrides: Optional[Dict[str, Any]] = None,
    ) -> "AppConfig":
        """Mutate the config with CLI-provided overrides."""

        if device:
            self.device = device
        if huggingface_token:
            self.huggingface_token = huggingface_token
        if model_overrides:
            clean_overrides = {k: v for k, v in model_overrides.items() if v is not None}
            if clean_overrides:
                self.model = replace(self.model, **clean_overrides)
        return self

    def resolved_device(self) -> str:
        """Resolve the execution device based on availability."""

        if self.device != "auto":
            return self.device

        # Basic auto-detection without importing heavy libs eagerly.
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible and cuda_visible != "":
            return "cuda"

        if os.uname().sysname == "Darwin":
            # Assume Metal Performance Shaders are available on Apple Silicon.
            if os.uname().machine.startswith("arm"):
                return "mps"

        return "cpu"

    def resolved_hf_token(self) -> Optional[str]:
        """Return the Hugging Face token from env or config."""

        if self.huggingface_token:
            return self.huggingface_token
        return os.environ.get("HUGGINGFACE_TOKEN")

    def enrollment_path_for(self, user_id: str) -> Path:
        """Return the default path to store a user embedding."""

        safe_user = user_id.replace("/", "_")
        self.paths.ensure()
        return self.paths.enrollment_dir / f"{safe_user}.json"


def build_config_from_cli(
    *,
    device: Optional[str] = None,
    huggingface_token: Optional[str] = None,
    asr_model: Optional[str] = None,
    compute_type: Optional[str] = None,
    language: Optional[str] = None,
    diarization_model: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> AppConfig:
    """Load persisted config and apply CLI overrides."""

    config = AppConfig.load()
    model_overrides = {
        "asr_model": asr_model,
        "compute_type": compute_type,
        "language": language,
        "diarization_model": diarization_model,
        "batch_size": batch_size,
    }
    config.apply_overrides(
        device=device,
        huggingface_token=huggingface_token,
        model_overrides=model_overrides,
    )
    return config
