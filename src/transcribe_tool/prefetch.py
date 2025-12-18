"""Model prefetch utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Any

from .config import AppConfig
from .constants import DIARIZATION_MODEL_ID, EMOTION_MODEL_ID
from .runtime import RuntimeContext


@dataclass(slots=True)
class PrefetchSummary:
    """Summary of cached artifacts."""

    asr_model: str
    alignment_language: str
    diarization_model: str
    downloaded_asr: bool
    downloaded_alignment: bool
    downloaded_diarization: bool
    downloaded_emotion: bool


class Prefetcher:
    """Downloads heavy-weight models ahead of time."""

    def __init__(
        self,
        config: AppConfig,
        alignment_language: Optional[str] = None,
        *,
        runtime: RuntimeContext | None = None,
    ):
        self.config = config
        self.runtime = runtime or RuntimeContext.from_config(config)
        self.alignment_language = alignment_language or config.model.language or "en"

    def prefetch(self) -> PrefetchSummary:
        """Download ASR, alignment, and diarization weights."""

        _allow_omegaconf_safe_globals()
        asr_ok, align_ok = self._download_asr_and_alignment()
        diar_ok = self._download_diarization()
        emotion_ok = self._download_emotion_model()
        return PrefetchSummary(
            asr_model=self.config.model.asr_model,
            alignment_language=self.alignment_language,
            diarization_model=self.config.model.diarization_model,
            downloaded_asr=asr_ok,
            downloaded_alignment=align_ok,
            downloaded_diarization=diar_ok,
            downloaded_emotion=emotion_ok,
        )

    def _download_asr_and_alignment(self) -> tuple[bool, bool]:
        import whisperx

        device = self.runtime.device
        model = whisperx.load_model(
            self.config.model.asr_model,
            device=device,
            compute_type=self.config.model.compute_type,
        )
        del model

        align_model, _ = whisperx.load_align_model(
            language_code=self.alignment_language,
            device=device,
        )
        del align_model
        return True, True

    def _download_diarization(self) -> bool:
        from pyannote.audio import Pipeline

        token = self.runtime.hf_token
        if not token:
            raise RuntimeError(
                "Hugging Face token is required to download the diarization pipeline."
            )
        try:
            pipeline = Pipeline.from_pretrained(
                self.config.model.diarization_model,
                use_auth_token=token,
            )
        except AttributeError as exc:  # pyannote returns None when auth fails
            raise _diarization_access_error(self.config.model.diarization_model) from exc
        except Exception as exc:  # pragma: no cover - defensive, surfacing HF auth errors
            if _is_auth_error(exc):
                raise _diarization_access_error(self.config.model.diarization_model) from exc
            raise
        if pipeline is None:
            raise _diarization_access_error(self.config.model.diarization_model)
        device = self.runtime.device
        if device not in {"cpu", "auto"}:
            pipeline.to(device)
        del pipeline
        return True

    def _download_emotion_model(self) -> bool:
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - optional dependency load
            raise RuntimeError(
                "transformers is required to download the emotion classifier. Install it via `uv pip install transformers`."
            ) from exc

        kwargs = self.runtime.transformer_pipeline_kwargs(force_cpu=True)
        pipeline(
            task="audio-classification",
            model=EMOTION_MODEL_ID,
            **kwargs,
        )
        return True


def _allow_omegaconf_safe_globals() -> None:
    """Allow PyTorch to unpickle OmegaConf containers when weights_only=True."""

    try:
        from omegaconf import DictConfig, ListConfig
        from omegaconf.listconfig import ListConfig as ConcreteListConfig
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.nodes import AnyNode
        import torch.serialization as serialization
        from torch.torch_version import TorchVersion
        from pyannote.audio.core.model import Introspection
        from pyannote.audio.core.task import Specifications, Problem, Resolution
    except ImportError:  # pragma: no cover - optional dependency already required upstream
        return

    add_safe = getattr(serialization, "add_safe_globals", None)
    if not callable(add_safe):  # Older Torch versions
        return

    globals_to_add = []
    for cfg_cls in (
        DictConfig,
        ListConfig,
        ConcreteListConfig,
        ContainerMetadata,
        Metadata,
        AnyNode,
        Any,
        list,
        dict,
        tuple,
        set,
        int,
        float,
        bool,
        str,
        defaultdict,
        TorchVersion,
        Introspection,
        Specifications,
        Problem,
        Resolution,
    ):
        globals_to_add.append(cfg_cls)

    try:
        add_safe(globals_to_add)
    except Exception:  # pragma: no cover - defensive
        pass


def _diarization_access_error(model_name: str) -> RuntimeError:
    message = (
        "Unable to download diarization model. Visit "
        f"https://huggingface.co/{model_name} and accept the terms for your Hugging Face account, "
        "then rerun with a token that has access."
    )
    if model_name == DIARIZATION_MODEL_ID:
        message += (
            " This pipeline also depends on https://huggingface.co/pyannote/segmentation-3.0 "
            "and https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM; accept those licenses as well."
        )
    return RuntimeError(message)


def _is_auth_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(keyword in text for keyword in ("403", "forbidden", "unauthorized", "access"))
