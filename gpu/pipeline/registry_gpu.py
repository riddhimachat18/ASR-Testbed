"""Model registry — GPU edition.

Identical to app/pipeline/registry.py except:
  - FasterWhisperWrapperGPU is used instead of FasterWhisperWrapper
  - device and compute_type come from GPUSettings (device="cuda", compute_type="float16")
    rather than being hardcoded to "cpu" / "int8"

All model names are the same ("faster-whisper-tiny" etc.) so the DB, the API,
and the evaluation scripts work identically — only the inference device changes.
"""
from typing import Callable

from pydantic import BaseModel

from gpu.config_gpu import GPUSettings
from gpu.pipeline.faster_whisper_wrapper_gpu import FasterWhisperWrapperGPU
from app.pipeline.model_wrapper import ModelWrapper, StubModelWrapper, StubOptions, StubPreparedWrapper
from app.pipeline.run_config import NoOptions


class ModelRegistryGPU:
    def __init__(self, settings: GPUSettings):
        device = settings.device
        compute_type = settings.compute_type

        self._factories: dict[str, Callable[[], ModelWrapper]] = {
            # Stub models are device-agnostic; kept unchanged for parity with CPU registry.
            "stub": lambda: StubModelWrapper(settings.stub_sleep_seconds),
            "stub-prepared": lambda: StubPreparedWrapper(settings.stub_sleep_seconds),

            # GPU-backed faster-whisper models.
            # device and compute_type come from .env.gpu (default: cuda / float16).
            "faster-whisper-tiny": lambda: FasterWhisperWrapperGPU(
                model_size="tiny",
                device=device,
                compute_type=compute_type,
            ),
            "faster-whisper-base": lambda: FasterWhisperWrapperGPU(
                model_size="base",
                device=device,
                compute_type=compute_type,
            ),
            "faster-whisper-small": lambda: FasterWhisperWrapperGPU(
                model_size="small",
                device=device,
                compute_type=compute_type,
            ),
        }

        self._options: dict[str, type[BaseModel]] = {
            "stub": StubOptions,
            "stub-prepared": StubOptions,
            "faster-whisper-tiny": NoOptions,
            "faster-whisper-base": NoOptions,
            "faster-whisper-small": NoOptions,
        }

    def names(self) -> list[str]:
        return sorted(self._factories)

    def get(self, name: str) -> ModelWrapper:
        return self._factories[name]()

    def options_model(self, name: str) -> type[BaseModel]:
        return self._options.get(name, NoOptions)
