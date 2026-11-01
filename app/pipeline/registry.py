from typing import Callable

from pydantic import BaseModel

from app.config import Settings

from .faster_whisper_wrapper import FasterWhisperWrapper
from .model_wrapper import ModelWrapper, StubModelWrapper, StubOptions, StubPreparedWrapper
from .run_config import NoOptions


class ModelRegistry:
    def __init__(self, settings: Settings):
        self._factories: dict[str, Callable[[], ModelWrapper]] = {
            "stub": lambda: StubModelWrapper(settings.stub_sleep_seconds),
            "stub-prepared": lambda: StubPreparedWrapper(settings.stub_sleep_seconds),
            # faster-whisper models — model weights are downloaded on first use from HuggingFace
            "faster-whisper-tiny": lambda: FasterWhisperWrapper(model_size="tiny",  device="cpu", compute_type="int8"),
            "faster-whisper-base": lambda: FasterWhisperWrapper(model_size="base",  device="cpu", compute_type="int8"),
            "faster-whisper-small": lambda: FasterWhisperWrapper(model_size="small", device="cpu", compute_type="int8"),
        }
        # model-specific settings accepted under config.model_options (anything undeclared is rejected)
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
