from typing import Callable

from pydantic import BaseModel

from app.config import Settings

from .faster_whisper_wrapper import FasterWhisperWrapper, faster_whisper_options
from .model_wrapper import ModelWrapper, StubModelWrapper, StubOptions, StubPreparedWrapper
from .run_config import NoOptions

FASTER_WHISPER_SIZES = ("tiny", "base", "small", "medium")


class ModelRegistry:
    """One registry for CPU and GPU machines. Device and precision come from settings as defaults and are
    recorded per run in config.model_options (see faster_whisper_wrapper.py)."""

    def __init__(self, settings: Settings):
        self._factories: dict[str, Callable[[], ModelWrapper]] = {
            "stub": lambda: StubModelWrapper(settings.stub_sleep_seconds),
            "stub-prepared": lambda: StubPreparedWrapper(settings.stub_sleep_seconds),
        }
        # model-specific settings accepted under config.model_options (anything undeclared is rejected)
        self._options: dict[str, type[BaseModel]] = {"stub": StubOptions, "stub-prepared": StubOptions}

        fw_options = faster_whisper_options(settings.device, settings.compute_type)
        for size in FASTER_WHISPER_SIZES:  # weights are downloaded from Hugging Face on first use
            name = f"faster-whisper-{size}"
            self._factories[name] = (lambda size=size: FasterWhisperWrapper(
                model_size=size, device=settings.device, compute_type=settings.compute_type))
            self._options[name] = fw_options

    def names(self) -> list[str]:
        return sorted(self._factories)

    def get(self, name: str) -> ModelWrapper:
        return self._factories[name]()

    def options_model(self, name: str) -> type[BaseModel]:
        return self._options.get(name, NoOptions)
