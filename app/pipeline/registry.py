from typing import Callable

from pydantic import BaseModel

from app.config import Settings

from .model_wrapper import ModelWrapper, StubModelWrapper, StubOptions, StubPreparedWrapper
from .run_config import NoOptions


class ModelRegistry:
    def __init__(self, settings: Settings):
        self._factories: dict[str, Callable[[], ModelWrapper]] = {
            "stub": lambda: StubModelWrapper(settings.stub_sleep_seconds),
            "stub-prepared": lambda: StubPreparedWrapper(settings.stub_sleep_seconds),
            # "whisper-small": lambda: WhisperWrapper("small"),   # Phase 3
        }
        # model-specific settings accepted under config.model_options (anything undeclared is rejected)
        self._options: dict[str, type[BaseModel]] = {"stub": StubOptions, "stub-prepared": StubOptions}

    def names(self) -> list[str]:
        return sorted(self._factories)

    def get(self, name: str) -> ModelWrapper:
        return self._factories[name]()

    def options_model(self, name: str) -> type[BaseModel]:
        return self._options.get(name, NoOptions)
