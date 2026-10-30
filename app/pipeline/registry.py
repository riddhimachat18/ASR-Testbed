from typing import Callable

from app.config import Settings

from .model_wrapper import ModelWrapper, StubModelWrapper


class ModelRegistry:
    def __init__(self, settings: Settings):
        self._factories: dict[str, Callable[[], ModelWrapper]] = {
            "stub": lambda: StubModelWrapper(settings.stub_sleep_seconds),
            # "whisper-small": lambda: WhisperWrapper("small"),   # Phase 3
        }

    def names(self) -> list[str]:
        return sorted(self._factories)

    def get(self, name: str) -> ModelWrapper:
        return self._factories[name]()
