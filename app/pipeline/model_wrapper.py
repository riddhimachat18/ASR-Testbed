import time
from abc import ABC, abstractmethod
from typing import BinaryIO


class ModelWrapper(ABC):
    """Swap point for research: new model/preprocessing/decoding = new subclass."""

    @abstractmethod
    def transcribe(self, audio: BinaryIO, config: dict) -> str: ...


class StubModelWrapper(ModelWrapper):
    def __init__(self, sleep_seconds: float = 0.5):
        self.sleep_seconds = sleep_seconds

    def transcribe(self, audio: BinaryIO, config: dict) -> str:
        time.sleep(self.sleep_seconds)
        if config.get("force_error"):  # exercises the failure path in tests/load tests
            raise RuntimeError("forced stub failure")
        return "this is a dummy transcript"
