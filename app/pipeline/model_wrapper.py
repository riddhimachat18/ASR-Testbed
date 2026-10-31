import time
from abc import ABC, abstractmethod
from typing import BinaryIO

from .run_config import RunConfig, Strict


class ModelWrapper(ABC):
    """Swap point for research: new model/preprocessing/decoding = new subclass."""

    @abstractmethod
    def transcribe(self, audio: BinaryIO, config: RunConfig) -> str: ...


class StubOptions(Strict):
    force_error: bool = False  # lets tests and load tests exercise the failure path


class StubModelWrapper(ModelWrapper):
    def __init__(self, sleep_seconds: float = 0.5):
        self.sleep_seconds = sleep_seconds

    def transcribe(self, audio: BinaryIO, config: RunConfig) -> str:
        time.sleep(self.sleep_seconds)
        if config.model_options.get("force_error"):  # exercises the failure path in tests/load tests
            raise RuntimeError("forced stub failure")
        return "this is a dummy transcript"
