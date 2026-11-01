import time
from abc import ABC, abstractmethod
from typing import BinaryIO

from .prepare import Prepared
from .run_config import RunConfig, Strict


class ModelWrapper(ABC):
    """Swap point for research: new model/preprocessing/decoding = new subclass."""

    # True: the worker passes the cached 16 kHz mono WAV as `audio` plus a chunk plan in `prepared`.
    # False: `audio` is the original file exactly as uploaded and `prepared` is None.
    needs_preparation: bool = False

    @abstractmethod
    def transcribe(self, audio: BinaryIO, config: RunConfig, prepared: Prepared | None = None) -> str: ...


class StubOptions(Strict):
    force_error: bool = False  # lets tests and load tests exercise the failure path


class StubModelWrapper(ModelWrapper):
    def __init__(self, sleep_seconds: float = 0.5):
        self.sleep_seconds = sleep_seconds

    def transcribe(self, audio: BinaryIO, config: RunConfig, prepared: Prepared | None = None) -> str:
        time.sleep(self.sleep_seconds)
        if config.model_options.get("force_error"):  # exercises the failure path in tests/load tests
            raise RuntimeError("forced stub failure")
        return "this is a dummy transcript"


class StubPreparedWrapper(StubModelWrapper):
    """Stub that goes through the preparation path, so the derived cache can be exercised without a real model.
    VAD chunking needs a VAD backend (Part G); use {"chunking": {"strategy": "fixed"}} until then."""
    needs_preparation = True

    def transcribe(self, audio: BinaryIO, config: RunConfig, prepared: Prepared | None = None) -> str:
        super().transcribe(audio, config, prepared)  # sleeps, honours force_error
        return f"stub transcript from {len(prepared.chunks)} chunks"
