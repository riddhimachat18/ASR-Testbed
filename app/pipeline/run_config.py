"""What exactly a run did: one strictly validated, hashable settings object.

Rules that make results reproducible:
  * extra="forbid" everywhere: a typo (beem_size) is an error, never silently ignored.
  * The STORED config is the fully resolved one (defaults filled in), so "omitted" and "explicit default" are the
    same run, and the row says what really happened even if the defaults change in a later release.
  * Settings that cannot apply are rejected (e.g. overlap with VAD chunking), so two equivalent configs cannot
    hash differently.
  * config_hash identifies model_version + resolved config: the leaderboard groups on it.
"""
import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AudioConfig(Strict):
    sample_rate: Literal[8000, 16000] = 16000  # rate the model sees; the stored original is never changed


class VadConfig(Strict):
    backend: Literal["silero", "webrtc"] = "silero"
    threshold: float = Field(0.5, gt=0, lt=1)
    min_speech_ms: int = Field(250, ge=0, le=10_000)
    min_silence_ms: int = Field(100, ge=0, le=10_000)
    speech_pad_ms: int = Field(30, ge=0, le=1_000)


class ChunkingConfig(Strict):
    strategy: Literal["vad", "fixed"] = "vad"
    max_chunk_s: float = Field(30.0, gt=1, le=60)  # also the forced-cut fallback for VAD chunking
    overlap_s: float = Field(0.0, ge=0)            # fixed windows only
    vad: VadConfig | None = None                   # VAD chunking only

    @model_validator(mode="after")
    def _consistent(self):
        if self.strategy == "vad":
            if self.overlap_s != 0:
                raise ValueError("overlap_s applies only to strategy 'fixed'; leave it at 0 with 'vad'")
            if self.vad is None:
                self.vad = VadConfig()
        else:
            if self.vad is not None:
                raise ValueError("vad settings apply only to strategy 'vad'")
            if self.overlap_s >= self.max_chunk_s / 2:
                raise ValueError("overlap_s must be less than half of max_chunk_s")
        return self


class DecodingConfig(Strict):
    beam_size: int = Field(5, ge=1, le=20)
    temperature: float = Field(0.0, ge=0, le=1)
    language: str | None = Field(None, pattern=r"^[a-z]{2,3}$")  # None = auto-detect
    task: Literal["transcribe", "translate"] = "transcribe"


class RunConfig(Strict):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    schema_version: Literal[1] = 1
    audio: AudioConfig = Field(default_factory=AudioConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    decoding: DecodingConfig = Field(default_factory=DecodingConfig)
    model_options: dict[str, Any] = Field(default_factory=dict)  # validated per model by the registry


class NoOptions(Strict):
    """Options model for wrappers that declare none: any option is rejected."""


def config_hash(model_version: str, resolved: dict) -> str:
    payload = json.dumps({"model_version": model_version, "config": resolved},
                         sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _errors(e: ValidationError, prefix: tuple = ()) -> list[dict]:
    errs = json.loads(e.json(include_url=False, include_context=False))
    for err in errs:
        err["loc"] = [*prefix, *err["loc"]]
    return errs


class ConfigInvalid(Exception):
    def __init__(self, errors: list[dict]):
        super().__init__(errors)
        self.errors = errors


def resolve_run_config(model_version: str, raw: dict, options_model: type[BaseModel]) -> tuple[dict, str]:
    """Validate, fill in defaults, validate model-specific options. Returns (resolved dict, hash)."""
    try:
        cfg = RunConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigInvalid(_errors(e)) from e
    try:
        cfg.model_options = options_model.model_validate(cfg.model_options).model_dump(mode="json")
    except ValidationError as e:
        raise ConfigInvalid(_errors(e, ("model_options",))) from e
    resolved = cfg.model_dump(mode="json")
    return resolved, config_hash(model_version, resolved)
