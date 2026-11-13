"""FasterWhisperWrapper: ModelWrapper backed by faster-whisper, for CPU and GPU alike.

Design decisions
────────────────
* Device and precision are run settings, not code paths.
  `model_options.device` ("cpu" | "cuda") and `model_options.compute_type` ("int8", "float16", ...) are validated by
  FasterWhisperOptions and stored in the run's resolved config, so they are part of config_hash. The defaults come
  from the submitting machine's settings (DEVICE, COMPUTE_TYPE). Before Part H a separate gpu/ package copied this
  module to change those two values, and CPU and GPU runs got the same hash.

* Decoding follows the stored config exactly.
  beam_size, language, task AND temperature are passed to faster-whisper. Before Part H temperature was never
  passed, so faster-whisper's fallback schedule (0.0, 0.2, ... 1.0) ran although the stored config said 0.0.

* Temperature fallback is an explicit, hashed choice (model_options.temperature_fallback, default False).
  Off: one temperature, so decoding is deterministic (same audio, same transcript) and compute per segment is
  bounded. On: Whisper's schedule; a segment whose output looks like a loop (compression ratio > 2.4) or has low
  confidence (avg log-prob < -1.0) is re-decoded by sampling at the next temperature, up to 5 more times. That
  is random and can cost up to ~6x on hard segments. Decided in Part H.7: off, with a deterministic loop guard.

* Deterministic loop guard: model_options.no_repeat_ngram_size (default 3; 0 = off) forbids repeating any
  n-gram of that size within a segment, and repetition_penalty (1.0 = off) down-weights tokens already
  generated. Neither samples, so decoding stays deterministic. Tiny without a guard looped once on test-clean
  ("North, North, ..."); n=3 fixed it (design log H.7). Legacy configs without the key decode with no guard.

* Decoder statistics for the run profile (last_stats): segments, segments that needed a fallback temperature,
  output words, and the longest run of one repeated word (a reference-free loop signal).

* needs_preparation = True
  The worker sends the cached 16 kHz mono WAV stream and a Prepared object (chunk time boundaries). We read each
  slice, transcribe it, and stitch the results.

* Process-level model cache
  Loading takes 1 to 5 s. One model per (size, device, compute_type) per process, lock-guarded. In Celery each child
  process loads it on its first task.

* Model internals are timed, not just the call (instrument_model)
  faster-whisper's transcribe() hides four steps: decode the audio bytes, compute the 80-bin log-mel features,
  run the ENCODER once per 30 s window (cost fixed per window: short audio is padded with silence), and run the
  DECODER token by token (cost grows with output length and beam size, and repeats on a temperature fallback).
  At model load we wrap the model's own feature_extractor, encode, detect_language and generate_with_fallback
  methods so each call becomes a span in the run's trace (app/observability/tracing.py), with what it processed:
  frames per encoder call, tokens out, avg log-prob, compression ratio, final temperature, no-speech probability.
  The wrappers only time and observe; inputs and outputs pass through unchanged. If faster-whisper renames a
  method, that span is simply missing (and a warning is logged), never an error.

* Chunk slicing and stitching
  We seek to each chunk's sample range and pass a WAV slice to faster-whisper, so one chunk is in memory at a time.
  Chunk transcripts are joined with a single space. VAD chunks do not overlap, so no de-duplication is needed.
"""
from __future__ import annotations

import io
import logging
import threading
from typing import BinaryIO, Literal

from pydantic import BaseModel, Field, create_model, model_validator

from app.observability.tracing import span
from app.pipeline.model_wrapper import ModelWrapper
from app.pipeline.prepare import Prepared
from app.pipeline.run_config import RunConfig, Strict

log = logging.getLogger("faster_whisper")

Device = Literal["cpu", "cuda"]
ComputeType = Literal["int8", "int8_float32", "int8_float16", "int8_bfloat16", "float16", "bfloat16", "float32"]
_GPU_ONLY = {"float16", "int8_float16"}  # CTranslate2 has no float16 kernels on CPU


FALLBACK_SCHEDULE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)  # Whisper's standard temperature schedule


class FasterWhisperOptions(Strict):
    """config.model_options for every faster-whisper model."""

    device: Device = "cpu"
    compute_type: ComputeType = "int8"
    temperature_fallback: bool = False                 # see module docstring; False keeps decoding deterministic
    # Deterministic loop guard; 0 = off. Default 3 since H.7: on test-clean it removed tiny's only loop (-109
    # errors) at a cost of +9 errors over 10 other files, e.g. it blocks a genuinely repeated phrase
    # ("an antipathy, an antipathy"). Re-check on domain audio: it can also block long digit runs ("0 0 0 0").
    no_repeat_ngram_size: int = Field(3, ge=0, le=10)
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0)  # deterministic loop guard; 1.0 = off

    @model_validator(mode="after")
    def _valid_combination(self):
        if self.device == "cpu" and self.compute_type in _GPU_ONLY:
            raise ValueError(f"compute_type '{self.compute_type}' needs device 'cuda'")
        return self


def faster_whisper_options(device: str, compute_type: str) -> type[BaseModel]:
    """FasterWhisperOptions whose defaults are this machine's settings.

    The defaults are filled into every stored config, so the hash records the device and precision a run used
    even when the client did not name them."""
    FasterWhisperOptions(device=device, compute_type=compute_type)  # fail fast on a bad DEVICE/COMPUTE_TYPE pair
    return create_model(
        "FasterWhisperOptions",
        __base__=FasterWhisperOptions,
        device=(Device, device),
        compute_type=(ComputeType, compute_type),
    )


# ── process-level model cache ──────────────────────────────────────────────────
_lock = threading.Lock()
_models: dict[str, object] = {}  # "<model_size>/<device>/<compute_type>" -> WhisperModel


def _get_model(model_size: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel  # heavy import; only workers that run the model pay for it

    cache_key = f"{model_size}/{device}/{compute_type}"
    if cache_key not in _models:
        with _lock:
            if cache_key not in _models:
                log.info("loading faster-whisper model",
                         extra={"model_size": model_size, "device": device, "compute_type": compute_type})
                _models[cache_key] = instrument_model(
                    WhisperModel(model_size, device=device, compute_type=compute_type))
    return _models[cache_key]


def is_loaded(model_size: str, device: str, compute_type: str) -> bool:
    return f"{model_size}/{device}/{compute_type}" in _models


class _TimedFeatureExtractor:
    """Proxy for faster-whisper's FeatureExtractor: times __call__ (log-mel), forwards every attribute."""

    def __init__(self, inner):
        self._inner = inner

    def __call__(self, audio, *args, **kwargs):
        with span("model.features", audio_s=round(len(audio) / self._inner.sampling_rate, 3)) as a:
            out = self._inner(audio, *args, **kwargs)
            a["frames"] = int(out.shape[-1])
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _instrument_decode_audio() -> None:
    """faster-whisper decodes WAV bytes with PyAV (module-level function); time it once, process-wide."""
    import faster_whisper.transcribe as fwt

    original = fwt.decode_audio
    if getattr(original, "_asr_timed", False):
        return

    def decode_audio(*args, **kwargs):
        with span("model.audio_decode"):
            return original(*args, **kwargs)

    decode_audio._asr_timed = True
    fwt.decode_audio = decode_audio


def instrument_model(model):
    """Turn the model's internal steps into trace spans (see module docstring). Idempotent."""
    if getattr(model, "_asr_instrumented", False):
        return model
    try:
        _instrument_decode_audio()
    except Exception:  # noqa: BLE001
        log.warning("could not instrument faster_whisper.transcribe.decode_audio")
    missing = [m for m in ("feature_extractor", "encode", "detect_language", "generate_with_fallback")
               if not hasattr(model, m)]
    if missing:
        log.warning("faster-whisper internals not found; these spans will be missing", extra={"missing": missing})
    if hasattr(model, "feature_extractor"):
        model.feature_extractor = _TimedFeatureExtractor(model.feature_extractor)
    if hasattr(model, "encode"):
        encode = model.encode

        def timed_encode(features):
            with span("model.encode", frames=int(features.shape[-1])):
                return encode(features)
        model.encode = timed_encode
    if hasattr(model, "detect_language"):
        detect_language = model.detect_language

        def timed_detect_language(*args, **kwargs):
            with span("model.language_detect") as a:
                out = detect_language(*args, **kwargs)
                if isinstance(out, tuple) and out:
                    a["language"] = out[0]
                return out
        model.detect_language = timed_detect_language
    if hasattr(model, "generate_with_fallback"):
        generate = model.generate_with_fallback

        def timed_generate(encoder_output, prompt, tokenizer, options):
            with span("model.decode", prompt_tokens=len(prompt), beam_size=getattr(options, "beam_size", None)) as a:
                out = generate(encoder_output, prompt, tokenizer, options)
                try:
                    result, avg_logprob, temperature, compression_ratio = out
                    a.update(tokens=len(result.sequences_ids[0]), avg_logprob=float(avg_logprob),
                             temperature=float(temperature), compression_ratio=float(compression_ratio),
                             no_speech_prob=float(getattr(result, "no_speech_prob", 0.0)))
                except Exception:  # noqa: BLE001  observation only
                    pass
                return out
        model.generate_with_fallback = timed_generate
    model._asr_instrumented = True
    return model


def _read_chunk_bytes(wav_stream: BinaryIO, start_s: float, end_s: float, sample_rate: int) -> bytes:
    """One time slice of an open WAV stream, returned as WAV bytes at the same sample rate."""
    import soundfile as sf

    start_frame = int(start_s * sample_rate)
    n_frames = max(1, int((end_s - start_s) * sample_rate))
    wav_stream.seek(0)
    with sf.SoundFile(wav_stream) as f:
        f.seek(start_frame)
        data = f.read(n_frames, dtype="float32", always_2d=False)
    buf = io.BytesIO()
    with sf.SoundFile(buf, mode="w", samplerate=sample_rate, channels=1, subtype="PCM_16", format="WAV") as out:
        out.write(data)
    return buf.getvalue()


class FasterWhisperWrapper(ModelWrapper):
    """faster-whisper transcriber, VAD-chunked, with chunk-and-stitch.

    device and compute_type given here are only the fallback for legacy runs whose stored config has no
    model_options; new runs always carry both."""

    needs_preparation: bool = True

    def __init__(self, model_size: str = "tiny", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

    def resolve_hardware(self, config: RunConfig) -> tuple[str, str]:
        opts = config.model_options or {}
        return opts.get("device", self.device), opts.get("compute_type", self.compute_type)

    @staticmethod
    def decode_kwargs(config: RunConfig) -> dict:
        """The exact keyword arguments given to faster-whisper for this config (legacy configs without the
        Part H options decode with fallback off and no loop guard)."""
        dec, opts = config.decoding, config.model_options or {}
        if opts.get("temperature_fallback", False):
            temperature = [dec.temperature] + [t for t in FALLBACK_SCHEDULE if t > dec.temperature]
        else:
            temperature = dec.temperature  # one value: no fallback, deterministic
        return {"language": dec.language, "beam_size": dec.beam_size, "task": dec.task,
                "temperature": temperature,
                "no_repeat_ngram_size": int(opts.get("no_repeat_ngram_size", 0)),
                "repetition_penalty": float(opts.get("repetition_penalty", 1.0)),
                "vad_filter": False,  # our own VAD chunking runs upstream
                "word_timestamps": False}

    def transcribe(self, audio: BinaryIO, config: RunConfig, prepared: Prepared | None = None) -> str:
        device, compute_type = self.resolve_hardware(config)
        with span("model.get", model_size=self.model_size, device=device, compute_type=compute_type,
                  cold=not is_loaded(self.model_size, device, compute_type)):
            model = _get_model(self.model_size, device, compute_type)
        kwargs = self.decode_kwargs(config)
        base_temperature = config.decoding.temperature
        sample_rate = prepared.sample_rate if prepared else config.audio.sample_rate
        chunks = prepared.chunks if prepared else None
        stats = {"segments": 0, "fallback_segments": 0}

        if not chunks:  # no chunk plan (or an empty one): transcribe the whole file in one shot
            audio.seek(0)
            with span("chunk", index=0, whole_file=True):
                parts = [self._transcribe_stream(model, audio, kwargs, base_temperature, stats)]
        else:
            parts = []
            for i, chunk in enumerate(chunks):
                with span("chunk", index=i, start_s=chunk.start, end_s=chunk.end,
                          audio_s=round(chunk.end - chunk.start, 3)):
                    with span("chunk.slice"):
                        chunk_buf = io.BytesIO(_read_chunk_bytes(audio, chunk.start, chunk.end, sample_rate))
                    parts.append(self._transcribe_stream(model, chunk_buf, kwargs, base_temperature, stats))
        text = " ".join(p for p in parts if p)
        self.last_stats = {**stats, **output_stats(text)}  # one wrapper instance per run (registry.get)
        return text

    @staticmethod
    def _transcribe_stream(model, stream: BinaryIO, kwargs: dict, base_temperature: float, stats: dict) -> str:
        stream.seek(0)
        segments, _info = model.transcribe(stream, **kwargs)
        texts = []
        for seg in segments:
            stats["segments"] += 1
            if (getattr(seg, "temperature", None) or 0.0) > base_temperature:
                stats["fallback_segments"] += 1  # this segment was re-decoded at a higher temperature
            if seg.text.strip():
                texts.append(seg.text.strip())
        return " ".join(texts)


def output_stats(text: str) -> dict:
    """Reference-free loop signal: output length and the longest run of one word repeated back to back
    (case and punctuation ignored). "North, North, North" has a run of 3."""
    words = ["".join(ch for ch in w.lower() if ch.isalnum()) for w in text.split()]
    words = [w for w in words if w]
    longest = run = 0
    for i, w in enumerate(words):
        run = run + 1 if i and w == words[i - 1] else 1
        longest = max(longest, run)
    return {"output_words": len(words), "longest_repeat": longest}
