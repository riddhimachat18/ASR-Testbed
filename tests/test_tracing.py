"""Tracing: every run attempt leaves timed, nested stage_events; faster-whisper's internals become spans; each
audio file gets its features once."""
import io
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import select

from app.db.models import AudioFeatures, StageEvent, TranscriptionRun
from app.observability import tracing
from app.observability.tracing import Trace, activate, annotate, span
from app.pipeline import faster_whisper_wrapper as fw
from app.pipeline.audio_features import compute_features
from app.pipeline.run_config import RunConfig
from tests.conftest import make_wav
from tests.test_evaluation import _post, _wait_done

FIXED = json.dumps({"chunking": {"strategy": "fixed"}})


# ── the tracer itself ─────────────────────────────────────────────────────────
def test_spans_nest_time_and_record_errors():
    trace = Trace("r1")
    with activate(trace):
        with span("outer", a=1):
            with span("inner") as attrs:
                time.sleep(0.01)
                attrs["tokens"] = 7
            annotate(extra=True)
        with pytest.raises(ValueError):
            with span("boom"):
                raise ValueError("x")
    rows = {r["stage"]: r for r in trace.rows()}
    assert rows["inner"]["parent_seq"] == rows["outer"]["seq"] and rows["outer"]["parent_seq"] is None
    assert rows["inner"]["duration_ms"] >= 9 and rows["outer"]["duration_ms"] >= rows["inner"]["duration_ms"]
    assert rows["inner"]["started_at"] >= rows["outer"]["started_at"]
    assert rows["inner"]["attrs"] == {"tokens": 7} and rows["outer"]["attrs"] == {"a": 1, "extra": True}
    assert rows["boom"]["attrs"] == {"error": "ValueError"}
    assert tracing.current() is None


def test_span_is_a_noop_outside_a_run():
    with span("anything", x=1) as attrs:
        attrs["y"] = 2  # must not raise
    annotate(z=3)


# ── a whole run ───────────────────────────────────────────────────────────────
def _events(client, run_id) -> list[StageEvent]:
    """Events are written just after the run's status, so poll briefly."""
    end = time.time() + 3
    while True:
        with client.app.state.session_factory() as s:
            rows = list(s.scalars(select(StageEvent).where(StageEvent.run_id == run_id).order_by(StageEvent.seq)))
            s.expunge_all()
        if any(r.stage == "run" for r in rows) or time.time() > end:
            return rows
        time.sleep(0.02)


def _features(client, audio_id):
    end = time.time() + 3
    while True:
        with client.app.state.session_factory() as s:
            f = s.get(AudioFeatures, audio_id)
            if f is not None or time.time() > end:
                if f is not None:
                    s.expunge(f)
                return f
        time.sleep(0.02)


def test_a_done_run_leaves_nested_stage_events(client, make_user):
    _, key = make_user("alice")
    sub = _post(client, key, reference="this is a dummy transcript", model_version="stub-prepared", config=FIXED)
    assert _wait_done(client, key, sub["run_id"])["status"] == "done"
    events = _events(client, sub["run_id"])
    by_stage = {e.stage: e for e in events}
    for stage in ("queue_wait", "run", "db.load", "prepare", "cache.chunks", "cache.resampled", "transcribe",
                  "score", "db.write"):
        assert stage in by_stage, stage
    run = by_stage["run"]
    for stage in ("db.load", "prepare", "transcribe", "score", "db.write"):
        assert by_stage[stage].parent_seq == run.seq
        assert by_stage[stage].duration_ms <= run.duration_ms + 1e-6
    assert by_stage["cache.chunks"].attrs["hit"] is False
    assert by_stage["transcribe"].duration_ms >= 40  # the stub sleeps 0.05 s
    assert all(e.attempt == 1 and e.worker for e in events)
    with client.app.state.session_factory() as s:
        profile = s.get(TranscriptionRun, sub["run_id"]).profile
    assert profile["stages"]["transcribe"] >= 0.04


def test_a_failed_run_still_leaves_its_events(client, make_user):
    _, key = make_user("alice")
    sub = _post(client, key, config=json.dumps({"model_options": {"force_error": True}}))
    assert _wait_done(client, key, sub["run_id"])["status"] == "failed"
    stages = {e.stage: e for e in _events(client, sub["run_id"])}
    assert "run" in stages and stages["run"].attrs.get("error")


def test_audio_features_are_computed_once_per_file(client, make_user):
    _, key = make_user("alice")
    sub = _post(client, key, model_version="stub-prepared", config=FIXED)
    _wait_done(client, key, sub["run_id"])
    with client.app.state.session_factory() as s:
        audio_id = s.get(TranscriptionRun, sub["run_id"]).audio_id
    f = _features(client, audio_id)
    assert f is not None and f.duration_s > 0 and f.n_chunks >= 1 and f.rms_dbfs < 0
    assert f.speech_ratio is None  # fixed chunking: no VAD, no speech estimate
    first = f.computed_at
    sub2 = _post(client, key, model_version="stub-prepared", config=FIXED)
    _wait_done(client, key, sub2["run_id"])
    _events(client, sub2["run_id"])  # the second run has finished its post-processing
    time.sleep(0.1)
    with client.app.state.session_factory() as s:
        audio2 = s.get(TranscriptionRun, sub2["run_id"]).audio_id
        assert s.get(AudioFeatures, audio_id).computed_at == first
        assert s.get(AudioFeatures, audio2) is not None or audio2 != audio_id


# ── audio features ────────────────────────────────────────────────────────────
def test_feature_definitions():
    sr = 16000
    rng = np.random.default_rng(0)
    noise = 0.001 * rng.standard_normal(sr * 10).astype("float32")
    x = noise.copy()
    t = np.arange(sr * 4) / sr
    x[sr * 2: sr * 6] += (0.3 * np.sin(2 * np.pi * 300 * t)).astype("float32")  # 4 s of "speech"
    f = compute_features(x, sr, [(2.0, 6.0)], vad=True)
    assert f["duration_s"] == 10.0 and f["n_chunks"] == 1 and f["padding_ratio"] == 3.0
    assert abs(f["speech_ratio"] - 0.4) < 1e-3 and f["n_speech_segments"] == 1
    assert 40 < f["snr_db"] < 50  # 0.3/sqrt(2) vs 0.001 -> ~46.5 dB
    assert f["clip_ratio"] == 0.0 and f["peak_dbfs"] < 0
    assert 250 < f["spectral_centroid_hz"] < 2000
    # force-cut pieces of one segment count as one segment
    assert compute_features(x, sr, [(2.0, 4.0), (4.0, 6.0)], vad=True)["n_speech_segments"] == 1


# ── faster-whisper internals ──────────────────────────────────────────────────
class _FE:
    sampling_rate, nb_max_frames = 16000, 3000

    def __call__(self, audio, **kw):
        return np.zeros((80, 101))


class _InternalsModel:
    """Mimics faster-whisper: transcribe() runs feature_extractor, encode, generate_with_fallback internally."""

    def __init__(self):
        self.feature_extractor = _FE()

    def encode(self, features):
        return "enc"

    def detect_language(self, *a, **k):
        return ("en", 0.99, [])

    def generate_with_fallback(self, encoder_output, prompt, tokenizer, options):
        result = SimpleNamespace(sequences_ids=[[1, 2, 3, 4]], no_speech_prob=0.01)
        return result, -0.25, 0.0, 1.3

    def transcribe(self, stream, **kw):
        audio = np.zeros(16000, dtype="float32")
        feats = self.feature_extractor(audio)
        enc = self.encode(np.zeros((80, 3000)))
        self.generate_with_fallback(enc, [50258, 50259], None, SimpleNamespace(beam_size=5))
        return [SimpleNamespace(text=" hello", temperature=0.0)], None


def test_model_internals_become_spans(monkeypatch):
    model = fw.instrument_model(_InternalsModel())
    assert fw.instrument_model(model) is model  # idempotent
    monkeypatch.setattr(fw, "_get_model", lambda *a: model)
    cfg = RunConfig.model_validate({"decoding": {"language": "en"},
                                    "model_options": {"device": "cpu", "compute_type": "int8"}})
    trace = Trace("r2")
    with activate(trace):
        text = fw.FasterWhisperWrapper("tiny").transcribe(io.BytesIO(make_wav(1, 16000)), cfg, None)
    assert text == "hello"
    rows = {r["stage"]: r for r in trace.rows()}
    assert {"model.get", "chunk", "model.features", "model.encode", "model.decode"} <= set(rows)
    assert rows["model.encode"]["attrs"]["frames"] == 3000
    assert rows["model.features"]["attrs"] == {"audio_s": 1.0, "frames": 101}
    d = rows["model.decode"]["attrs"]
    assert d["tokens"] == 4 and d["avg_logprob"] == -0.25 and d["compression_ratio"] == 1.3
    assert d["prompt_tokens"] == 2 and d["beam_size"] == 5
    assert rows["model.encode"]["parent_seq"] == rows["chunk"]["seq"]
