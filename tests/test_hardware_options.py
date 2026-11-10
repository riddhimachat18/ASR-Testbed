"""Part H: device and precision are hashed run settings; the wrapper decodes exactly as the config says."""
import io

import pytest

from app.config import Settings
from app.pipeline import faster_whisper_wrapper as fw
from app.pipeline.derived import Chunk
from app.pipeline.prepare import Prepared
from app.pipeline.registry import ModelRegistry
from app.pipeline.run_config import ConfigInvalid, RunConfig, resolve_run_config
from tests.test_derived import wav_bytes

STUB_DEFAULT_HASH = "37ffc2f867748bda61b825b16163e1c112ccb288e3f38ad14fdcbfbf7a6323d7"


def _registry(**kw) -> ModelRegistry:
    return ModelRegistry(Settings(_env_file=None, **kw))


def _resolve(reg: ModelRegistry, raw: dict, model: str = "faster-whisper-tiny"):
    return resolve_run_config(model, raw, reg.options_model(model))


def test_machine_defaults_are_written_into_every_faster_whisper_config():
    cfg, _ = _resolve(_registry(), {})
    assert cfg["model_options"] == {"device": "cpu", "compute_type": "int8"}
    cfg, _ = _resolve(_registry(device="cuda", compute_type="float16"), {})
    assert cfg["model_options"] == {"device": "cuda", "compute_type": "float16"}


def test_cpu_int8_and_gpu_float16_never_share_a_hash():
    _, cpu = _resolve(_registry(), {})
    _, gpu = _resolve(_registry(device="cuda", compute_type="float16"), {})
    _, gpu_int8 = _resolve(_registry(device="cuda", compute_type="float16"), {"model_options": {"compute_type": "int8"}})
    assert len({cpu, gpu, gpu_int8}) == 3


def test_explicit_option_equal_to_the_default_hashes_the_same():
    reg = _registry()
    assert _resolve(reg, {})[1] == _resolve(reg, {"model_options": {"device": "cpu", "compute_type": "int8"}})[1]


@pytest.mark.parametrize("opts", [{"device": "cpu", "compute_type": "float16"},
                                  {"device": "cpu", "compute_type": "int8_float16"},
                                  {"device": "tpu"}, {"compute_type": "int4"}, {"devcie": "cuda"}])
def test_impossible_or_misspelled_options_are_rejected(opts):
    with pytest.raises(ConfigInvalid):
        _resolve(_registry(), {"model_options": opts})


def test_a_bad_machine_default_fails_at_startup():
    with pytest.raises(ValueError):
        _registry(device="cpu", compute_type="float16")


def test_stub_hash_is_unchanged_by_part_h():
    reg = _registry(device="cuda", compute_type="float16")
    assert _resolve(reg, {}, model="stub")[1] == STUB_DEFAULT_HASH


def test_registry_lists_every_size():
    assert {"faster-whisper-tiny", "faster-whisper-base", "faster-whisper-small",
            "faster-whisper-medium"} <= set(_registry().names())


# ── the wrapper, with a fake model recording what it was asked ──────────────────────────────────────────────

class _Seg:
    def __init__(self, text):
        self.text = text


class _FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, stream, **kw):
        self.calls.append(kw)
        return [_Seg(" hello "), _Seg("")], None


@pytest.fixture
def fake(monkeypatch):
    model, loaded = _FakeModel(), []

    def get_model(size, device, compute_type):
        loaded.append((size, device, compute_type))
        return model
    monkeypatch.setattr(fw, "_get_model", get_model)
    return model, loaded


def test_temperature_and_decoding_settings_reach_faster_whisper(fake):
    model, _ = fake
    cfg = RunConfig.model_validate({"decoding": {"temperature": 0.0, "beam_size": 3, "language": "en"},
                                    "model_options": {"device": "cpu", "compute_type": "int8"}})
    text = fw.FasterWhisperWrapper("tiny").transcribe(io.BytesIO(wav_bytes(1, 16000)), cfg, None)
    assert text == "hello"
    kw = model.calls[0]
    assert kw["temperature"] == 0.0 and isinstance(kw["temperature"], float)  # one value: no fallback schedule
    assert (kw["beam_size"], kw["language"], kw["task"], kw["vad_filter"]) == (3, "en", "transcribe", False)


def test_device_and_precision_come_from_the_run_config(fake):
    _, loaded = fake
    cfg = RunConfig.model_validate({"model_options": {"device": "cuda", "compute_type": "float16"}})
    fw.FasterWhisperWrapper("base", device="cpu", compute_type="int8").transcribe(
        io.BytesIO(wav_bytes(1, 16000)), cfg, None)
    assert loaded == [("base", "cuda", "float16")]


def test_legacy_config_without_model_options_uses_the_wrapper_defaults(fake):
    _, loaded = fake
    fw.FasterWhisperWrapper("tiny", device="cpu", compute_type="int8").transcribe(
        io.BytesIO(wav_bytes(1, 16000)), RunConfig.model_validate({}), None)
    assert loaded == [("tiny", "cpu", "int8")]


def test_each_chunk_is_transcribed_and_stitched(fake):
    model, _ = fake
    prepared = Prepared(16000, [Chunk(0.0, 0.5), Chunk(0.5, 1.0)])
    text = fw.FasterWhisperWrapper("tiny").transcribe(io.BytesIO(wav_bytes(1, 16000)), RunConfig(), prepared)
    assert text == "hello hello" and len(model.calls) == 2
