"""Part H scripts: key rotation, LibriSpeech import with utterance IDs (and backfill), the baseline summary,
export and the paired-bootstrap comparison. SQLite + local storage; no model needed."""
import gzip
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy import select

from app.db.models import Audio, Base, Dataset, ReferenceTranscript, RunStatus, TranscriptionRun, User
from app.db.repositories import RunRepo, UserRepo, hash_api_key
from app.db.session import make_engine, make_session_factory
from app.evaluation.metrics import compute_metrics
from app.evaluation.summary import MixedNormalization, summarize
from app.storage.local import LocalDiskStorage
from scripts import compare_runs, export_runs, import_librispeech, rotate_keys, run_baseline


@pytest.fixture
def sf_(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    Base.metadata.create_all(engine)
    return make_session_factory(engine)


# ── rotate_keys ──────────────────────────────────────────────────────────────────────────────────────────────

def test_rotate_keys_replaces_every_key_in_place(sf_):
    with sf_() as s:
        old = {name: UserRepo(s).create(name)[1] for name in ("alice", "loaduser0", "loadtest_user_1")}
        s.commit()
    load_keys, others = rotate_keys.rotate(sf_, rotate_keys.LOAD_TEST_PREFIXES)
    assert len(load_keys) == 2 and set(others) == {"alice"}
    with sf_() as s:
        repo = UserRepo(s)
        assert all(repo.get_by_api_key(k) is None for k in old.values())          # leaked keys are dead
        assert repo.get_by_api_key(others["alice"]).username == "alice"           # same user row, new key
        assert {repo.get_by_api_key(k).username for k in load_keys} == {"loaduser0", "loadtest_user_1"}
        assert len(s.scalars(select(User)).all()) == 3


# ── import_librispeech ───────────────────────────────────────────────────────────────────────────────────────

def _fake_split(root, utterances: dict[str, str]):
    """LibriSpeech layout: <split>/<speaker>/<chapter>/<utt>.flac + <speaker>-<chapter>.trans.txt"""
    chapter = root / "test-clean" / "1089" / "134686"
    chapter.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for utt in utterances:
        sf.write(chapter / f"{utt}.flac", (0.1 * rng.standard_normal(16000)).astype("float32"), 16000)
    (chapter / "1089-134686.trans.txt").write_text("".join(f"{u} {t}\n" for u, t in utterances.items()))
    return root / "test-clean"


def _owner(sf_):
    with sf_() as s:
        user, _ = UserRepo(s).create("alice")
        s.commit()
        return user.id


def test_import_sets_utterance_ids_and_is_idempotent(sf_, tmp_path):
    split = _fake_split(tmp_path, {"1089-134686-0000": "HELLO WORLD", "1089-134686-0001": "TEN APPLES"})
    storage, uid = LocalDiskStorage(str(tmp_path / "blobs")), _owner(sf_)
    name = import_librispeech.default_dataset_name(split)
    assert name == "librispeech-test-clean"
    first = import_librispeech.import_split(sf_, storage, split, name, uid)
    assert (first["imported"], first["skipped"]) == (2, 0)
    again = import_librispeech.import_split(sf_, storage, split, name, uid)
    assert (again["imported"], again["skipped"], again["backfilled"]) == (0, 2, 0)
    with sf_() as s:
        assert sorted(s.scalars(select(Audio.external_id))) == ["1089-134686-0000", "1089-134686-0001"]


def test_the_same_files_in_another_dataset_get_their_own_rows(sf_, tmp_path):
    """A profiling subset is its own dataset: same audio, new rows (and so fresh runs), shared derived cache."""
    split = _fake_split(tmp_path, {"1089-134686-0000": "HELLO WORLD", "1089-134686-0001": "TEN APPLES"})
    storage, uid = LocalDiskStorage(str(tmp_path / "blobs")), _owner(sf_)
    import_librispeech.import_split(sf_, storage, split, "librispeech-test-clean", uid)
    sub = import_librispeech.import_split(sf_, storage, split, "librispeech-profile", uid, limit=1)
    assert (sub["imported"], sub["skipped"]) == (1, 0)
    with sf_() as s:
        rows = s.execute(select(Dataset.name, Audio.external_id, Audio.checksum)
                         .join(Dataset, Dataset.id == Audio.dataset_id)).all()
    by_ds = {}
    for name, utt, checksum in rows:
        by_ds.setdefault(name, {})[utt] = checksum
    assert len(by_ds["librispeech-test-clean"]) == 2 and len(by_ds["librispeech-profile"]) == 1
    utt, checksum = next(iter(by_ds["librispeech-profile"].items()))
    assert by_ds["librispeech-test-clean"][utt] == checksum


def test_reimport_backfills_missing_utterance_ids(sf_, tmp_path):
    split = _fake_split(tmp_path, {"1089-134686-0000": "HELLO WORLD", "1089-134686-0001": "TEN APPLES"})
    storage, uid = LocalDiskStorage(str(tmp_path / "blobs")), _owner(sf_)
    import_librispeech.import_split(sf_, storage, split, "librispeech-test-clean", uid)
    with sf_() as s:  # as imported before migration 0005
        for a in s.scalars(select(Audio)):
            a.external_id = None
        s.commit()
    res = import_librispeech.import_split(sf_, storage, split, "librispeech-test-clean", uid)
    assert (res["imported"], res["backfilled"]) == (0, 2)
    with sf_() as s:
        assert None not in s.scalars(select(Audio.external_id)).all()


def test_backfill_falls_back_to_reference_and_duration_when_checksums_changed(sf_, tmp_path):
    split = _fake_split(tmp_path, {"1089-134686-0000": "HELLO WORLD"})
    storage, uid = LocalDiskStorage(str(tmp_path / "blobs")), _owner(sf_)
    import_librispeech.import_split(sf_, storage, split, "librispeech-test-clean", uid)
    with sf_() as s:  # pretend an older encoder produced different bytes
        a = s.scalars(select(Audio)).one()
        a.external_id, a.checksum = None, "0" * 64
        s.commit()
    res = import_librispeech.import_split(sf_, storage, split, "librispeech-test-clean", uid)
    assert (res["imported"], res["backfilled"]) == (0, 1)


# ── baseline config and summary ──────────────────────────────────────────────────────────────────────────────

def test_baseline_cli_flags_override_the_json_config():
    a = SimpleNamespace(config='{"decoding": {"beam_size": 5}, "model_options": {"device": "cpu"}}',
                        device="cuda", compute_type="float16", beam_size=1, temperature=None)
    raw = run_baseline.build_raw_config(a)
    assert raw["model_options"] == {"device": "cuda", "compute_type": "float16"}
    assert raw["decoding"] == {"beam_size": 1}


def _row(ref, hyp, latency=1.0, duration=4.0, start=0.0, inference=0.5, norm="whisper-en-v1"):
    m = compute_metrics(ref, hyp, norm)
    t0 = datetime(2026, 11, 10, tzinfo=timezone.utc) + timedelta(seconds=start)
    return {"status": "done", "wer": m["wer"], "metrics": m, "latency": latency, "duration": duration,
            "started_at": t0, "profile": {"timings": {"prepare_s": 0.1, "inference_s": inference, "score_s": 0.01},
                                          "cache_hits": {"chunks": True}, "runtime": {"device": "cpu"}}}


def test_summary_is_micro_averaged_with_stage_and_throughput_figures():
    rows = [_row("a b c d e", "a b c d e", start=0), _row("x", "x y z w", start=1),
            {"status": "failed", "wer": None, "metrics": None, "latency": 0.2, "duration": 1.0,
             "started_at": None, "profile": None}]
    s = summarize(rows)
    assert (s["runs_done"], s["runs_failed"]) == (2, 1)
    assert s["wer"]["micro"] == pytest.approx(round(3 / 6, 4))                   # not the mean of 0 and 3
    assert s["errors"] == {"substitutions": 0, "deletions": 0, "insertions": 3}
    assert s["inference_rtf"]["mean"] == pytest.approx(0.125)
    assert s["throughput"]["audio_hours_per_wall_hour"] == pytest.approx(8 / 2)   # 8 s audio in 2 s wall
    assert s["normalization"] == "whisper-en-v1" and s["cache_hit_rate"]["chunks"] == 1.0


def test_summary_counts_loops_separately_from_wer():
    loop = _row("they knew no north", "they knew north north north north north north north north")
    loop["profile"]["decoder"] = {"segments": 2, "fallback_segments": 0, "longest_repeat": 8}
    ok = _row("a b c", "a b c")
    ok["profile"]["decoder"] = {"segments": 1, "fallback_segments": 1, "longest_repeat": 1}
    loops = summarize([loop, ok])["loops"]
    assert loops == {"runaway_files": 1, "repeat_files": 1, "longest_repeat_max": 8, "fallback_files": 1,
                     "fallback_rate": round(1 / 3, 4)}


def test_summary_refuses_mixed_normalizations():
    with pytest.raises(MixedNormalization):
        summarize([_row("a", "a"), _row("a", "a", norm="basic-v1")])


# ── export and compare ───────────────────────────────────────────────────────────────────────────────────────

def _seed_runs(sf_, hyps: dict[str, str], cfg_hash="h" * 64):
    with sf_() as s:
        user, _ = UserRepo(s).create("alice")
        ds = Dataset(user_id=user.id, name="librispeech-test-clean")
        s.add(ds)
        s.flush()
        for i, (utt, (ref, hyp)) in enumerate(hyps.items()):
            a = Audio(user_id=user.id, storage_key=f"audio/x/{i}.wav", checksum=f"{i:064d}", duration=4.0,
                      original_sample_rate=16000, size_bytes=1, dataset_id=ds.id, external_id=utt)
            s.add(a)
            s.flush()
            s.add(ReferenceTranscript(audio_id=a.id, text=ref))
            run = RunRepo(s).add(a.id, "faster-whisper-tiny", {"model_options": {"device": "cpu"}}, cfg_hash)
            m = compute_metrics(ref, hyp)
            run.status, run.transcript, run.latency, run.wer, run.metrics = RunStatus.done, hyp, 1.0, m["wer"], m
            run.profile = {"timings": {"inference_s": 0.5}}
        s.commit()
        return ds.id


def test_export_writes_one_line_per_file_keyed_by_utterance(sf_, tmp_path):
    ds_id = _seed_runs(sf_, {"u1": ("the cat sat", "the cat sat"), "u2": ("ten dogs", "10 dog")})
    with sf_() as s:
        assert export_runs.config_hashes(s, ds_id, "faster-whisper-tiny") == {"h" * 64: 2}
        rows, config = export_runs.collect(s, ds_id, "faster-whisper-tiny", "h" * 64)
    assert [r["utt_id"] for r in rows] == ["u1", "u2"] and config == {"model_options": {"device": "cpu"}}
    assert rows[1]["word_substitutions"] == 1 and rows[1]["normalization"] == "whisper-en-v1"
    assert summarize(export_runs.summary_rows(rows))["wer"]["micro"] == pytest.approx(round(1 / 5, 4))


def _export(path, label, rows, normalization="whisper-en-v1", concurrency=1):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    path.with_name(path.name.replace(".jsonl.gz", ".meta.json")).write_text(json.dumps(
        {"machine": label, "normalization": normalization, "dataset": "librispeech-test-clean",
         "worker_concurrency": concurrency}))
    return path


def _erow(utt, ref_words, errors, norm="whisper-en-v1", latency=1.0):
    return {"utt_id": utt, "status": "done", "wer": errors / ref_words, "ref_words": ref_words,
            "word_substitutions": errors, "word_deletions": 0, "word_insertions": 0, "normalization": norm,
            "reference": f"ref {utt}", "duration_s": 4.0, "latency_s": latency,
            "profile": {"timings": {"inference_s": latency / 2}}}


def test_identical_exports_show_no_difference(tmp_path):
    rows = [_erow(f"u{i}", 20, i % 3) for i in range(200)]
    a = _export(tmp_path / "a.jsonl.gz", "mac-cpu", rows)
    b = _export(tmp_path / "b.jsonl.gz", "rtx3050", rows)
    res = compare_runs.compare(*compare_runs.load(a), *compare_runs.load(b), iters=2000)
    assert res["diff"] == 0 and res["ci95"][0] <= 0 <= res["ci95"][1] and res["p"] == 1.0
    assert "no difference shown" in compare_runs.render(res)


def test_a_real_difference_is_detected(tmp_path):
    rng = np.random.default_rng(3)
    base = [int(x) for x in rng.integers(0, 3, 300)]
    a = _export(tmp_path / "a.jsonl.gz", "A", [_erow(f"u{i}", 20, e) for i, e in enumerate(base)])
    b = _export(tmp_path / "b.jsonl.gz", "B", [_erow(f"u{i}", 20, e + (i % 2), latency=0.5)
                                              for i, e in enumerate(base)])
    res = compare_runs.compare(*compare_runs.load(a), *compare_runs.load(b), iters=2000)
    assert res["diff"] == pytest.approx(150 / 6000) and res["ci95"][0] > 0 and res["p"] < 0.01
    assert res["speed"]["b"]["rtf_mean"] == pytest.approx(0.125)
    assert "B is worse" in compare_runs.render(res)


def test_comparison_refuses_mixed_normalizations_and_warns_on_gaps(tmp_path):
    a = _export(tmp_path / "a.jsonl.gz", "A", [_erow("u1", 10, 1)])
    b = _export(tmp_path / "b.jsonl.gz", "B", [_erow("u1", 10, 1, norm="basic-v1")], normalization="basic-v1")
    with pytest.raises(ValueError, match="normalizations"):
        compare_runs.compare(*compare_runs.load(a), *compare_runs.load(b))
    c = _export(tmp_path / "c.jsonl.gz", "C", [_erow("u1", 10, 1), _erow("u2", 10, 0)], concurrency=8)
    res = compare_runs.compare(*compare_runs.load(a), *compare_runs.load(c), iters=200)
    assert res["files"] == 1 and res["speed"] is None
    assert any("only in B" in w for w in res["warnings"]) and any("concurrency" in w for w in res["warnings"])
