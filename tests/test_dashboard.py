"""Dashboard: the page loads, groups and rows reflect traced runs, exports appear beside live data, the live
stream sends events, and the token guard works."""
import gzip
import json
import time

import pytest
from fastapi.testclient import TestClient

from app.db.models import Base
from app.db.repositories import UserRepo
from app.main import create_app
from tests.test_evaluation import _post, _wait_done

FIXED = json.dumps({"chunking": {"strategy": "fixed"}})


def _client(settings, **overrides):
    app = create_app(settings.model_copy(update=overrides))
    Base.metadata.create_all(app.state.engine)
    return TestClient(app)


def _user(c):
    with c.app.state.session_factory() as s:
        _, key = UserRepo(s).create("alice")
        s.commit()
    return key


@pytest.fixture
def dash(settings, tmp_path):
    exports = tmp_path / "exports"
    exports.mkdir()
    rows = [{"run_id": f"x{i}", "utt_id": f"u-{i}", "status": "done", "wer": 0.1, "ref_words": 10,
             "word_substitutions": 1, "word_deletions": 0, "word_insertions": 0, "latency_s": 0.5 + i,
             "duration_s": 5.0, "profile": {"timings": {"prepare_s": 0.1, "inference_s": 0.3, "score_s": 0.01,
                                                        "total_s": 0.5}, "n_chunks": 2,
                                            "runtime": {"device": "cuda", "compute_type": "float16"}}}
            for i in range(3)]
    with gzip.open(exports / "gpu_faster-whisper-tiny_abc_whisper-en-v1.jsonl.gz", "wt") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    (exports / "gpu_faster-whisper-tiny_abc_whisper-en-v1.meta.json").write_text(json.dumps(
        {"model": "faster-whisper-tiny", "machine": "rtx3050", "config_hash": "abc" * 21 + "a",
         "normalization": "whisper-en-v1", "dataset": "d", "config": {"model_options": {"device": "cuda"}}}))
    with _client(settings, dashboard_exports_dir=str(exports)) as c:
        yield c


def _wait_features(c, timeout=3.0):
    """Features and stage events are written just after the run is marked done."""
    from app.db.models import AudioFeatures
    end = time.time() + timeout
    while time.time() < end:
        with c.app.state.session_factory() as s:
            if s.query(AudioFeatures).count():
                return
        time.sleep(0.02)
    raise AssertionError("no audio features")


def test_page_and_static_assets_are_served(dash):
    page = dash.get("/dashboard")
    assert page.status_code == 200 and "ASR Testbed" in page.text
    assert dash.get("/dashboard/static/echarts.min.js").status_code == 200
    assert dash.get("/dashboard/static/..%2Fdata.py").status_code == 404


def test_groups_rows_stages_and_run_detail_follow_a_traced_run(dash):
    key = _user(dash)
    sub = _post(dash, key, reference="this is a dummy transcript", model_version="stub-prepared", config=FIXED,
                dataset="d")
    assert _wait_done(dash, key, sub["run_id"])["status"] == "done"
    _wait_features(dash)
    groups = dash.get("/dashboard/api/groups").json()
    db = next(g for g in groups if g["src"] == "db")
    assert db["traced"] and db["done"] == 1 and db["wer"] is not None and db["model"] == "stub-prepared"
    exp = next(g for g in groups if g["src"] == "export")
    assert exp["device"] == "cuda" and exp["n"] == 3 and abs(exp["wer"] - 0.1) < 1e-9 and not exp["duplicate"]

    rows = dash.get("/dashboard/api/runs", params={"group": [db["key"], exp["key"]]}).json()
    r = rows[db["key"]][0]
    assert r["id"] == sub["run_id"] and r["stages"]["transcribe"] >= 0.04 and r["queue_ms"] is not None
    assert r["f_duration_s"] > 0  # audio features joined in
    assert [x["utt"] for x in rows[exp["key"]]] == ["u-0", "u-1", "u-2"]

    st = dash.get("/dashboard/api/stages", params={"group": [db["key"], exp["key"]]}).json()
    assert st[db["key"]]["basis"] == "traced" and abs(sum(s["share"] or 0 for s in st[db["key"]]["stages"]) - 1) < 1e-3
    assert any(s["stage"] == "transcribe" and s["calls"] == 1 for s in st[db["key"]]["stages"])
    assert {s["stage"] for s in st[exp["key"]]["stages"]} == {"prepare", "inference", "score", "other"}

    d = dash.get(f"/dashboard/api/run/{sub['run_id']}").json()
    assert d["reference"] == "this is a dummy transcript" and {e["stage"] for e in d["events"]} >= {"run", "score"}
    assert dash.get("/dashboard/api/run/nope").status_code == 404
    assert dash.get("/dashboard/api/runs", params={"group": "db|x|y|z|w"}).status_code in (200, 404)
    assert dash.get("/dashboard/api/runs", params={"group": "export|../../etc.jsonl.gz"}).status_code == 404


def test_live_stream_sends_events(dash):
    key = _user(dash)
    sub = _post(dash, key, model_version="stub-prepared", config=FIXED)
    _wait_done(dash, key, sub["run_id"])
    _wait_features(dash)
    body = dash.get("/dashboard/api/stream", params={"once": 1}).text
    assert body.startswith("data: ")
    payload = json.loads(body[len("data: "):].strip())
    assert payload["counts"]["done"] == 1 and sub["run_id"] in payload["runs"]
    assert any(e["stage"] == "run" for e in payload["events"])
    again = json.loads(dash.get("/dashboard/api/stream", params={"once": 1, "since": payload["last_id"]})
                       .text[len("data: "):].strip())
    assert again["events"] == [] and again["last_id"] == payload["last_id"]


def test_token_guard_and_disable(settings):
    with _client(settings, dashboard_token="s3cret", dashboard_exports_dir="") as c:
        assert c.get("/dashboard/api/groups").status_code == 401
        assert c.get("/dashboard/api/groups", params={"token": "s3cret"}).status_code == 200
    with _client(settings, dashboard_enabled=False) as c:
        assert c.get("/dashboard").status_code == 404
