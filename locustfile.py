"""Locust load test. Each simulated user owns one API key, submits audio and polls its runs.

Settings (environment variables):
  LOCUST_MODEL   model_version to submit (default "stub": tests the API, queue and database, NOT the model)
  LOCUST_AUDIO   WAV file to upload (default: 0.5 s of silence, generated). For a real-model test, use real
                 speech, e.g. a LibriSpeech utterance converted to WAV: silence produces no VAD chunks, so the
                 model never runs.
  LOCUST_KEYS    JSON list of API keys (default keys.json; written by scripts/rotate_keys.py or create_users.py)

Run headless with a fixed length per stage, so stages are comparable:
  locust -f locustfile.py --headless -u 20 -r 5 -t 5m --host http://localhost:8000 --csv results/CPU/stub_20users
  LOCUST_MODEL=faster-whisper-tiny LOCUST_AUDIO=sample.wav \\
      locust -f locustfile.py --headless -u 20 -r 5 -t 5m --host http://localhost:8001 --csv results/GPU/tiny_20users
For a real model, the numbers that matter are runs completed per second and p95 queue wait (/metrics and the
database), not API latency: the API was never the bottleneck.
"""
import io
import json
import os
import random
import wave
from pathlib import Path

from locust import HttpUser, between, task

MODEL = os.getenv("LOCUST_MODEL", "stub")
KEYS = json.loads(Path(os.getenv("LOCUST_KEYS", "keys.json")).read_text())


def silence_wav(seconds: float = 0.5) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * int(16000 * seconds))  # ~16 KB
    return buf.getvalue()


WAV = Path(os.environ["LOCUST_AUDIO"]).read_bytes() if os.getenv("LOCUST_AUDIO") else silence_wav()


class Uploader(HttpUser):
    wait_time = between(0.5, 2)

    def on_start(self):
        self.h = {"X-API-Key": random.choice(KEYS)}
        self.pending = []

    @task(3)
    def submit(self):
        with self.client.post("/audio", headers=self.h, name="POST /audio", data={"model_version": MODEL},
                              files={"file": ("a.wav", WAV, "audio/wav")}, catch_response=True) as r:
            if r.status_code == 202:
                self.pending.append(r.json()["run_id"])
            else:
                r.failure(f"{r.status_code}: {r.text[:100]}")

    @task(5)
    def poll(self):
        if not self.pending:
            return
        r = self.client.get(f"/runs/{self.pending[0]}", headers=self.h, name="GET /runs/[id]")
        if r.ok and r.json()["status"] in ("done", "failed"):
            self.pending.pop(0)
