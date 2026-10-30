import io, json, random, wave
from locust import HttpUser, task, between

KEYS = json.load(open("keys.json"))

def tiny_wav():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\0\0" * 8000)   # 0.5 s of silence, ~16 KB
    return buf.getvalue()

WAV = tiny_wav()

class Uploader(HttpUser):
    wait_time = between(0.5, 2)

    def on_start(self):
        self.h = {"X-API-Key": random.choice(KEYS)}
        self.pending = []

    @task(3)
    def submit(self):
        with self.client.post("/audio", headers=self.h, name="POST /audio",
                              files={"file": ("a.wav", WAV, "audio/wav")},
                              catch_response=True) as r:
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