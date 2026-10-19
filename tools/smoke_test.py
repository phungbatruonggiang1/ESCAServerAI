"""End-to-end check against a running server.

    python tools/smoke_test.py http://localhost:8099 /path/to/reference/repos

Exercises exactly what the board does: fetch the contract, send a clip as a
WAV and again as raw PCM, be refused at the wrong rate, upload a labelled
clip. Exits non-zero if any of it misbehaves.
"""
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

base = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8099"
repos = Path(sys.argv[2]) if len(sys.argv) > 2 else None

failures = []


def check(ok: bool, what: str, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}: {what}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(what)


def call(path, data=None, headers=None, method=None):
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}
    except Exception as e:  # noqa: BLE001
        return 0, {"detail": str(e)}


print("contract")
status, spec = call("/v1/anomaly/model")
check(status == 200, "GET /v1/anomaly/model answers")
check(spec.get("sample_rate") == 44100, "declares 44100 Hz", str(spec.get("sample_rate")))
check(spec.get("samples") == 88200, "declares 88200 samples per clip")

if repos:
    anomalous = sorted((repos / "ESCA_Anomaly_Detection/inference/data").glob("*.wav"))
    normal = sorted((repos / "ESCA_Qt/ESCA/database").glob("basefile17426581*.wav"))

    if anomalous:
        wav = anomalous[0].read_bytes()
        print("inference")
        status, verdict = call("/v1/anomaly/infer", wav, {"Content-Type": "audio/wav"})
        check(status == 200 and verdict.get("anomaly") is True,
              "a known anomalous clip is flagged", f"score {verdict.get('score')}")

        status, verdict = call("/v1/anomaly/infer", wav[44:],
                               {"Content-Type": "application/octet-stream",
                                "X-Sample-Rate": "44100", "X-Sequence": "77"})
        check(status == 200 and verdict.get("sequence") == 77,
              "raw PCM works and echoes the sequence")

        status, body = call("/v1/anomaly/infer", wav[44:],
                            {"Content-Type": "application/octet-stream",
                             "X-Sample-Rate": "16000"})
        check(status == 400 and "44100" in str(body.get("detail", "")),
              "the wrong sample rate is refused, not resampled")

        status, body = call("/v1/anomaly/infer", wav[44:],
                            {"Content-Type": "application/octet-stream"})
        check(status == 400, "raw PCM without a declared rate is refused")

        status, body = call("/v1/anomaly/infer", wav[:4096],
                            {"Content-Type": "audio/wav"})
        check(status == 400, "a clip of the wrong length is refused")

    if normal:
        status, verdict = call("/v1/anomaly/infer", normal[0].read_bytes(),
                               {"Content-Type": "audio/wav"})
        check(status == 200 and verdict.get("anomaly") is False,
              "a known normal clip is not flagged", f"score {verdict.get('score')}")

    if anomalous:
        print("corpus")
        before = call("/v1/data/stats")[1].get("clips", 0)
        machine = "smoke-" + uuid.uuid4().hex[:6]
        body, boundary = b"", uuid.uuid4().hex
        payload = anomalous[0].read_bytes()
        parts = [("machine", machine), ("label", "normal"), ("device", "hw:9,0")]
        for name, value in parts:
            body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\""
                     f"\r\n\r\n{value}\r\n").encode()
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                 f"filename=\"clip.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
        body += payload + f"\r\n--{boundary}--\r\n".encode()

        status, stored = call("/v1/data/upload", body,
                              {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        check(status == 200 and machine in stored.get("stored", ""),
              "a labelled clip is stored under its machine")
        after = call("/v1/data/stats")[1].get("clips", 0)
        check(after == before + 1, "the manifest grew by one")

print("speech")
import urllib.request as _u

def post_raw(path, data, ctype):
    req = _u.Request(base + path, data=data, headers={"Content-Type": ctype}, method="POST")
    try:
        with _u.urlopen(req, timeout=600) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:  # noqa: BLE001
        return 0, str(e).encode()

spoken = b"Gear one is running normally."
status, wav = post_raw("/v1/tts", json.dumps({"text": spoken.decode()}).encode(),
                       "application/json")
check(status == 200 and wav[:4] == b"RIFF", "TTS returns a playable WAV",
      f"{len(wav)} bytes")

if status == 200:
    # Round trip: say it, then listen to what was said. Catches the whole
    # chain -- synthesis, the WAV header, resampling, transcription -- in
    # one check, and needs no recorded sample to exist.
    status, body = post_raw("/v1/stt", wav, "audio/wav")
    heard = json.loads(body).get("text", "") if status == 200 else ""
    check(status == 200 and "running normally" in heard.lower(),
          "STT transcribes what TTS just said", heard)

check(call("/v1/tts", json.dumps({"text": ""}).encode(),
           {"Content-Type": "application/json"})[0] == 400,
      "empty text is refused")
check(post_raw("/v1/stt", b"RIFF" + b"\0" * 100, "audio/wav")[0] == 400,
      "a clip too short for speech is refused")

print()
if failures:
    print(f"{len(failures)} check(s) failed")
    sys.exit(1)
print("all checks passed")
