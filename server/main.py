"""The ESCA AI server.

The Raspberry Pi records and acts; everything that needs a model happens
here. Three jobs: say what format the board should send, score two-second
clips for anomalies, and take in labelled clips for the next round of
training. Speech in and out are declared but not yet implemented.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response

from .anomaly import AnomalyDetector
from .audio import AudioError, decode, fit_to_length, from_wav
from .features import FeatureSpec
from .speech import SpeechToText, TextToSpeech
from .storage import DatasetStore

HERE = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(os.environ.get("ESCA_MODEL", HERE / "models" / "vqvae32.onnx"))
DATA_ROOT = Path(os.environ.get("ESCA_DATA", HERE / "data"))

# Measured, not inherited. The number in result2/metrics_detail.json
# (5.6e-05) is four orders of magnitude below anything this model actually
# produces, and inference.py carries a third value (0.00263) belonging to an
# earlier model; either would call every clip anomalous.
#
# Scoring every clip in the two reference repositories through this exact
# pipeline gives anomalous 0.408..0.428 and normal 0.277..0.413, which
# tools/calibrate.py separates best at 0.290 (F1 0.98).
#
# Treat this as a starting point, not a calibration: it rests on 29
# anomalous and 5 normal clips, and one of those 5 sits with the anomalies.
# A threshold meant for production is measured on a normal corpus recorded
# from the machine being watched, which is what the collection work is for.
# 0.35 rather than the 0.290379 that maximises F1: that value is itself one
# of the clip scores, leaving a clip sitting exactly on the boundary. 0.35
# sits in the empty space between the normal cluster (0.277..0.290) and the
# anomalous one (0.408..0.428) and scores the same F1 with room either side.
THRESHOLD = float(os.environ.get("ESCA_THRESHOLD", "0.35"))

# How far a clip may be from the expected length before it is refused rather
# than padded. A chunk boundary costs a few hundred samples; anything beyond
# a tenth of a second means the caller sent the wrong thing.
LENGTH_SLACK = 0.1

STT_MODEL = os.environ.get("ESCA_STT_MODEL", "base.en")
TTS_VOICE = os.environ.get("ESCA_TTS_VOICE", "en_US-lessac-medium")

app = FastAPI(title="ESCA AI Server", version="0.1.0")

# Both load on first use. They are hundreds of megabytes fetched from the
# internet, so a server that has never spoken should still start, score
# anomalies and collect data.
stt = SpeechToText(STT_MODEL)
tts = TextToSpeech(TTS_VOICE)

spec = FeatureSpec()
store = DatasetStore(DATA_ROOT)
detector: AnomalyDetector | None = None
model_error: str | None = None

try:
    detector = AnomalyDetector(MODEL_PATH, THRESHOLD, spec, threads=int(
        os.environ.get("ESCA_THREADS", "2")))
except Exception as exc:  # noqa: BLE001 - reported through /v1/health
    # A missing model must not stop the server: data collection is useful on
    # its own, and the board should get a clear answer rather than a refused
    # connection.
    model_error = f"{type(exc).__name__}: {exc}"


@app.get("/v1/health")
def health() -> dict:
    return {
        "status": "ok" if detector else "degraded",
        "anomaly_model": detector.describe() if detector else None,
        "model_error": model_error,
        "stt": stt.status(),
        "tts": tts.status(),
        "dataset": store.stats(),
    }


@app.get("/v1/anomaly/model")
def anomaly_model() -> dict:
    """The contract the board configures itself from.

    Published rather than hard-coded on both sides, because the corpus is
    44.1 kHz while one config file in the training repository says 16 kHz,
    and a disagreement there produces confident nonsense instead of an error.
    """
    if not detector:
        raise HTTPException(503, detail=model_error or "no model loaded")
    return detector.describe()


@app.post("/v1/anomaly/infer")
async def anomaly_infer(
    request: Request,
    x_sample_rate: str | None = Header(default=None),
    x_sequence: str | None = Header(default=None),
) -> JSONResponse:
    """Score one clip. Body is a WAV, or raw s16le with X-Sample-Rate set."""
    if not detector:
        raise HTTPException(503, detail=model_error or "no model loaded")

    payload = await request.body()
    if not payload:
        raise HTTPException(400, detail="empty body")

    declared = int(x_sample_rate) if x_sample_rate else None
    try:
        pcm, rate = decode(payload, request.headers.get("content-type", ""), declared)
    except AudioError as exc:
        raise HTTPException(400, detail=str(exc)) from exc

    if rate != spec.sample_rate:
        # Refused rather than resampled. The filterbank settings are tied to
        # the rate, so the wrong one would score confidently and wrongly;
        # better that the board learns it is misconfigured.
        raise HTTPException(
            400,
            detail=(
                f"this model expects {spec.sample_rate} Hz, got {rate}. "
                f"Fetch /v1/anomaly/model and record at that rate."
            ),
        )

    drift = abs(len(pcm) - spec.samples) / spec.sample_rate
    if drift > LENGTH_SLACK:
        raise HTTPException(
            400,
            detail=(
                f"expected about {spec.seconds}s ({spec.samples} samples), "
                f"got {len(pcm)} samples"
            ),
        )

    try:
        verdict = detector.score(fit_to_length(pcm, spec.samples))
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc

    return JSONResponse(
        {
            "anomaly": verdict.anomaly,
            "score": verdict.score,
            "threshold": verdict.threshold,
            "inference_ms": round(verdict.inference_ms, 2),
            "sequence": int(x_sequence) if x_sequence else None,
        }
    )


@app.post("/v1/data/upload")
async def data_upload(
    file: UploadFile = File(...),
    machine: str = Form("unknown"),
    label: str = Form("unlabelled"),
    device: str = Form(""),
    started_ns: int = Form(0),
    dropped: int = Form(0),
) -> dict:
    """Take one labelled clip for the training corpus.

    Multipart rather than a raw body: a clip is useless without its label,
    and one request that cannot carry half of the pair is easier to get
    right than two that can disagree.
    """
    payload = await file.read()
    if not payload:
        raise HTTPException(400, detail="empty file")

    try:
        pcm, rate = from_wav(payload)
    except AudioError as exc:
        raise HTTPException(400, detail=f"upload must be a WAV: {exc}") from exc

    meta = store.save(
        payload,
        machine=machine,
        label=label,
        sample_rate=rate,
        channels=1,
        seconds=round(len(pcm) / rate, 3) if rate else 0.0,
        started_ns=started_ns,
        device=device,
        dropped=dropped,
    )

    # The board deletes its local copy on the strength of this reply, so it
    # only comes after the bytes and the manifest line are both written.
    return {"stored": meta.path, "bytes": meta.bytes, "seconds": meta.seconds}


@app.get("/v1/data/stats")
def data_stats() -> dict:
    return store.stats()


@app.post("/v1/stt")
async def speech_to_text(request: Request,
                         x_sample_rate: str | None = Header(default=None)) -> JSONResponse:
    """A WAV in, what was said out. Any sample rate -- resampled here."""
    payload = await request.body()
    if not payload:
        raise HTTPException(400, detail="empty body")

    declared = int(x_sample_rate) if x_sample_rate else None
    try:
        pcm, rate = decode(payload, request.headers.get("content-type", ""), declared)
    except AudioError as exc:
        raise HTTPException(400, detail=str(exc)) from exc

    if len(pcm) < rate * 0.2:
        # Shorter than a word. The board's own detector should have caught
        # this, so saying so is more useful than returning an empty string
        # that looks like a failed transcription.
        raise HTTPException(400, detail="clip is too short to contain speech")

    try:
        result = stt.transcribe(pcm, rate)
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc)) from exc

    return JSONResponse({
        "text": result.text,
        "language": result.language,
        "confidence": result.confidence,
        "ms": round(result.ms, 1),
        "seconds": round(len(pcm) / rate, 2),
    })


@app.post("/v1/tts")
async def text_to_speech(request: Request) -> Response:
    """Text in, WAV out. WAV because the board has no mp3 decoder."""
    try:
        body = json.loads(await request.body() or b"{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(400, detail="body must be JSON") from exc

    text = str(body.get("text", "")).strip()
    if not text:
        raise HTTPException(400, detail="nothing to say")
    if len(text) > 1000:
        raise HTTPException(400, detail="text is too long; split it into sentences")

    try:
        wav = tts.synthesise(text)
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc)) from exc

    return Response(content=wav, media_type="audio/wav")
