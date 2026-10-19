# ESCA AI Server

The models the Raspberry Pi 4 cannot run. The board records, segments and
acts; this scores the audio and takes in the corpus that trains the next
model.

    USB mic → esca-audiod → esca-anomalyd ──HTTP──► this server
                          → esca-recorder → esca-uploader ──┘

## Running it

    pip install -r requirements.txt
    uvicorn server.main:app --host 0.0.0.0 --port 8099

Environment: `ESCA_MODEL` (default `models/vqvae32.onnx`), `ESCA_DATA`
(default `data/`), `ESCA_THRESHOLD`, `ESCA_THREADS`.

A missing model does not stop the server. Collecting data is useful on its
own, and the board gets a clear answer from `/v1/health` rather than a
refused connection.

## Endpoints

| | |
|---|---|
| `GET /v1/health` | status, loaded model, corpus counts |
| `GET /v1/anomaly/model` | the contract the board configures itself from |
| `POST /v1/anomaly/infer` | one clip in, a verdict out |
| `POST /v1/data/upload` | one labelled clip into the corpus |
| `GET /v1/data/stats` | what has been collected |
| `POST /v1/stt`, `POST /v1/tts` | declared, not implemented; the 501 carries the contract |

`/v1/anomaly/infer` takes a WAV, or raw little-endian int16 with
`X-Sample-Rate` set. `X-Sequence` is echoed back so the board can tell which
audio a verdict belongs to.

    curl -X POST --data-binary @clip.wav -H 'Content-Type: audio/wav' \
         http://localhost:8099/v1/anomaly/infer
    {"anomaly":true,"score":0.4199,"threshold":0.35,"inference_ms":85.1,"sequence":null}

## Two things the reference code gets wrong

**The sample rate is 44100, not 16000.** `params.yaml` says 16000 under
`RECORD`, and `tools/rt_test.py` hard-codes 44100 for the same feature.
Every actual recording settles it: each WAV in
`ESCA_Anomaly_Detection/inference/data` is 88200 frames of 44.1 kHz mono,
which is exactly the 176400-byte shared memory block `inference.py` reads.

The gammatone filterbank is tied to its sample rate, so 16 kHz audio through
these settings shifts every band and produces confident nonsense. The server
therefore **refuses** a clip at the wrong rate instead of resampling it, and
publishes the rate it wants at `/v1/anomaly/model` so the board can
configure itself rather than hard-code a second guess.

**The shipped threshold does not match the shipped model.**
`result2/save_parameter/metrics_detail.json` says 5.6e-05 and
`inference.py` says 0.00263. This model produces scores around 0.3 to 0.43,
so either would call everything anomalous.

Scoring every clip in both reference repositories through this exact
pipeline:

| | n | min | median | max |
|---|---|---|---|---|
| anomalous | 29 | 0.408 | 0.418 | 0.428 |
| normal | 5 | 0.277 | 0.282 | 0.413 |

The default is **0.35**, which sits in the gap between the clusters. The
value that maximises F1 is 0.290379, but that is itself one of the clip
scores and leaves a sample exactly on the boundary; 0.35 scores the same
F1 (0.98) with room on either side.

Treat it as a starting point, not a calibration. It rests on 5 normal clips,
one of which scores with the anomalies and is probably mislabelled or from
another machine. A production threshold is measured on a normal corpus
recorded from the machine being watched — which is what the collection work
on the board is for. `tools/calibrate.py` does that measurement.

## How a clip becomes a score

Every constant comes from `python_ai/inference/inference.py`, the code that
produced the scores the threshold was calibrated against. Changing one
without retraining silently invalidates the threshold.

1. 2 s of 44.1 kHz mono int16, 88200 samples.
2. `gtgram(audio, 44100, window=0.12, hop=0.06, bands=32, f_min=100)` → 32×32.
3. `flipud(20·log10(g + 1e-9))`, then clip to 0..1 between −23.1452 and 51.6774 dB.
4. Into the VQ-VAE as NCHW `(1,1,32,32)`.
5. Mean squared error against the reconstruction; above the threshold is an anomaly.

Step 4 and 5 hide a trap: **the exported graph takes NCHW and returns
NHWC**. The input and output shapes are not the same layout, and comparing
them without transposing one gives a plausible, meaningless number.

## Layout

    server/features.py   gammatone and the normalisation bounds
    server/anomaly.py    the ONNX session and the reconstruction error
    server/audio.py      WAV and raw PCM decoding
    server/storage.py    corpus files plus a JSONL manifest
    server/main.py       the endpoints
    tools/calibrate.py   measures a threshold from labelled clips

`server/audio.py` accepts a WAV whose header declares a `data` size of zero.
That is not defensiveness for its own sake: the Qt recorder in `ESCA_Qt`
writes the header and never goes back to patch it, so half the clips in
`ESCA/database` are unreadable by anything stricter.
