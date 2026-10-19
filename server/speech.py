"""Speech in and out.

Both models are loaded lazily and both failures are survivable. A board that
cannot transcribe should still be able to detect anomalies, and a missing
voice should produce a clear message rather than a stack trace -- these
models are hundreds of megabytes fetched from the internet on first use, so
"not available" is a normal state, not an exception.
"""

from __future__ import annotations

import io
import threading
import time
import wave
from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly


# Whisper works at 16 kHz internally whatever it is given. The board records
# at 44100 because the anomaly model demands exactly that and a USB
# microphone offers one stream at one rate, so the resampling happens here.
STT_RATE = 16000


@dataclass
class Transcript:
    text: str
    language: str
    confidence: float
    ms: float


def to_mono_float(pcm: np.ndarray, rate: int) -> np.ndarray:
    """int16 at any rate to float32 at 16 kHz, which is what Whisper wants."""
    audio = pcm.astype(np.float32) / 32768.0
    if rate == STT_RATE:
        return audio

    # resample_poly rather than a naive decimation: 44100 to 16000 is 160/441,
    # not an integer ratio, and dropping samples would alias speech into
    # something the model has never heard.
    from math import gcd

    divisor = gcd(STT_RATE, rate)
    return resample_poly(audio, STT_RATE // divisor, rate // divisor).astype(np.float32)


class SpeechToText:
    """faster-whisper, loaded on first use."""

    def __init__(self, model_name: str = "base.en", compute_type: str = "int8",
                 language: str | None = "en") -> None:
        self.model_name = model_name
        self.compute_type = compute_type
        self.language = language
        self._model = None
        self._error: str | None = None
        self._lock = threading.Lock()

    def status(self) -> dict:
        return {
            "engine": "faster-whisper",
            "model": self.model_name,
            "language": self.language or "auto",
            "loaded": self._model is not None,
            "error": self._error,
        }

    def load(self) -> bool:
        """Returns False and records why, rather than raising."""
        with self._lock:
            if self._model is not None:
                return True
            try:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self.model_name, device="cpu", compute_type=self.compute_type
                )
                self._error = None
                return True
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                self._error = f"{type(exc).__name__}: {exc}"
                return False

    def transcribe(self, pcm: np.ndarray, rate: int) -> Transcript:
        if not self.load():
            raise RuntimeError(self._error or "speech model unavailable")

        audio = to_mono_float(pcm, rate)
        started = time.perf_counter()

        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            # The board has already decided where the utterance begins and
            # ends; a second VAD here would trim the edges twice.
            vad_filter=False,
            beam_size=1,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        elapsed = (time.perf_counter() - started) * 1000.0

        # Whisper reports no per-word confidence in this mode; the language
        # probability is the honest thing to hand back and is named as such.
        return Transcript(
            text=text,
            language=getattr(info, "language", self.language or ""),
            confidence=float(getattr(info, "language_probability", 0.0)),
            ms=elapsed,
        )


class TextToSpeech:
    """Piper, loaded on first use. Produces WAV, which is all the board plays."""

    def __init__(self, voice: str = "en_US-lessac-medium") -> None:
        self.voice_name = voice
        self._voice = None
        self._error: str | None = None
        self._lock = threading.Lock()

    def status(self) -> dict:
        return {
            "engine": "piper",
            "voice": self.voice_name,
            "loaded": self._voice is not None,
            "error": self._error,
        }

    def load(self) -> bool:
        with self._lock:
            if self._voice is not None:
                return True
            try:
                from piper import PiperVoice
                from piper.download_voices import download_voice
                from pathlib import Path

                cache = Path.home() / ".cache" / "piper"
                cache.mkdir(parents=True, exist_ok=True)
                model = cache / f"{self.voice_name}.onnx"
                if not model.exists():
                    download_voice(self.voice_name, cache)

                self._voice = PiperVoice.load(str(model))
                self._error = None
                return True
            except Exception as exc:  # noqa: BLE001
                self._error = f"{type(exc).__name__}: {exc}"
                return False

    def synthesise(self, text: str) -> bytes:
        if not self.load():
            raise RuntimeError(self._error or "voice unavailable")

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            self._voice.synthesize_wav(text, wav)
        return buffer.getvalue()
