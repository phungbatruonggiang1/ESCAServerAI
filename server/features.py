"""Turning two seconds of PCM into what the VQ-VAE was trained on.

Every constant here is dictated by the training pipeline, not chosen. They
come from ``python_ai/inference/inference.py`` in ESCA_Qt, which is the code
that produced the scores the threshold was calibrated against. Changing one
without retraining silently invalidates the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gammatone import gtgram


@dataclass(frozen=True)
class FeatureSpec:
    """The audio the model expects, and how it is turned into a feature.

    ``sample_rate`` is 44100 and not 16000. The corpus is 44.1 kHz -- every
    WAV in ``ESCA_Anomaly_Detection/inference/data`` is 88200 frames of
    44.1 kHz mono -- and ``inference.py`` computes the gammatone at 44100.
    ``params.yaml`` says 16000 under ``RECORD``, which belongs to a different
    collection run; feeding 16 kHz audio through these filterbank settings
    shifts every band and produces confident nonsense.
    """

    sample_rate: int = 44100
    channels: int = 1
    seconds: float = 2.0
    window_time: float = 0.12
    hop_time: float = 0.06
    bands: int = 32
    f_min: int = 100

    # Normalisation bounds of the log-gammatone, in dB. Properties of the
    # training set, not of any one clip.
    db_min: float = -23.14520263671875
    db_max: float = 51.67738342285156

    @property
    def samples(self) -> int:
        return int(self.sample_rate * self.seconds)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.bands, self.bands)


def pcm_to_float(pcm: np.ndarray) -> np.ndarray:
    """int16 samples as the filterbank wants them.

    inference.py hands gtgram the raw int16 values rather than scaling to
    +/-1. That choice is baked into db_min/db_max, so it is kept.
    """
    return pcm.astype(np.float64)


def extract(pcm: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    """One (bands, frames) feature, normalised to 0..1.

    Returns float32 so it can go straight into the session.
    """
    audio = pcm_to_float(pcm)

    gtg = gtgram.gtgram(
        audio,
        spec.sample_rate,
        spec.window_time,
        spec.hop_time,
        spec.bands,
        spec.f_min,
    )

    # flipud so the lowest band ends up at the bottom of the image, which is
    # the orientation the model was trained in.
    a = np.flipud(20.0 * np.log10(gtg + 1e-9))
    a = np.clip((a - spec.db_min) / (spec.db_max - spec.db_min), 0.0, 1.0)
    return a.astype(np.float32)
