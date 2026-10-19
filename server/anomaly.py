"""The VQ-VAE, and the reconstruction error that decides what is anomalous.

The model is an autoencoder: it was trained only on normal sound, so it
reconstructs normal sound well and unfamiliar sound badly. The score is the
mean squared error between what went in and what came back, and anything
above the calibrated threshold is called an anomaly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .features import FeatureSpec, extract


@dataclass
class Verdict:
    score: float
    threshold: float
    anomaly: bool
    inference_ms: float


class AnomalyDetector:
    def __init__(self, model_path: Path, threshold: float, spec: FeatureSpec,
                 threads: int = 0) -> None:
        self.spec = spec
        self.threshold = threshold
        self.model_path = model_path

        options = ort.SessionOptions()
        if threads:
            # Pinning thread affinity fails in some containers; saying how
            # many threads to use avoids onnxruntime trying to set it.
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def describe(self) -> dict:
        return {
            "model": self.model_path.name,
            "input": self.input_name,
            "output": self.output_name,
            "threshold": self.threshold,
            "sample_rate": self.spec.sample_rate,
            "channels": self.spec.channels,
            "seconds": self.spec.seconds,
            "samples": self.spec.samples,
            "format": "s16le",
            "feature": {
                "type": "gammatone",
                "bands": self.spec.bands,
                "f_min": self.spec.f_min,
                "window_time": self.spec.window_time,
                "hop_time": self.spec.hop_time,
                "db_min": self.spec.db_min,
                "db_max": self.spec.db_max,
            },
        }

    def score(self, pcm: np.ndarray) -> Verdict:
        started = time.perf_counter()

        feature = extract(pcm, self.spec)
        if feature.shape != self.spec.shape:
            raise ValueError(
                f"feature came out {feature.shape}, expected {self.spec.shape}; "
                f"the clip is probably not {self.spec.seconds}s of "
                f"{self.spec.sample_rate} Hz audio"
            )

        # The exported graph is asymmetric: it takes NCHW and returns NHWC.
        # Comparing them without transposing one compares a transpose of the
        # image against itself and gives a plausible, meaningless number.
        x = feature.reshape(1, 1, *self.spec.shape)
        y = self.session.run([self.output_name], {self.input_name: x})[0]

        mse = float(np.mean((np.transpose(x, (0, 2, 3, 1)) - y) ** 2))
        elapsed = (time.perf_counter() - started) * 1000.0

        return Verdict(
            score=mse,
            threshold=self.threshold,
            anomaly=mse > self.threshold,
            inference_ms=elapsed,
        )
