"""Getting int16 mono samples out of whatever the board sent.

Two shapes arrive: a WAV file, which carries its own rate and is what the
recorder uploads, and raw s16le, which is what the live detector streams
because a header per two seconds is pointless once both ends agree.
"""

from __future__ import annotations

import io
import struct

import numpy as np


class AudioError(ValueError):
    """The payload was not audio this server can use."""


def from_raw_s16le(payload: bytes, channels: int = 1) -> np.ndarray:
    if len(payload) % 2:
        raise AudioError("raw s16le payload has an odd byte count")

    pcm = np.frombuffer(payload, dtype="<i2")
    if channels > 1:
        if len(pcm) % channels:
            raise AudioError("sample count is not a multiple of the channel count")
        # Average rather than take channel 0: the model was trained on mono
        # and a quiet channel should not halve the level.
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return pcm


def from_wav(payload: bytes) -> tuple[np.ndarray, int]:
    """Returns (mono int16 samples, sample rate).

    A deliberately small parser. The alternative is a dependency that knows
    every audio format, to read the one the board writes.
    """
    if len(payload) < 12 or payload[0:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise AudioError("not a RIFF/WAVE payload")

    stream = io.BytesIO(payload[12:])
    rate = 0
    channels = 1
    bits = 0
    data = b""

    while True:
        header = stream.read(8)
        if len(header) < 8:
            break
        chunk_id, size = struct.unpack("<4sI", header)

        if chunk_id == b"fmt ":
            fmt = stream.read(size)
            if len(fmt) < 16:
                raise AudioError("truncated fmt chunk")
            tag, channels, rate, _byte_rate, _align, bits = struct.unpack("<HHIIHH", fmt[:16])
            if tag != 1:
                raise AudioError("only uncompressed PCM is supported")
            if bits != 16:
                raise AudioError(f"only 16-bit samples are supported, got {bits}")
        elif chunk_id == b"data":
            # A declared size of zero, or one longer than the file, means the
            # writer never went back to patch the header -- which is what the
            # Qt recorder in ESCA_Qt does, leaving whole sessions unreadable.
            # The samples are there; take what is left rather than lose them.
            rest = stream.read()
            data = rest if (size == 0 or size > len(rest)) else rest[:size]
            break
        else:
            stream.seek(size + (size & 1), io.SEEK_CUR)

    if not rate or not data:
        raise AudioError("no fmt or data chunk")

    return from_raw_s16le(data, channels), rate


def decode(payload: bytes, content_type: str, declared_rate: int | None) -> tuple[np.ndarray, int]:
    """Returns (mono int16 samples, sample rate), whichever form arrived."""
    looks_like_wav = payload[:4] == b"RIFF"

    if looks_like_wav or "wav" in content_type:
        return from_wav(payload)

    if declared_rate is None:
        raise AudioError(
            "raw PCM needs its sample rate; send X-Sample-Rate or upload a WAV"
        )
    return from_raw_s16le(payload), declared_rate


def fit_to_length(pcm: np.ndarray, samples: int) -> np.ndarray:
    """Trim or zero-pad to exactly the length the model wants.

    A clip a few samples short of two seconds -- which is what rounding a
    chunk boundary produces -- should be padded rather than rejected. A clip
    that is wildly the wrong length is a caller bug, and the endpoint checks
    for that before calling this.
    """
    if len(pcm) == samples:
        return pcm
    if len(pcm) > samples:
        return pcm[:samples]

    out = np.zeros(samples, dtype=np.int16)
    out[: len(pcm)] = pcm
    return out
