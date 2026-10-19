"""Where uploaded training clips land.

One directory per machine and label, plus a JSONL manifest that keeps the
metadata the board knew and a filename cannot carry -- which device recorded
it, whether samples were dropped, when it started.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


SAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_name(value: str, fallback: str) -> str:
    """A path component that cannot escape the dataset directory.

    Everything here arrives over the network from a device on the LAN, so a
    label of "../../etc" has to become a filename and not a traversal.
    """
    cleaned = SAFE.sub("_", value.strip())[:64]
    return cleaned or fallback


@dataclass
class ClipMeta:
    machine: str
    label: str
    sample_rate: int
    channels: int
    seconds: float
    started_ns: int
    device: str
    dropped: int
    received_utc: str
    path: str
    bytes: int


class DatasetStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def manifest_path(self) -> Path:
        return self.root / "manifest.jsonl"

    def save(self, payload: bytes, *, machine: str, label: str, sample_rate: int,
             channels: int, seconds: float, started_ns: int, device: str,
             dropped: int, suffix: str = ".wav") -> ClipMeta:
        machine = safe_name(machine, "unknown-machine")
        label = safe_name(label, "unlabelled")

        folder = self.root / machine / label
        folder.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        name = f"{stamp}{suffix}"
        path = folder / name
        path.write_bytes(payload)

        meta = ClipMeta(
            machine=machine,
            label=label,
            sample_rate=sample_rate,
            channels=channels,
            seconds=seconds,
            started_ns=started_ns,
            device=device,
            dropped=dropped,
            received_utc=datetime.now(timezone.utc).isoformat(),
            path=str(path.relative_to(self.root)),
            bytes=len(payload),
        )

        # Appending under a lock, because uvicorn will happily run two
        # uploads at once and a half-written JSON line poisons the manifest
        # for every later reader.
        with self._lock:
            with self.manifest_path().open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")

        return meta

    def stats(self) -> dict:
        counts: dict[str, dict[str, int]] = {}
        total = 0
        seconds = 0.0

        manifest = self.manifest_path()
        if manifest.exists():
            with manifest.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    machine = row.get("machine", "?")
                    label = row.get("label", "?")
                    counts.setdefault(machine, {}).setdefault(label, 0)
                    counts[machine][label] += 1
                    total += 1
                    seconds += float(row.get("seconds", 0.0))

        return {"clips": total, "seconds": round(seconds, 1), "by_machine": counts}
