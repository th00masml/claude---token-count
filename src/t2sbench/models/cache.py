"""On-disk cache for model calls, keyed by a hash of (model, adapter, prompt, params).

Every call is stored as one JSON file, so an interrupted run resumes by replaying
cached answers. Latency stored is the original latency; ``cached=True`` marks replays.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _canon(obj: Any) -> Any:
    if is_dataclass(obj):
        return _canon(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    return obj


def cache_key(model: str, adapter: str | None, prompt: Any, params: Any) -> str:
    payload = json.dumps(
        {"model": model, "adapter": adapter, "prompt": _canon(prompt), "params": _canon(params)},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, root: str | Path = "cache/calls"):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def put(self, key: str, value: dict) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
        os.replace(tmp, p)  # atomic: a crash never leaves a half-written entry
