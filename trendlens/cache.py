"""JSON classification cache.

Keyed by record id + model + prompt version so a prompt change invalidates stale
entries. Precomputing the dataset into this cache is what lets the live demo run
without spending the first 15 FPS of a judge's attention on model latency.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import CACHE_DIR

PROMPT_VERSION = "v1"


def _cache_file(model: str) -> Path:
    safe = model.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"classify_{safe}_{PROMPT_VERSION}.json"


def load_cache(model: str) -> dict[str, dict]:
    path = _cache_file(model)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(model: str, cache: dict[str, dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_file(model).write_text(json.dumps(cache, indent=2, sort_keys=True))
