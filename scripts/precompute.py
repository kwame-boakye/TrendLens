"""Batch-classify the whole dataset into the cache so the live demo never stalls.

Run from repo root:  python scripts/precompute.py
Re-running is a cache hit (no model calls) unless the prompt version changes.
Works offline too — it will use the deterministic keyword fallback and still cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trendlens.classify import classify_records  # noqa: E402
from trendlens.data_io import load_campaigns, load_posts  # noqa: E402
from trendlens.inference import GemmaClient  # noqa: E402


def _chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main() -> int:
    posts = load_posts()
    campaigns = load_campaigns()
    client = GemmaClient()
    mode = "live Gemma" if client.available else "offline keyword fallback"
    items = [(p.id, p.text) for p in posts] + [(c.id, c.message) for c in campaigns]
    print(f"Classifying {len(items)} records via {mode} (chunked, cache saved per chunk) ...")

    done = 0
    for chunk in _chunked(items, 8):
        # Each call persists genuine Gemma results to the cache, so progress survives
        # interruption / rate limits — rerun to resume where it left off.
        classify_records(chunk, client=client, live=True)
        done += len(chunk)
        print(f"  {done}/{len(items)} done")

    print(f"Cache written under data/cache/ (model: {client.cfg.model_id}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
