"""Dataset ingestion: CSV -> validated models -> in-memory SQLite index.

SQLite is built fresh from the CSVs at runtime (no DB service, nothing to deploy).
It backs keyword evidence retrieval in evidence.py.
"""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from .config import DATA_DIR
from .schemas import CompetitorCampaign, ConsumerPost, GoldLabel


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def load_posts(path: Path | None = None) -> list[ConsumerPost]:
    rows = _read_csv(path or DATA_DIR / "consumer_posts.csv")
    return [ConsumerPost(**{**r, "is_synthetic": _to_bool(r.get("is_synthetic", "true"))}) for r in rows]


def load_campaigns(path: Path | None = None) -> list[CompetitorCampaign]:
    rows = _read_csv(path or DATA_DIR / "competitor_campaigns.csv")
    return [CompetitorCampaign(**{**r, "is_synthetic": _to_bool(r.get("is_synthetic", "true"))}) for r in rows]


def load_gold(path: Path | None = None) -> list[GoldLabel]:
    p = path or DATA_DIR / "labels_gold.csv"
    if not p.exists():
        return []
    return [GoldLabel(**r) for r in _read_csv(p)]


def build_index(
    posts: list[ConsumerPost], campaigns: list[CompetitorCampaign]
) -> sqlite3.Connection:
    """Fresh in-memory SQLite index over posts + campaigns for retrieval."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE posts(id TEXT PRIMARY KEY, date TEXT, source TEXT, brand TEXT,
                           text TEXT, url TEXT, is_synthetic INT);
        CREATE TABLE campaigns(id TEXT PRIMARY KEY, date TEXT, brand TEXT, channel TEXT,
                               message TEXT, url TEXT, is_synthetic INT);
        """
    )
    conn.executemany(
        "INSERT INTO posts VALUES (?,?,?,?,?,?,?)",
        [(p.id, p.date, p.source, p.brand, p.text, p.url, int(p.is_synthetic)) for p in posts],
    )
    conn.executemany(
        "INSERT INTO campaigns VALUES (?,?,?,?,?,?,?)",
        [(c.id, c.date, c.brand, c.channel, c.message, c.url, int(c.is_synthetic)) for c in campaigns],
    )
    conn.commit()
    return conn
