"""Evidence retrieval + grounded explanation (used by slice 2 Evidence Inspector).

Retrieval is deterministic SQLite/keyword lookup over real record ids — never
invented. The grounded explanation (Gemma) may cite ONLY the ids passed to it; if
the model cites an id not in the evidence set, we reject the output.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import ThemeConfig, load_theme_config
from .inference import GemmaClient
from .schemas import Opportunity


def representative_posts(conn: sqlite3.Connection, ids: list[str], limit: int = 5) -> list[dict]:
    if not ids:
        return []
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, date, brand, text, url FROM posts WHERE id IN ({marks}) LIMIT ?",
        (*ids, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def contradicting_campaigns(conn: sqlite3.Connection, ids: list[str], limit: int = 5) -> list[dict]:
    if not ids:
        return []
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, date, brand, channel, message, url FROM campaigns WHERE id IN ({marks}) LIMIT ?",
        (*ids, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def explain_opportunity(
    opp: Opportunity,
    conn: sqlite3.Connection,
    *,
    client: GemmaClient | None = None,
    tc: ThemeConfig | None = None,
    live: bool = True,
) -> tuple[str, bool]:
    """Return (explanation, grounded_ok). grounded_ok=False if a cited id is invalid.

    live=False forces the instant deterministic summary (no model call) — used for
    anything rendered automatically so the UI never blocks on a slow/rate-limited call.
    """
    tc = tc or load_theme_config()
    client = client or GemmaClient()

    support = representative_posts(conn, opp.support_ids)
    counter = contradicting_campaigns(conn, opp.counter_ids)
    valid_ids = {r["id"] for r in support} | {r["id"] for r in counter}

    if not live or not client.available:
        # Deterministic, fully-grounded summary when offline.
        lead = ", ".join(f'[{r["id"]}]' for r in support[:3]) or "no records"
        return (
            f"{opp.label}: {opp.sample_sizes.get('support', 0)} supporting posts "
            f"(e.g. {lead}); competitor coverage {opp.competitor_coverage:.0%}.",
            True,
        )

    template = (Path(__file__).parent / "prompts" / "explain.txt").read_text()
    prompt = template.format(
        theme=opp.theme,
        label=opp.label,
        consumer_share=f"{opp.consumer_share:.0%}",
        competitor_coverage=f"{opp.competitor_coverage:.0%}",
        score=opp.score,
        components=opp.components.model_dump(),
        sample_sizes=opp.sample_sizes,
        support_block="\n".join(f'{r["id"]}: {r["text"]}' for r in support) or "(none)",
        counter_block="\n".join(f'{r["id"]}: {r["message"]}' for r in counter) or "(none)",
    )
    result = client.chat(prompt, max_tokens=220)

    # Grounding check: every [id] cited must be in the evidence set.
    import re

    cited = set(re.findall(r"\[([a-zA-Z0-9_-]+)\]", result.text))
    grounded_ok = cited.issubset(valid_ids) if cited else True
    return result.text, grounded_ok
