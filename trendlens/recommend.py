"""Contract D: executive_brief(...) -> ExecutiveBrief.

Timi's feedback: TrendLens's value isn't scraping — it's turning validated analysis
into something a strategist can tell the CEO. This module does that WITHOUT handing
the judgment to a black box:

  * which opportunity leads, and how confident we are, are computed DETERMINISTICALLY
    (confidence_label, from score + sample size + contradictions),
  * data-reliability watch-outs (manipulation / concentration risk) are computed here,
  * Gemma only writes the grounded narrative around those facts.
"""
from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

from .config import ThemeConfig, load_theme_config
from .inference import GemmaClient
from .schemas import ConsumerPost, ExecutiveBrief, Opportunity


def confidence_label(opp: Opportunity) -> str:
    """Transparent, deterministic confidence — never a model-claimed percentage."""
    support = opp.sample_sizes.get("support", 0)
    contradictions = len(opp.counter_ids)
    if opp.status != "ok":
        return "low"
    if opp.score >= 0.40 and support >= 10 and contradictions <= 1:
        return "high"
    if opp.score >= 0.20 and support >= 3:
        return "medium"
    return "low"


def reliability_flags(opp: Opportunity, posts: list[ConsumerPost]) -> list[str]:
    """Data-reliability / manipulation-risk signals (Timi's point #2)."""
    flags: list[str] = []
    by_id = {p.id: p for p in posts}
    support_posts = [by_id[i] for i in opp.support_ids if i in by_id]
    n = len(support_posts)
    if n:
        # Evidence clustered in one brand or one source is a bias / astroturf signal.
        brand_top, brand_cnt = Counter(p.brand for p in support_posts).most_common(1)[0]
        if brand_cnt / n > 0.6:
            flags.append(f"Evidence concentrated in one brand ({brand_top}, {brand_cnt}/{n}) — possible bias.")
        src_top, src_cnt = Counter(p.source for p in support_posts).most_common(1)[0]
        if src_cnt / n > 0.8:
            flags.append(f"Evidence from a single source type ({src_top}) — verify across channels.")
        if any(p.is_synthetic for p in support_posts):
            flags.append("Based on labeled synthetic/demo data — validate on real sourced records.")
    return flags


def executive_brief(
    opportunities: list[Opportunity],
    posts: list[ConsumerPost],
    conn: sqlite3.Connection,
    *,
    client: GemmaClient | None = None,
    tc: ThemeConfig | None = None,
) -> ExecutiveBrief | None:
    tc = tc or load_theme_config()
    client = client or GemmaClient()

    supported = [o for o in opportunities if o.status == "ok"]
    if not supported:
        return None
    top = supported[0]
    runner = supported[1].label if len(supported) > 1 else "—"
    confidence = confidence_label(top)
    watchouts = list(top.limitations) + reliability_flags(top, posts)

    by_id = {p.id: p for p in posts}
    voices = "\n".join(
        f"  [{pid}] {by_id[pid].text}" for pid in top.support_ids[:4] if pid in by_id
    ) or "  (none)"

    # Deterministic fallback brief (also the offline path).
    fallback = ExecutiveBrief(
        headline=f"Lead with {top.label.lower()}: consumers raise it ({top.consumer_share:.0%} of posts) "
        f"but competitors barely message it ({top.competitor_coverage:.0%}).",
        top_theme=top.theme,
        top_label=top.label,
        confidence=confidence,
        rationale=f"{top.sample_sizes.get('support', 0)} supporting posts "
        f"(e.g. {', '.join('[' + i + ']' for i in top.support_ids[:3])}); "
        f"{len(top.counter_ids)} competitor message(s) contradict it. Small sample — treat as a hypothesis.",
        recommended_action=f"Run an A/B experiment emphasizing {top.label.lower()} before committing budget.",
        watchouts=watchouts,
        runner_up=runner,
    )

    if not client.available:
        return fallback

    template = (Path(__file__).parent / "prompts" / "brief.txt").read_text()
    prompt = template.format(
        top_label=top.label, top_theme=top.theme, confidence=confidence,
        consumer_share=f"{top.consumer_share:.0%}", competitor_coverage=f"{top.competitor_coverage:.0%}",
        support_n=top.sample_sizes.get("support", 0), counter_n=len(top.counter_ids),
        support_ids=", ".join(top.support_ids[:8]), voices=voices,
        runner_up=runner, watchouts="; ".join(watchouts) or "none",
    )
    parsed, _log = client.chat_json(prompt, max_tokens=600)
    if not parsed:
        return fallback
    try:
        return ExecutiveBrief(
            headline=str(parsed.get("headline", fallback.headline))[:300],
            top_theme=top.theme, top_label=top.label,
            confidence=confidence,  # deterministic — never taken from the model
            rationale=str(parsed.get("rationale", fallback.rationale))[:600],
            recommended_action=str(parsed.get("recommended_action", fallback.recommended_action))[:300],
            watchouts=watchouts, runner_up=runner,
        )
    except Exception:  # noqa: BLE001
        return fallback
