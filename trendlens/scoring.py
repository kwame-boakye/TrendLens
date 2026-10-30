"""Contract B: rank_opportunities(...) -> list[Opportunity].

100% deterministic. No model calls here — Gemma's themes are *inputs*; every number
a judge sees is computed in this file and is reproducible. Scoring is transparent:
each component and sample size is returned for display.

    score = 0.30*consumer_freq_percentile
          + 0.25*momentum
          + 0.25*underrepresentation
          + 0.20*evidence_quality
          - penalties (tiny sample, contradictions)

Signature carries the raw posts/campaigns alongside their classifications so the
momentum component can read dates and support/counter ids trace to real records.
"""
from __future__ import annotations

from .config import ThemeConfig, load_theme_config
from .schemas import (
    ClassifiedRecord,
    CompetitorCampaign,
    ConsumerPost,
    Opportunity,
    ScoreComponents,
)


def _in_period(date: str, period: dict) -> bool:
    return bool(period) and period.get("start", "") <= date <= period.get("end", "9999")


def _percentile(value: int, population: list[int]) -> float:
    """Fraction of the population at or below `value` (0..1). Stable for ties."""
    if not population:
        return 0.0
    at_or_below = sum(1 for x in population if x <= value)
    return at_or_below / len(population)


def rank_opportunities(
    classified_posts: list[ClassifiedRecord],
    classified_campaigns: list[ClassifiedRecord],
    posts: list[ConsumerPost],
    campaigns: list[CompetitorCampaign],
    *,
    tc: ThemeConfig | None = None,
) -> list[Opportunity]:
    tc = tc or load_theme_config()
    weights = tc.scoring.get("weights", {})
    min_support = tc.scoring.get("min_support_records", 3)
    tiny_penalty = tc.scoring.get("tiny_sample_penalty", 0.10)
    contra_penalty = tc.scoring.get("contradiction_penalty", 0.15)

    post_by_id = {p.id: p for p in posts}
    cls_post_by_id = {c.id: c for c in classified_posts}
    cls_camp_by_id = {c.id: c for c in classified_campaigns}

    total_posts = max(1, len(posts))
    total_campaigns = max(1, len(campaigns))
    prev_period = tc.periods.get("previous", {})
    recent_period = tc.periods.get("recent", {})

    # Support counts per theme first (needed for the frequency percentile).
    support_ids: dict[str, list[str]] = {t: [] for t in tc.vocab}
    for p in posts:
        cls = cls_post_by_id.get(p.id)
        if not cls:
            continue
        for theme in cls.themes:
            if theme in support_ids:
                support_ids[theme].append(p.id)
    freq_population = [len(ids) for ids in support_ids.values()]

    opportunities: list[Opportunity] = []
    for theme in tc.vocab:
        s_ids = support_ids[theme]
        support_n = len(s_ids)
        consumer_share = support_n / total_posts

        # Competitor coverage + contradicting campaign ids.
        c_ids = [c.id for c in campaigns
                 if theme in (cls_camp_by_id.get(c.id).themes if cls_camp_by_id.get(c.id) else [])]
        competitor_coverage = len(c_ids) / total_campaigns

        # --- components (each 0..1) ---
        consumer_freq = _percentile(support_n, freq_population)

        prev_n = sum(1 for pid in s_ids if _in_period(post_by_id[pid].date, prev_period))
        rec_n = sum(1 for pid in s_ids if _in_period(post_by_id[pid].date, recent_period))
        denom = prev_n + rec_n
        # 0.5 = flat; >0.5 = growing recently; mapped into [0,1].
        momentum = 0.5 if denom == 0 else rec_n / denom

        underrepresentation = max(0.0, consumer_share - competitor_coverage)

        provenance = (
            sum(1 for pid in s_ids if post_by_id[pid].url) / support_n if support_n else 0.0
        )
        evidence_quality = min(1.0, support_n / (2 * min_support)) * (0.5 + 0.5 * provenance)

        components = ScoreComponents(
            consumer_freq=round(consumer_freq, 3),
            momentum=round(momentum, 3),
            underrepresentation=round(underrepresentation, 3),
            evidence_quality=round(evidence_quality, 3),
        )

        raw = (
            weights.get("consumer_freq", 0) * consumer_freq
            + weights.get("momentum", 0) * momentum
            + weights.get("underrepresentation", 0) * underrepresentation
            + weights.get("evidence_quality", 0) * evidence_quality
        )

        limitations: list[str] = []
        penalty = 0.0
        if 0 < support_n <= min_support:
            penalty += tiny_penalty
            limitations.append(f"Tiny sample ({support_n} supporting posts).")
        # Each competitor campaign already covering the theme shrinks the gap.
        penalty += contra_penalty * len(c_ids)
        if c_ids:
            limitations.append(f"{len(c_ids)} competitor message(s) already emphasize this theme.")
        if provenance < 1.0:
            limitations.append("Some records lack source URLs.")

        score = max(0.0, min(1.0, raw - penalty))
        status = "insufficient_evidence" if support_n < min_support else "ok"
        if status == "insufficient_evidence":
            limitations.insert(0, f"Insufficient evidence: need >= {min_support} supporting posts.")

        opportunities.append(
            Opportunity(
                theme=theme,
                label=tc.label(theme),
                status=status,
                score=round(score, 3),
                components=components,
                consumer_share=round(consumer_share, 3),
                competitor_coverage=round(competitor_coverage, 3),
                support_ids=s_ids,
                counter_ids=c_ids,
                sample_sizes={
                    "posts": total_posts,
                    "campaigns": total_campaigns,
                    "support": support_n,
                },
                limitations=limitations,
            )
        )

    # Rank: valid opportunities by score desc; insufficient-evidence ones sink.
    opportunities.sort(key=lambda o: (o.status == "ok", o.score), reverse=True)
    return opportunities
