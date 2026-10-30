"""Deterministic scoring fixtures — must be 100% reproducible.

Includes the signature negative test: adding a competitor campaign that emphasizes
a theme raises competitor coverage and LOWERS that theme's gap score.
"""
from __future__ import annotations

from trendlens.config import ThemeConfig
from trendlens.schemas import ClassifiedRecord, CompetitorCampaign, ConsumerPost
from trendlens.scoring import rank_opportunities


def _tc() -> ThemeConfig:
    return ThemeConfig(
        vertical="test",
        brands=["X"],
        periods={
            "previous": {"start": "2026-07-01", "end": "2026-08-31"},
            "recent": {"start": "2026-09-01", "end": "2026-10-31"},
        },
        themes={
            "taste": {"label": "Taste", "synonyms": ["taste"]},
            "affordability": {"label": "Affordability", "synonyms": ["price"]},
        },
        scoring={
            "weights": {
                "consumer_freq": 0.30,
                "momentum": 0.25,
                "underrepresentation": 0.25,
                "evidence_quality": 0.20,
            },
            "min_support_records": 3,
            "tiny_sample_penalty": 0.10,
            "contradiction_penalty": 0.15,
        },
    )


def _posts(n_taste: int) -> tuple[list, list]:
    posts, cls = [], []
    for i in range(n_taste):
        pid = f"p{i:03d}"
        posts.append(ConsumerPost(id=pid, date="2026-10-01", source="s", brand="X",
                                   text="taste", url=f"synthetic://{pid}"))
        cls.append(ClassifiedRecord(id=pid, themes=["taste"], sentiment="neg"))
    return posts, cls


def test_insufficient_evidence_below_threshold():
    posts, cls_posts = _posts(2)  # < min_support (3)
    opps = rank_opportunities(cls_posts, [], posts, [], tc=_tc())
    taste = next(o for o in opps if o.theme == "taste")
    assert taste.status == "insufficient_evidence"


def test_clear_gap_scores_high_with_no_competitor_coverage():
    posts, cls_posts = _posts(10)
    opps = rank_opportunities(cls_posts, [], posts, [], tc=_tc())
    taste = next(o for o in opps if o.theme == "taste")
    assert taste.status == "ok"
    assert taste.competitor_coverage == 0.0
    assert taste.score > 0.4


def test_counterevidence_lowers_score():
    """Add a taste campaign -> coverage up, underrepresentation down, score falls."""
    posts, cls_posts = _posts(10)
    before = next(o for o in rank_opportunities(cls_posts, [], posts, [], tc=_tc())
                  if o.theme == "taste").score

    camp = CompetitorCampaign(id="c001", date="2026-10-02", brand="X",
                              channel="tv", message="great taste", url="synthetic://c001")
    cls_camp = [ClassifiedRecord(id="c001", themes=["taste"], sentiment="pos")]
    after_opp = next(o for o in rank_opportunities(cls_posts, cls_camp, posts, [camp], tc=_tc())
                     if o.theme == "taste")

    assert after_opp.competitor_coverage > 0.0
    assert "c001" in after_opp.counter_ids
    assert after_opp.score < before  # the falsification demo, as a unit test
