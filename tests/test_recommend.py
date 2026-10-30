"""Deterministic parts of the executive brief — confidence + reliability flags."""
from __future__ import annotations

from trendlens.recommend import confidence_label, reliability_flags
from trendlens.schemas import ConsumerPost, Opportunity


def _opp(score, support, counters=0, status="ok"):
    return Opportunity(
        theme="taste", label="Taste", status=status, score=score,
        counter_ids=[f"c{i}" for i in range(counters)],
        support_ids=[f"p{i}" for i in range(support)],
        sample_sizes={"support": support, "posts": 100, "campaigns": 20},
    )


def test_confidence_high_medium_low():
    assert confidence_label(_opp(0.5, 15, 0)) == "high"
    assert confidence_label(_opp(0.25, 5, 2)) == "medium"
    assert confidence_label(_opp(0.1, 2, 0)) == "low"
    assert confidence_label(_opp(0.9, 50, status="insufficient_evidence")) == "low"


def test_reliability_flags_detect_brand_concentration():
    opp = _opp(0.5, 5)
    # all five supporting posts are the same brand -> concentration flag
    posts = [ConsumerPost(id=f"p{i}", date="2026-10-01", source="synthetic",
                          brand="FizzWell", text="taste", url="synthetic://x") for i in range(5)]
    flags = reliability_flags(opp, posts)
    assert any("concentrated in one brand" in f for f in flags)
    assert any("synthetic" in f for f in flags)
