"""Frozen data contracts for the whole pipeline (Pydantic v2).

These are the integration boundaries between the three contributors' modules:

    Contract A  classify_records(records)                 -> list[ClassifiedRecord]
    Contract B  rank_opportunities(posts, campaigns)      -> list[Opportunity]
    Contract C  generate_experiment(opportunity, evidence)-> Experiment

Changing a field name here is a breaking change — discuss before editing.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Sentiment = Literal["pos", "neu", "neg"]


# --------------------------------------------------------------------------- #
# Raw dataset rows (mirror the CSV columns exactly).
# --------------------------------------------------------------------------- #
class ConsumerPost(BaseModel):
    id: str
    date: str                      # ISO date YYYY-MM-DD
    source: str                    # e.g. "reddit", "review", "synthetic"
    brand: str                     # brand referenced, or "none"
    text: str
    url: str = ""
    is_synthetic: bool = True


class CompetitorCampaign(BaseModel):
    id: str
    date: str
    brand: str
    channel: str                   # e.g. "instagram", "tv", "email"
    message: str
    url: str = ""
    is_synthetic: bool = True


class GoldLabel(BaseModel):
    """Hand-labeled expectations for evaluation (labels_gold.csv)."""
    id: str
    expected_theme: str
    expected_sentiment: Sentiment


# --------------------------------------------------------------------------- #
# Contract A — classification output.
# --------------------------------------------------------------------------- #
class ClassifiedRecord(BaseModel):
    id: str
    themes: list[str] = Field(default_factory=list)   # constrained to the config vocab
    sentiment: Sentiment = "neu"
    reason: str = ""                                   # short, grounded justification

    @field_validator("themes")
    @classmethod
    def _dedupe(cls, v: list[str]) -> list[str]:
        # Preserve order, drop duplicates — keeps aggregation denominators honest.
        seen: set[str] = set()
        out: list[str] = []
        for t in v:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out


# --------------------------------------------------------------------------- #
# Contract B — one ranked opportunity.
# --------------------------------------------------------------------------- #
class ScoreComponents(BaseModel):
    consumer_freq: float = 0.0
    momentum: float = 0.0
    underrepresentation: float = 0.0
    evidence_quality: float = 0.0


class Opportunity(BaseModel):
    theme: str
    label: str = ""
    status: Literal["ok", "insufficient_evidence"] = "ok"
    score: float = 0.0                                 # 0..1, deterministic
    components: ScoreComponents = Field(default_factory=ScoreComponents)
    consumer_share: float = 0.0                        # posts mentioning / total posts
    competitor_coverage: float = 0.0                   # campaigns emphasizing / total campaigns
    support_ids: list[str] = Field(default_factory=list)      # consumer post IDs
    counter_ids: list[str] = Field(default_factory=list)      # contradicting campaign IDs
    sample_sizes: dict[str, int] = Field(default_factory=dict)  # {posts, campaigns, support}
    limitations: list[str] = Field(default_factory=list)

    @field_validator("score")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


# --------------------------------------------------------------------------- #
# Contract C — generated experiment.
# --------------------------------------------------------------------------- #
class Experiment(BaseModel):
    hypothesis: str
    a_copy: str
    b_copy: str
    audience: str
    metric: str
    limitations: list[str] = Field(default_factory=list)
    # Deliberately NO "predicted_winner" field — TrendLens never predicts the winner.


# --------------------------------------------------------------------------- #
# Inference telemetry (logged, never shown as "AI confidence").
# --------------------------------------------------------------------------- #
class InferenceLog(BaseModel):
    model: str
    latency_ms: float
    status: Literal["ok", "retried", "fallback", "error"]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    note: str = ""
