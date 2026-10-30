"""Parse a company's uploaded data into the pipeline's schemas.

Lenient column detection so any company can bring a CSV without matching an exact
template: we look for the obvious text/message columns and fill the rest. This is
what makes TrendLens industry-agnostic — the same pipeline runs on whatever a
company uploads.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .schemas import CompetitorCampaign, ConsumerPost

_TEXT_COLS = ["text", "review", "comment", "feedback", "content", "message", "post"]
_MSG_COLS = ["message", "ad", "campaign", "text", "content", "copy"]
_BRAND_COLS = ["brand", "company", "competitor", "product", "name"]
_DATE_COLS = ["date", "created", "timestamp", "time"]


def _find(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    # fall back to first column containing the word
    for key, orig in lower.items():
        if any(cand in key for cand in candidates):
            return orig
    return None


def parse_feedback(df: pd.DataFrame, company: str = "Company") -> list[ConsumerPost]:
    text_col = _find(df, _TEXT_COLS)
    if text_col is None:
        # no obvious text column — use the first string-ish column
        text_col = df.columns[0]
    brand_col = _find(df, _BRAND_COLS)
    date_col = _find(df, _DATE_COLS)
    today = date.today().isoformat()
    out: list[ConsumerPost] = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        text = str(row[text_col]).strip()
        if not text or text.lower() == "nan":
            continue
        out.append(ConsumerPost(
            id=f"p{i:04d}",
            date=str(row[date_col])[:10] if date_col else today,
            source="uploaded",
            brand=str(row[brand_col]) if brand_col else company,
            text=text,
            url="",
            is_synthetic=False,
        ))
    return out


def parse_competitors(df: pd.DataFrame) -> list[CompetitorCampaign]:
    msg_col = _find(df, _MSG_COLS) or df.columns[0]
    brand_col = _find(df, _BRAND_COLS)
    date_col = _find(df, _DATE_COLS)
    today = date.today().isoformat()
    out: list[CompetitorCampaign] = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        msg = str(row[msg_col]).strip()
        if not msg or msg.lower() == "nan":
            continue
        out.append(CompetitorCampaign(
            id=f"c{i:04d}",
            date=str(row[date_col])[:10] if date_col else today,
            brand=str(row[brand_col]) if brand_col else "Competitor",
            channel="uploaded",
            message=msg,
            url="",
            is_synthetic=False,
        ))
    return out


def parse_campaign_results(df: pd.DataFrame) -> list[dict]:
    """Optional past-performance rows: keep theme + a conversion/effectiveness metric.

    Returns a list of {label, metric, value} for display; empty if nothing usable.
    """
    theme_col = _find(df, ["theme", "topic", "message", "campaign", "focus"])
    conv_col = _find(df, ["conversion", "conversion_rate", "ctr", "cvr", "effectiveness", "rate"])
    if theme_col is None or conv_col is None:
        return []
    rows: list[dict] = []
    for _, row in df.iterrows():
        try:
            val = float(str(row[conv_col]).replace("%", "").strip())
        except ValueError:
            continue
        rows.append({"label": str(row[theme_col]), "value": val})
    return rows
