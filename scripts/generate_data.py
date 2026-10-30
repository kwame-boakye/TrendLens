"""Generate the clearly-labeled synthetic demo dataset.

Deterministic (fixed seed) and reproducible. Every row has is_synthetic=true and a
synthetic:// provenance URL — TrendLens never presents these as real market data.

Dataset is engineered to tell a clear story:
  * taste          -> strong opportunity  (high consumer demand, low competitor coverage)
  * affordability  -> already covered      (competitors message it heavily)
  * sustainability -> insufficient evidence (<3 consumer posts, yet over-marketed)

Run from repo root:  python scripts/generate_data.py
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

BRANDS = ["FizzWell", "PureSip", "NovaCola"]
rng = random.Random(42)

# period windows must match config/themes.yaml
PREV = ("2026-07-01", "2026-08-31")
RECENT = ("2026-09-01", "2026-10-31")

# (theme, count, recent_bias) — recent_bias = share of posts placed in the recent window.
POST_PLAN = [
    ("taste", 45, 0.78),          # dominant + growing -> high momentum
    ("health", 30, 0.60),
    ("affordability", 25, 0.45),
    ("convenience", 15, 0.50),
    ("packaging", 8, 0.50),
    ("sustainability", 2, 0.50),  # below min_support -> insufficient evidence
]

# Consumer post templates per theme (synonyms embedded so the offline keyword
# classifier recovers the same theme the gold labels expect).
POST_TEXT = {
    "taste": [
        "The {b} flavor is so refreshing and delicious, honestly the best taste I've had.",
        "I wish {b} had a better taste, it's too bland and not sweet enough for me.",
        "Love how {b} tastes — that flavor keeps me coming back.",
        "{b} needs to fix the taste, it's bitter and not refreshing at all.",
    ],
    "health": [
        "Finally a low-sugar option from {b}, way fewer calories and feels healthy.",
        "Too much sugar in {b}, I want a natural low-sugar drink without additives.",
        "{b} should cut the sugar, I'm watching my calories and want a diet version.",
    ],
    "affordability": [
        "{b} is way too expensive for what it is, not worth the price.",
        "Great value from {b}, cheap and affordable compared to the rest.",
        "The price of {b} keeps going up, I can't afford it weekly anymore.",
    ],
    "convenience": [
        "I love the on-the-go {b} cans, so convenient and easy to grab.",
        "Hard to find {b} in stock near me, wish it were more available.",
    ],
    "packaging": [
        "The {b} bottle design looks premium but the plastic isn't recyclable.",
        "Nice packaging on {b}, the new can label really stands out.",
    ],
    "sustainability": [
        "I'd pay more if {b} were truly sustainable and eco friendly for the planet.",
    ],
}

# Competitor campaign coverage (theme, count). sustainability over-marketed vs demand.
CAMPAIGN_PLAN = [
    ("taste", 2),
    ("affordability", 6),
    ("health", 4),
    ("convenience", 5),
    ("packaging", 4),
    ("sustainability", 7),
]
CAMPAIGN_TEXT = {
    "taste": "Bold refreshing flavor you'll love — taste the {b} difference.",
    "affordability": "Unbeatable value. {b} gives you more for less — affordable every day.",
    "health": "Low-sugar, fewer calories. {b} is the healthy choice.",
    "convenience": "Grab {b} on the go — convenient cans, always within reach.",
    "packaging": "New {b} bottle design — sleek, premium, recyclable.",
    "sustainability": "{b} for the planet — sustainable, eco-friendly, carbon-neutral.",
}
CHANNELS = ["instagram", "tv", "email", "youtube", "billboard"]


def rand_date(window, bias_recent: float) -> str:
    use = RECENT if rng.random() < bias_recent else PREV
    y, m1, d1 = map(int, use[0].split("-"))
    _, m2, d2 = map(int, use[1].split("-"))
    month = rng.choice([m1, m2])
    day = rng.randint(1, 28)
    return f"2026-{month:02d}-{day:02d}"


def main() -> None:
    posts = []
    pid = 1
    gold = []
    for theme, count, bias in POST_PLAN:
        templates = POST_TEXT[theme]
        for i in range(count):
            brand = rng.choice(BRANDS)
            text = templates[i % len(templates)].format(b=brand)
            sentiment = "neg" if any(w in text for w in ("too", "wish", "can't", "fix", "needs")) else "pos"
            rid = f"p{pid:03d}"
            posts.append(
                {
                    "id": rid,
                    "date": rand_date(theme, bias),
                    "source": "synthetic",
                    "brand": brand,
                    "text": text,
                    "url": f"synthetic://posts/{rid}",
                    "is_synthetic": "true",
                }
            )
            # Hand-label ~2 per theme as gold (incl. contradictions via neg sentiment).
            if i < 2:
                gold.append({"id": rid, "expected_theme": theme, "expected_sentiment": sentiment})
            pid += 1

    campaigns = []
    cid = 1
    for theme, count in CAMPAIGN_PLAN:
        for _ in range(count):
            brand = rng.choice(BRANDS)
            rid = f"c{cid:03d}"
            campaigns.append(
                {
                    "id": rid,
                    "date": rand_date(theme, 0.5),
                    "brand": brand,
                    "channel": rng.choice(CHANNELS),
                    "message": CAMPAIGN_TEXT[theme].format(b=brand),
                    "url": f"synthetic://campaigns/{rid}",
                    "is_synthetic": "true",
                }
            )
            cid += 1

    _write(DATA / "consumer_posts.csv",
           ["id", "date", "source", "brand", "text", "url", "is_synthetic"], posts)
    _write(DATA / "competitor_campaigns.csv",
           ["id", "date", "brand", "channel", "message", "url", "is_synthetic"], campaigns)
    _write(DATA / "labels_gold.csv",
           ["id", "expected_theme", "expected_sentiment"], gold)

    print(f"Wrote {len(posts)} posts, {len(campaigns)} campaigns, {len(gold)} gold labels.")


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
