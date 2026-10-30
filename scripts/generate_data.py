"""Generate the clearly-labeled synthetic demo dataset (brand-neutral streaming).

Deterministic (fixed seed) and reproducible. Every row has is_synthetic=true and a
synthetic:// provenance URL — TrendLens never presents these as real market data.
Brands are fictional (Streamly / Wavecast / Tunefox); no real company is named.

Mirrors the Spotify-style story WITHOUT naming real companies:
  * variety  -> strong opportunity (listeners want less-repetitive, varied playlists;
                competitors barely message it)
  * control  -> strong opportunity (want control/customization; under-messaged)
  * discovery/price -> already covered (competitors over-message these)
  * ads      -> insufficient evidence (<3 consumer posts)

Run from repo root:  python scripts/generate_data.py
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

BRANDS = ["Streamly", "Wavecast", "Tunefox"]
rng = random.Random(42)

# period windows must match config/themes.yaml
PREV = ("2026-07-01", "2026-08-31")
RECENT = ("2026-09-01", "2026-10-31")

# (theme, count, recent_bias) — recent_bias = share of posts placed in the recent window.
POST_PLAN = [
    ("variety", 42, 0.78),        # dominant + growing -> high momentum, the lead opportunity
    ("control", 28, 0.62),
    ("discovery", 24, 0.45),
    ("price", 20, 0.45),
    ("audio_quality", 10, 0.50),
    ("ads", 2, 0.50),             # below min_support -> insufficient evidence
]

# Consumer post templates per theme (synonyms embedded so the offline keyword
# classifier recovers the same theme the gold labels expect).
POST_TEXT = {
    "variety": [
        "My {b} playlists are so repetitive, I keep hearing the same tracks on repeat.",
        "I wish {b} gave me more variety — the rotation feels stale and boring.",
        "Finally {b} feels less repetitive lately, the mixes are way more varied.",
        "{b} keeps repeating the same songs, I need fresh variety not repeats.",
    ],
    "control": [
        "I want more control over {b} recommendations — let me customize what I skip.",
        "{b} should let me manually tune my mix and choose my own settings.",
        "Love that {b} added settings to control my feed, the customization is great.",
    ],
    "discovery": [
        "{b} is great for discovery — the personalized recommendations feel spot on.",
        "I wish {b} helped me discover more, its suggestions feel generic.",
    ],
    "price": [
        "{b} premium is too expensive for what you get, not worth the subscription.",
        "Good value from {b}, the premium price is affordable for me.",
    ],
    "audio_quality": [
        "The audio quality on {b} is crisp, love the lossless sound.",
        "{b} sound quality dropped, the bitrate feels low now.",
    ],
    "ads": [
        "Too many ads interrupting my {b} free tier, so annoying.",
    ],
}

# Competitor campaign coverage (theme, count). variety/control under-messaged vs demand;
# discovery/price over-marketed -> variety & control become the clear gaps.
CAMPAIGN_PLAN = [
    ("variety", 2),
    ("control", 2),
    ("discovery", 7),
    ("price", 6),
    ("audio_quality", 5),
    ("ads", 6),
]
CAMPAIGN_TEXT = {
    "variety": "Never repetitive — {b} keeps your playlists fresh with endless variety.",
    "control": "Take control — customize and fine-tune every recommendation on {b}.",
    "discovery": "Discover your next favorite — {b} personalized recommendations made for you.",
    "price": "Premium sound at an affordable price — try {b} Premium today.",
    "audio_quality": "Lossless hi-fi audio — hear every detail with {b}.",
    "ads": "Ad-free listening, no interruptions — upgrade to {b} Premium.",
}
CHANNELS = ["instagram", "tiktok", "youtube", "email", "billboard"]


def rand_date(bias_recent: float) -> str:
    use = RECENT if rng.random() < bias_recent else PREV
    _, m1, _ = map(int, use[0].split("-"))
    _, m2, _ = map(int, use[1].split("-"))
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
            sentiment = "neg" if any(w in text for w in ("too", "wish", "stale", "boring", "repeating", "dropped", "annoying", "repetitive")) else "pos"
            rid = f"p{pid:03d}"
            posts.append(
                {
                    "id": rid,
                    "date": rand_date(bias),
                    "source": "synthetic",
                    "brand": brand,
                    "text": text,
                    "url": f"synthetic://posts/{rid}",
                    "is_synthetic": "true",
                }
            )
            if i < 2:  # hand-label ~2 per theme as gold (incl. contradictions via neg sentiment)
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
                    "date": rand_date(0.5),
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
