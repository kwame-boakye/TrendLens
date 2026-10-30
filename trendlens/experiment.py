"""Contract C: generate_experiment(opportunity, evidence) -> Experiment (slice 4).

Gemma drafts a testable A/B experiment. It never predicts the winner and never
promises ROI. Offline, a deterministic template keeps the demo alive.
"""
from __future__ import annotations

from pathlib import Path

from .config import ThemeConfig, load_theme_config
from .inference import GemmaClient
from .schemas import Experiment, Opportunity


def generate_experiment(
    opp: Opportunity,
    evidence_summary: str,
    *,
    client: GemmaClient | None = None,
    tc: ThemeConfig | None = None,
) -> Experiment:
    tc = tc or load_theme_config()
    client = client or GemmaClient()

    if client.available:
        template = (Path(__file__).parent / "prompts" / "experiment.txt").read_text()
        prompt = template.format(
            theme=opp.theme,
            label=opp.label,
            evidence_summary=evidence_summary[:400],
            support_ids=", ".join(opp.support_ids[:8]),
        )
        parsed, _log = client.chat_json(prompt, max_tokens=400)
        if parsed:
            try:
                # Defensively drop any winner-prediction the model may volunteer.
                parsed.pop("predicted_winner", None)
                return Experiment(**parsed)
            except Exception:  # noqa: BLE001 — fall through to template
                pass

    # Deterministic fallback.
    return Experiment(
        hypothesis=f"If we emphasize {opp.label.lower()} in messaging, engagement with the "
        f"target audience will change measurably versus generic brand copy.",
        a_copy="Our drink. Made for you.",
        b_copy=f"Finally — {opp.label.lower()} done right. Taste the difference.",
        audience=f"Consumers who recently voiced {opp.label.lower()} needs.",
        metric="Click-through rate on the campaign landing page.",
        limitations=[
            "Based on a small, partly synthetic sample.",
            "Does not predict which variant wins — must be A/B tested live.",
        ],
    )
