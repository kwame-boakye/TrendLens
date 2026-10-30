"""GO/NO-GO gate: confirm DigitalOcean Gemma access before trusting live classify.

Run from repo root:  python scripts/smoke_inference.py

Exits 0 only if the configured model is listed AND returns one valid completion.
Prints latency. Never prints the API key.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trendlens.inference import GemmaClient  # noqa: E402


def main() -> int:
    client = GemmaClient()
    if not client.available:
        print("NO-GO: no API key set. Copy .env.example -> .env and set INFERENCE_API_KEY")
        print("       (free key: https://aistudio.google.com/apikey).")
        return 1

    print(f"Base URL : {client.cfg.base_url}")
    print(f"Model    : {client.cfg.model_id}")

    try:
        models = client.list_models()
    except Exception as exc:  # noqa: BLE001
        print(f"NO-GO: could not list models: {exc}")
        return 1

    # Show the exact Gemma ids this key can use — copy one into GEMMA_MODEL_ID.
    gemma_ids = sorted(m for m in models if "gemma" in m.lower())
    if gemma_ids:
        print("Gemma ids available to your key:")
        print("  " + ", ".join(gemma_ids))

    if client.cfg.model_id not in models:
        suggestion = gemma_ids[0] if gemma_ids else client.cfg.fallback_model_id
        print(f"WARNING: '{client.cfg.model_id}' not available. Trying '{suggestion}' instead.")
        print(f"         -> set GEMMA_MODEL_ID={suggestion} in .env to make it the default.")

    if client.cfg.model_id in models:
        test_model = client.cfg.model_id
    elif gemma_ids:
        test_model = gemma_ids[0]
    else:
        test_model = client.cfg.fallback_model_id
    result = client.chat("Reply with exactly: TRENDLENS_OK", model=test_model, max_tokens=16)
    print(f"Status   : {result.log.status}")
    print(f"Latency  : {result.log.latency_ms:.0f} ms")
    print(f"Tokens   : prompt={result.log.prompt_tokens} completion={result.log.completion_tokens}")
    print(f"Response : {result.text!r}")

    if result.log.status == "ok" and result.text:
        print("\nGO: live inference works.")
        return 0
    print(f"\nNO-GO: {result.log.note or 'no response'}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
