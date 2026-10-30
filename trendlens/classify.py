"""Contract A: classify_records(records) -> list[ClassifiedRecord].

Gemma tags each record with themes + sentiment. Every result is cached by id so the
dataset is classified once. When the model is unreachable or returns junk, a
deterministic keyword classifier (driven by themes.yaml synonyms) fills in — the
pipeline never stalls and never silently drops a record.
"""
from __future__ import annotations

import re

from .cache import load_cache, save_cache
from .config import ThemeConfig, load_theme_config
from .inference import GemmaClient
from .schemas import ClassifiedRecord

_PROMPT = None


def _prompt_template() -> str:
    global _PROMPT
    if _PROMPT is None:
        from pathlib import Path

        _PROMPT = (Path(__file__).parent / "prompts" / "classify.txt").read_text()
    return _PROMPT


def _theme_descriptions(tc: ThemeConfig) -> str:
    return "\n".join(f"- {k}: {tc.label(k)}" for k in tc.vocab)


def keyword_classify(record_id: str, text: str, tc: ThemeConfig) -> ClassifiedRecord:
    """Deterministic fallback — whole-word matches theme synonyms against the text."""
    low = text.lower()

    def hit(syn: str) -> bool:
        # Word-boundary match so "can" doesn't fire inside "can't".
        return re.search(rf"\b{re.escape(syn)}\b", low) is not None

    themes = [k for k in tc.vocab if any(hit(s) for s in tc.synonyms(k))]
    neg = any(w in low for w in ("not", "no ", "too ", "expensive", "bad", "hate", "bland", "boring"))
    pos = any(w in low for w in ("love", "great", "delicious", "perfect", "refreshing", "worth"))
    sentiment = "neg" if neg and not pos else "pos" if pos else "neu"
    return ClassifiedRecord(
        id=record_id, themes=themes, sentiment=sentiment, reason="keyword fallback"
    )


def _coerce(parsed: dict, record_id: str, tc: ThemeConfig) -> ClassifiedRecord | None:
    """Validate a model result against the frozen vocab; drop out-of-vocab themes."""
    try:
        rec = ClassifiedRecord(
            id=record_id,
            themes=[t for t in parsed.get("themes", []) if t in tc.vocab],
            sentiment=parsed.get("sentiment", "neu"),
            reason=str(parsed.get("reason", ""))[:200],
        )
        return rec
    except Exception:  # noqa: BLE001
        return None


def classify_records(
    records: list[tuple[str, str]],
    *,
    client: GemmaClient | None = None,
    tc: ThemeConfig | None = None,
    use_cache: bool = True,
    live: bool = True,
) -> list[ClassifiedRecord]:
    """records: list of (id, text). Returns one ClassifiedRecord per input, in order.

    live=False forces cache-or-keyword only (no API calls) — used by the UI so the
    board never blocks on ~10s/record reasoning calls. Precompute uses live=True.
    """
    tc = tc or load_theme_config()
    client = client or GemmaClient()
    model = client.cfg.model_id
    cache = load_cache(model) if use_cache else {}

    template = _prompt_template()
    theme_desc = _theme_descriptions(tc)
    out: list[ClassifiedRecord] = []
    dirty = False

    for rid, text in records:
        if use_cache and rid in cache:
            out.append(ClassifiedRecord(**cache[rid]))
            continue

        rec: ClassifiedRecord | None = None
        from_model = False
        if live and client.available:
            prompt = template.format(
                vocab=", ".join(tc.vocab),
                theme_descriptions=theme_desc,
                id=rid,
                text=text,
            )
            # Generous budget: reasoning Gemma 4 models spend tokens thinking before JSON.
            parsed, _log = client.chat_json(prompt, max_tokens=600)
            if parsed:
                rec = _coerce(parsed, rid, tc)
                from_model = rec is not None

        if rec is None:
            rec = keyword_classify(rid, text, tc)  # offline / malformed fallback

        out.append(rec)
        # Cache ONLY genuine Gemma results — never persist the keyword fallback as if
        # it were the model (keeps the cache honest; fallback is recomputed each run).
        if use_cache and from_model:
            cache[rid] = rec.model_dump()
            dirty = True

    if use_cache and dirty:
        save_cache(model, cache)
    return out
