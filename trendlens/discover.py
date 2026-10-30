"""Open theme discovery for uploaded data (industry-agnostic path).

For the example dataset we use the fixed vocabulary in config/themes.yaml. For an
arbitrary company's upload we can't assume the themes — so Gemma DISCOVERS them from
the data itself:

  1. open-extract 1-3 normalized need-phrases per customer comment (batched),
  2. build a dynamic vocabulary from the most common phrases,
  3. classify competitor messages CONSTRAINED to that same vocabulary (batched),
  4. hand both to the normal deterministic scorer.

Batching (≈10 records/call) keeps a live upload to a handful of Gemma calls instead
of one per row.
"""
from __future__ import annotations

import re
from collections import Counter

from .config import ThemeConfig
from .inference import GemmaClient
from .schemas import ClassifiedRecord

_OPEN_PROMPT = """Extract the customer NEEDS or topics in EACH comment as 1-3 short,
normalized phrases (2-4 words, lowercase, general — e.g. "delivery speed", "pricing",
"customer support", "app reliability", "product quality"). Reuse the SAME phrase across
comments when the need is the same. Also give sentiment.
Do NOT show reasoning or <thought> tags.
Return a STRICT JSON array, one object per comment, matching ids exactly:
[{{"id":"<id>","themes":["<phrase>", ...],"sentiment":"pos|neu|neg"}}, ...]

Comments:
{block}

JSON array:"""

_CONSTRAINED_PROMPT = """For EACH competitor message below, say which of these customer
NEEDS it emphasizes. Choose ONLY from this list (exact strings), or [] if none:
{vocab}
Do NOT show reasoning or <thought> tags.
Return a STRICT JSON array, one object per message, matching ids exactly:
[{{"id":"<id>","themes":["<need>", ...]}}, ...]

Messages:
{block}

JSON array:"""


def _norm(phrase: str) -> str:
    return re.sub(r"\s+", " ", str(phrase).strip().lower())[:40]


def _batches(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def open_extract(
    records: list[tuple[str, str]], client: GemmaClient, *, batch_size: int = 10, progress=None
) -> list[ClassifiedRecord]:
    """Discover free-form needs per record (batched). Needs live Gemma."""
    out: list[ClassifiedRecord] = []
    done = 0
    for batch in _batches(records, batch_size):
        block = "\n".join(f"{rid}: {text}" for rid, text in batch)
        arr, _ = _chat_array(client, _OPEN_PROMPT.format(block=block))
        by_id = {str(o.get("id")): o for o in arr} if arr else {}
        for rid, _text in batch:
            o = by_id.get(rid, {})
            themes = [_norm(t) for t in o.get("themes", []) if str(t).strip()][:3]
            out.append(ClassifiedRecord(id=rid, themes=themes,
                                        sentiment=o.get("sentiment", "neu"), reason="open-extracted"))
        done += len(batch)
        if progress:
            progress(done, len(records))
    return out


def classify_constrained(
    records: list[tuple[str, str]], vocab: list[str], client: GemmaClient,
    *, batch_size: int = 10, progress=None
) -> list[ClassifiedRecord]:
    """Tag each record with themes drawn ONLY from vocab (batched)."""
    out: list[ClassifiedRecord] = []
    done = 0
    for batch in _batches(records, batch_size):
        block = "\n".join(f"{rid}: {text}" for rid, text in batch)
        arr, _ = _chat_array(client, _CONSTRAINED_PROMPT.format(vocab=", ".join(vocab), block=block))
        by_id = {str(o.get("id")): o for o in arr} if arr else {}
        for rid, _text in batch:
            o = by_id.get(rid, {})
            themes = [_norm(t) for t in o.get("themes", []) if _norm(t) in vocab]
            out.append(ClassifiedRecord(id=rid, themes=themes, sentiment="neu", reason="constrained"))
        done += len(batch)
        if progress:
            progress(done, len(records))
    return out


def _chat_array(client: GemmaClient, prompt: str):
    from .inference import extract_json_array
    result = client.chat(prompt, max_tokens=1200)
    arr = extract_json_array(result.text)
    if arr is None:
        retry = client.chat(prompt + "\n\nReturn ONLY the JSON array.", max_tokens=1200)
        arr = extract_json_array(retry.text)
    return arr, result.log


def build_dynamic_config(
    cls_posts: list[ClassifiedRecord], base: ThemeConfig, company: str, *, top_n: int = 8
) -> tuple[ThemeConfig, list[ClassifiedRecord]]:
    """Vocabulary = the most common discovered needs. Returns (config, filtered posts)."""
    counts = Counter(t for c in cls_posts for t in c.themes)
    vocab = [t for t, _ in counts.most_common(top_n)]
    themes = {t: {"label": t.title(), "synonyms": [t] + t.split()} for t in vocab}
    tc = ThemeConfig(vertical=company, brands=[], periods=base.periods,
                     themes=themes, scoring=base.scoring)
    filtered = [ClassifiedRecord(id=c.id, themes=[t for t in c.themes if t in vocab],
                                 sentiment=c.sentiment, reason=c.reason) for c in cls_posts]
    return tc, filtered
