"""DigitalOcean Gradient AI (Gemma 4) client — OpenAI-compatible.

Thin wrapper that:
  * points the OpenAI SDK at https://inference.do-ai.run/v1,
  * logs model / latency / token-usage / status for every call,
  * retries ONCE on malformed JSON, then signals the caller to fall back.

It never raises on a bad key at import time; `GemmaClient.available` lets callers
degrade to cached / deterministic paths so the app runs fully offline.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from .config import InferenceConfig, load_inference_config
from .schemas import InferenceLog

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
# Gemma 4 reasoning models wrap output in <thought>…</thought> (or <thinking>). Strip them.
_THOUGHT_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>|<thought>.*?</thought>", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict | None:
    """Best-effort: pull the first JSON object out of a model response."""
    if not text:
        return None
    # Drop reasoning blocks so a trailing JSON object is found cleanly.
    text = _THOUGHT_RE.sub("", text)
    # Strip ```json fences if present.
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def extract_json_array(text: str) -> list | None:
    """Pull the first JSON array out of a model response (after stripping reasoning)."""
    if not text:
        return None
    text = _THOUGHT_RE.sub("", text).strip()
    if text.startswith("```"):
        text = text.strip("`")
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        return None


@dataclass
class ChatResult:
    text: str
    log: InferenceLog


class GemmaClient:
    def __init__(self, cfg: InferenceConfig | None = None):
        self.cfg = cfg or load_inference_config()
        self._client = None  # lazy — avoids importing openai when offline

    @property
    def available(self) -> bool:
        return self.cfg.is_ready

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI  # local import keeps offline paths import-free

            self._client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)
        return self._client

    def list_models(self) -> list[str]:
        """Used by the GO/NO-GO smoke test."""
        client = self._ensure_client()
        return [m.id for m in client.models.list().data]

    def chat(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        system: str | None = None,
    ) -> ChatResult:
        model = model or self.cfg.model_id
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        client = self._ensure_client()
        start = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001 — surface as a logged error, don't crash the app
            latency = (time.perf_counter() - start) * 1000
            return ChatResult(
                text="",
                log=InferenceLog(model=model, latency_ms=latency, status="error", note=str(exc)[:200]),
            )

        latency = (time.perf_counter() - start) * 1000
        usage = getattr(resp, "usage", None)
        text = resp.choices[0].message.content or ""
        return ChatResult(
            text=text,
            log=InferenceLog(
                model=model,
                latency_ms=latency,
                status="ok",
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
            ),
        )

    def chat_json(self, prompt: str, **kwargs) -> tuple[dict | None, InferenceLog]:
        """Chat, parse JSON, retry ONCE with a stricter nudge on malformed output."""
        result = self.chat(prompt, **kwargs)
        parsed = extract_json(result.text)
        if parsed is not None or result.log.status == "error":
            return parsed, result.log

        # Retry once, asking explicitly for bare JSON.
        retry = self.chat(prompt + "\n\nReturn ONLY valid JSON. No prose, no code fences.", **kwargs)
        parsed = extract_json(retry.text)
        log = retry.log.model_copy(update={"status": "retried" if parsed else "error"})
        return parsed, log
