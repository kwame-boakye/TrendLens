"""Load runtime configuration: env vars (secrets) + themes.yaml (vertical)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
THEMES_PATH = ROOT / "config" / "themes.yaml"

load_dotenv(ROOT / ".env")  # no-op if the file is absent


@dataclass(frozen=True)
class InferenceConfig:
    base_url: str
    model_id: str
    fallback_model_id: str
    api_key: str | None

    @property
    def is_ready(self) -> bool:
        """True when a non-placeholder key is present."""
        return bool(self.api_key) and self.api_key != "replace-me"


@dataclass(frozen=True)
class ThemeConfig:
    vertical: str
    brands: list[str]
    periods: dict
    themes: dict           # {key: {label, synonyms}}
    scoring: dict

    @property
    def vocab(self) -> list[str]:
        return list(self.themes.keys())

    def label(self, theme_key: str) -> str:
        return self.themes.get(theme_key, {}).get("label", theme_key)

    def synonyms(self, theme_key: str) -> list[str]:
        return self.themes.get(theme_key, {}).get("synonyms", [])


def load_inference_config() -> InferenceConfig:
    # Provider-neutral key lookup: Google AI Studio (default), with back-compat for
    # a DigitalOcean key if the team ever switches the base URL back.
    api_key = (
        os.getenv("INFERENCE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("DO_MODEL_ACCESS_KEY")
    )
    return InferenceConfig(
        base_url=os.getenv(
            "INFERENCE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
        ),
        model_id=os.getenv("GEMMA_MODEL_ID", "gemma-4-12b-it"),
        fallback_model_id=os.getenv("FALLBACK_MODEL_ID", "gemma-3-27b-it"),
        api_key=api_key,
    )


@lru_cache(maxsize=1)
def load_theme_config(path: str | None = None) -> ThemeConfig:
    p = Path(path) if path else THEMES_PATH
    raw = yaml.safe_load(p.read_text())
    return ThemeConfig(
        vertical=raw["vertical"],
        brands=raw.get("brands", []),
        periods=raw.get("periods", {}),
        themes=raw.get("themes", {}),
        scoring=raw.get("scoring", {}),
    )
