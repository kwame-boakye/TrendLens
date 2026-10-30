# 🔭 TrendLens

**Marketing opportunity discovery & falsification — powered by open-weight Gemma 4 on DigitalOcean.**

Dashboards tell marketers *what people say*. TrendLens tests whether a **customer need is
missing from competitor messaging** — surfaces the evidence for and against each gap, lets you
**falsify** a gap by adding a contradicting competitor message (the score drops in front of you),
and turns only *supported* gaps into a testable A/B experiment.

**The one-line demo:** *open TrendLens, pick a customer need, and it shows — with the actual
customer quotes as proof — the gap your competitors are missing; add a competitor message covering
that need and the gap score objectively drops, proving the recommendation is driven by evidence, not
guesswork.*

**Why the output is objective (not "cut spend 20%").** Scrapers and dashboards give raw research;
anyone can produce a confident-sounding number. TrendLens refuses to assert a magnitude it didn't
measure. Every opportunity score is a published formula over **auditable record counts**, confidence
is a **fixed deterministic rule** (score + sample size + contradictions), each claim **cites real
record IDs**, and the recommendation is an **A/B experiment with a defined success metric** — i.e. a
way to *measure* the effect, not a fabricated percentage. The "How we arrived at this" audit trail in
the app shows the exact inputs for any recommendation.

**Data: we analyze, we don't scrape.** Collection (social listening, app-store/review APIs, scraping
agents) is a solved, commoditized layer — TrendLens is the missing *analysis & validation* layer that
sits on top of it. You bring consumer records + competitor campaigns as CSV (from any source, with
provenance); TrendLens flags data-reliability risks (evidence concentrated in one brand/source — a
manipulation/astroturf signal) and never lets an unsupported claim through.

Built for the MLH open-source AI challenges using **Google Gemma 4** (open-weight)
([`mlh.link/gemma`](https://mlh.link/gemma)), served via the Google Gemini API's
OpenAI-compatible endpoint. The inference layer is provider-agnostic — it also runs
on **DigitalOcean Gradient AI** ([`mlh.link/hf26-digitalocean`](https://mlh.link/hf26-digitalocean))
by changing two env vars.

---

## Why this isn't just another dashboard

| Most tools | TrendLens |
|---|---|
| Report what consumers say | Compare consumer **need** vs competitor **messaging** |
| Confident single scores | Transparent components + sample sizes + **uncertainty** |
| Insights you can't check | Every claim cites **real dataset record ids** |
| Static | **Add counterevidence → the gap score falls** (falsifiable) |

## How open-weight AI is load-bearing

Gemma 4 (open-weight) does the language work — theme extraction, evidence-grounded synthesis, and
experiment drafting. **Deterministic Python does all counting and scoring**, so no number shown to a
user is model-invented. If Gemma is unreachable, a keyword fallback keeps the whole app running
offline — the AI is central, not decorative.

## Architecture

```
CSV ─▶ schema validation (Pydantic) ─▶ Gemma theme extraction (cached)
    ─▶ deterministic aggregation + gap scoring ─▶ evidence retrieval (SQLite)
    ─▶ Gemma evidence-grounded explanation ─▶ Streamlit UI
    ─▶ add counterevidence ─▶ re-score ─▶ Gemma experiment draft
```

- **Inference:** Gemma 4 via the Google Gemini API, OpenAI-compatible
  (`https://generativelanguage.googleapis.com/v1beta/openai/`). Swappable to DigitalOcean Gradient AI.
- **Data:** CSV + in-memory SQLite. No database service.
- **UI:** Streamlit (Opportunity Board · Evidence Inspector · Experiment Builder). Charts: Plotly.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env            # then paste a free Google AI Studio key (aistudio.google.com/apikey)
python scripts/smoke_inference.py   # GO/NO-GO: lists your Gemma ids + confirms live access & latency

python scripts/generate_data.py     # (re)generate the labeled synthetic dataset
python scripts/precompute.py        # classify dataset into cache (works offline too)
streamlit run app/streamlit_app.py
```

### Environment variables (`.env`)

| Var | Purpose |
|---|---|
| `INFERENCE_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` (Gemini/Gemma) |
| `GEMMA_MODEL_ID` | `gemma-4-12b-it` (confirm your exact id via `smoke_inference.py`) |
| `FALLBACK_MODEL_ID` | `gemma-3-27b-it` — open-weight fallback if the Gemma 4 id isn't enabled |
| `INFERENCE_API_KEY` | your Google AI Studio key — **never committed** |

## Sample data

All rows are **clearly-labeled synthetic** (`is_synthetic=true`, `synthetic://` provenance URLs) —
never presented as real market data. The set is engineered so the demo has one strong gap (**taste**),
one already-covered theme (**affordability**), and one **insufficient-evidence** theme
(**sustainability**). Swap vertical/brands/themes in [`config/themes.yaml`](config/themes.yaml) with
zero code changes.

## Testing

```bash
pytest -q
```

Covers schema invariants and deterministic scoring, including the signature negative test:
*adding a competitor message that emphasizes a theme lowers that theme's gap score.*

## Measured results

> Fill in after measurement (do not use placeholders in the submission):
> theme-extraction agreement on gold labels · evidence-citation validity · JSON parse success ·
> median & p95 Gemma latency · end-to-end demo completion rate · scoring correctness on fixtures.

## Limitations

Small, partly synthetic sample. Scores are heuristics, **not** statistically comparable market share.
TrendLens never claims a guaranteed market gap, calibrated AI confidence, or guaranteed ROI, and never
predicts which A/B variant will win.

## Team roles

- **Engineer A** — AI & inference (DO connectivity, prompts, classification, grounded synthesis, JSON validation, caching).
- **Engineer B** — data & app (ingestion, scoring, evidence retrieval, Streamlit, counterevidence, deploy, tests).
- **Marketing owner** — vertical/brands, sourced/labeled data, gold cases, pitch, judge FAQ, business-claim veto.

## License

Code: **MIT** (see [LICENSE](LICENSE)). Gemma weights: Google's Gemma Terms of Use.
