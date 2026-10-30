# TrendLens — judge FAQ & defense

**"Dash Social / Pendulum / Tableau already do analytics."**
We focus on one narrow thing those tools don't foreground: the mismatch between a consumer *need* and
competitor *messaging*. We expose contradicting evidence and turn only supported gaps into testable
hypotheses. We don't claim existing tools can't do similar analysis — we make this specific workflow
fast, grounded, and falsifiable.

**"How central is the open-source AI?"**
Gemma 4 (open-weight) does theme extraction, evidence-grounded synthesis, and experiment drafting —
remove it and the product loses its language understanding. Scoring is deterministic *on purpose*, so
AI augments judgment instead of fabricating numbers. Repo is public under MIT.

**"Is this real data?"**
No — it's clearly-labeled synthetic demonstration data (`is_synthetic=true`). We never present it as a
real market trend. The pipeline accepts sourced public data with provenance the same way.

**"How do I know the insights aren't hallucinated?"**
Every opportunity cites real dataset record ids, and the grounded explanation is rejected if it cites
an id outside the evidence set. Counts, shares, and scores are computed in `trendlens/scoring.py`, not
by the model.

**"How is the model served?"**
Gemma 4 (open-weight) runs via the Google Gemini API's OpenAI-compatible endpoint
(`https://generativelanguage.googleapis.com/v1beta/openai/`). We log model, latency, and token usage
per call. The inference layer is provider-agnostic — the same client also runs Gemma on DigitalOcean
Gradient AI by changing two env vars, so we're not locked to one host.

**"What are the limitations?"**
Small sample, heuristic scores (not statistically comparable market share), no winner prediction, no
ROI promise, no calibrated confidence. These are stated in the UI and README.

**"What would you build next?"**
Sourced live data ingestion, optional embedding retrieval (all-MiniLM-L6-v2), Gemma multimodal for
campaign images, and a provider-abstraction + latency/cost dashboard.
