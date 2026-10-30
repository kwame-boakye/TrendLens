# TrendLens — 90-second demo script

**Setup before you present:** run `python scripts/precompute.py` so the cache is warm, open
`streamlit run app/streamlit_app.py`, and have the Opportunity Board tab on screen.

**0–15s — the hook**
> "Dashboards tell marketers what people say. TrendLens tests whether a customer need is *missing*
> from competitor messaging — and then lets you try to prove yourself wrong."

**15–35s — the gap (Opportunity Board)**
Point to the **Taste** card. Show: high consumer demand (share of posts), low competitor coverage,
the four score components, and the sample size. Note the honesty: "it's a small, labeled synthetic
sample — a heuristic, not market share."

**35–55s — falsification (Evidence Inspector → counterevidence, slice 3)**
Open Evidence Inspector, show the supporting posts cited by **real record ids**. Then add a competitor
campaign that emphasizes taste → competitor coverage rises, the gap score **falls**.
> "New evidence can falsify an opportunity. That's the point."

**55–75s — turn a supported gap into a test (Experiment Builder, slice 4)**
Pick a still-supported opportunity → **Generate Experiment**. Show audience, falsifiable hypothesis,
A/B copy, and the success metric.
> "It designs the test — it does not predict the winner."

**75–90s — why it's different**
> "Gemma 4, open-weight, via the Google Gemini API does the language work. Deterministic Python does every
> number, each insight cites record ids, and the whole thing runs offline if inference drops. Not a
> chatbot, not another dashboard."

**Backup:** keep a screen recording of this exact flow in case live inference or deploy fails.
