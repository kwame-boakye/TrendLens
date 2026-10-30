"""TrendLens Streamlit app — Discover → Connect → Validate → Recommend.

The four tabs mirror the product workflow 1:1:
  ① Discover  — Gemma extracts the recurring customer NEEDS (not just +/- sentiment)
  ② Connect   — compare needs vs competitor messaging -> potential opportunities
  ③ Validate  — examine evidence, add contradicting competitor message, re-score (live Gemma)
  ④ Recommend — the CEO-ready brief + a testable A/B experiment

The board reads cached/keyword labels so it never blocks; Gemma 4 runs LIVE for the
counterevidence reclassification, the grounded explanation, the brief, and the
experiment. Run from repo root:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from trendlens.classify import classify_records  # noqa: E402
from trendlens.config import load_theme_config  # noqa: E402
from trendlens.data_io import build_index, load_campaigns, load_posts  # noqa: E402
from trendlens.evidence import explain_opportunity  # noqa: E402
from trendlens.experiment import generate_experiment  # noqa: E402
from trendlens.inference import GemmaClient  # noqa: E402
from trendlens.recommend import executive_brief  # noqa: E402
from trendlens.schemas import CompetitorCampaign  # noqa: E402
from trendlens.scoring import rank_opportunities  # noqa: E402

st.set_page_config(page_title="TrendLens", page_icon="🔭", layout="wide")


@st.cache_data(show_spinner="Loading data and classifying (cached)…")
def load_base():
    tc = load_theme_config()
    posts = load_posts()
    campaigns = load_campaigns()
    # live=False: instant board (cached Gemma labels or keyword), never blocks the UI.
    cls_posts = classify_records([(p.id, p.text) for p in posts], tc=tc, live=False)
    cls_camps = classify_records([(c.id, c.message) for c in campaigns], tc=tc, live=False)
    return tc, posts, campaigns, cls_posts, cls_camps


tc, posts, base_campaigns, cls_posts, base_cls_camps = load_base()
client = GemmaClient()
post_by_id = {p.id: p for p in posts}
cls_post_by_id = {c.id: c for c in cls_posts}
st.session_state.setdefault("extra", [])  # list of (CompetitorCampaign, ClassifiedRecord)


def current_campaigns():
    extra = st.session_state["extra"]
    camps = base_campaigns + [c for c, _ in extra]
    cls = base_cls_camps + [r for _, r in extra]
    return camps, cls


campaigns, cls_camps = current_campaigns()
opportunities = rank_opportunities(cls_posts, cls_camps, posts, campaigns, tc=tc)
opp_by_theme = {o.theme: o for o in opportunities}

# --- header --------------------------------------------------------------- #
st.title("🔭 TrendLens")
st.caption(
    "From market research to marketing decisions — for **any industry**. We connect "
    "customer needs with competitor messaging, let you check the evidence, and recommend "
    "what to test — powered by **Gemma 4** (open-weight) via the Google Gemini API."
)
mode = "🟢 Live Gemma 4 ready" if client.available else "⚪ Offline (deterministic fallback)"
st.markdown(
    f"**Discover → Connect → Validate → Recommend**  ·  "
    f"**Example dataset (swappable):** {tc.vertical}  ·  "
    f"{len(posts)} posts, {len(campaigns)} campaigns  ·  {mode}"
)
st.info(
    "TrendLens is industry-agnostic — point it at any industry's customer feedback + "
    "competitor campaigns (one config file). This demo uses a brand-neutral example; "
    "brands are fictional and all records are clearly-labeled synthetic data. Scores are "
    "deterministic heuristics over a small sample, not statistically comparable market share.",
    icon="⚠️",
)

discover, connect, validate, recommend = st.tabs(
    ["① Discover", "② Connect", "③ Validate", "④ Recommend"]
)

# --- ① Discover — what are customers saying? ------------------------------ #
with discover:
    st.subheader("① Discover — what are customers saying?")
    st.caption(
        "Gemma reads customer comments and extracts the recurring **needs** behind them — "
        "what people care about, not just whether a comment is positive or negative."
    )
    counts: Counter = Counter()
    sentiment_by_theme: dict[str, Counter] = {}
    for c in cls_posts:
        for t in c.themes:
            counts[t] += 1
            sentiment_by_theme.setdefault(t, Counter())[c.sentiment] += 1
    total = len(posts)
    for theme, n in counts.most_common():
        with st.container(border=True):
            sents = sentiment_by_theme.get(theme, Counter())
            st.markdown(
                f"**{tc.label(theme)}** — mentioned in **{n}/{total}** comments ({n/total:.0%})  ·  "
                f"🙂 {sents.get('pos',0)} · 😐 {sents.get('neu',0)} · 🙁 {sents.get('neg',0)}"
            )
            examples = [p for p in posts if theme in cls_post_by_id[p.id].themes][:2]
            for p in examples:
                st.caption(f"[{p.id}] “{p.text}”")

# --- ② Connect — where might competitors be falling short? ---------------- #
with connect:
    st.subheader("② Connect — where might competitors be falling short?")
    st.write(
        "Each card compares a customer need against competitor messaging. A **gap** = "
        "consumers care but competitors aren't emphasizing it. These are *potential* "
        "opportunities, not proven market gaps — every component and sample size is shown."
    )
    for opp in opportunities:
        header = f"{opp.label}  ·  score {opp.score:.2f}"
        if opp.status == "insufficient_evidence":
            header += "  ·  ⛔ insufficient evidence"
        with st.container(border=True):
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown(f"### {header}")
                st.markdown(
                    f"- Consumer demand (share of posts): **{opp.consumer_share:.0%}**\n"
                    f"- Competitor coverage (share of campaigns): **{opp.competitor_coverage:.0%}**\n"
                    f"- Supporting posts: **{opp.sample_sizes.get('support', 0)}**  ·  "
                    f"contradicting campaigns: **{len(opp.counter_ids)}**"
                )
                if opp.limitations:
                    st.caption("⚠ " + "  ·  ".join(opp.limitations))
            with c2:
                comp = opp.components
                fig = go.Figure(
                    go.Bar(
                        x=[comp.consumer_freq, comp.momentum,
                           comp.underrepresentation, comp.evidence_quality],
                        y=["Frequency", "Momentum", "Underrep.", "Evidence"],
                        orientation="h", marker_color="#2563eb",
                    )
                )
                fig.update_layout(height=180, margin=dict(l=0, r=0, t=10, b=0),
                                  xaxis_range=[0, 1], showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{opp.theme}")

# --- ③ Validate — does the opportunity hold up? --------------------------- #
with validate:
    st.subheader("③ Validate — does the opportunity hold up?")
    st.caption(
        "Our defining feature: don't blindly trust the AI. Examine the evidence, then "
        "challenge it — add a competitor message and watch the opportunity reassess itself."
    )
    themes = [o.theme for o in opportunities]
    chosen = st.selectbox("Customer need", themes, format_func=lambda t: tc.label(t))
    opp = opp_by_theme[chosen]
    conn = build_index(posts, campaigns)

    colA, colB = st.columns(2)
    with colA:
        st.metric("Gap score", f"{opp.score:.2f}")
        st.caption(f"Consumer demand {opp.consumer_share:.0%} · competitor coverage {opp.competitor_coverage:.0%}")
        explanation, grounded = explain_opportunity(opp, conn, client=client, tc=tc)
        if not grounded:
            st.error("Explanation cited a record id outside the evidence set — rejected.")
        else:
            st.markdown(f"**Grounded summary:** {explanation}")

        st.markdown("**Supporting consumer posts (real record ids):**")
        for pid in opp.support_ids[:6]:
            p = post_by_id.get(pid)
            if p:
                st.markdown(f"- `[{pid}]` ({p.brand}, {p.date}) — {p.text}")
        if not opp.support_ids:
            st.caption("No supporting records.")

    with colB:
        st.markdown("#### 🧪 Challenge it — add counterevidence")
        st.caption(
            "Add a competitor campaign that emphasizes this need. **Gemma 4 reclassifies "
            "it live**, competitor coverage rises, and the gap score should fall — new "
            "evidence can falsify an opportunity."
        )
        default_msg = f"Never settle — {tc.label(chosen).lower()} like never before. Switch today."
        new_msg = st.text_area("New competitor message", value=default_msg, height=90)
        if st.button("Add & re-score with Gemma", type="primary"):
            cid = f"cuser{len(st.session_state['extra']) + 1:02d}"
            with st.spinner("Gemma 4 is classifying the new message…"):
                cls = classify_records([(cid, new_msg)], client=client, tc=tc, live=True)[0]
            camp = CompetitorCampaign(id=cid, date="2026-10-30", brand="Competitor",
                                      channel="demo", message=new_msg, url=f"synthetic://{cid}",
                                      is_synthetic=True)
            st.session_state["extra"].append((camp, cls))
            st.success(f"Gemma tagged `[{cid}]` → themes {cls.themes or '[none]'}. Re-scoring…")
            st.rerun()

        if st.session_state["extra"]:
            st.markdown("**Counterevidence added this session:**")
            for camp, cls in st.session_state["extra"]:
                st.markdown(f"- `[{camp.id}]` → {cls.themes or '[none]'} — {camp.message}")
            if st.button("Reset counterevidence"):
                st.session_state["extra"] = []
                st.rerun()

# --- ④ Recommend — what should the company do next? ----------------------- #
with recommend:
    st.subheader("④ Recommend — what should the company do next?")
    st.caption(
        "The final output: one validated recommendation a marketer can take into a "
        "meeting, plus the experiment to test it. Prioritization & confidence are "
        "deterministic; Gemma 4 writes the grounded narrative and the A/B copy."
    )

    st.markdown("### 📋 Executive brief")
    if st.button("Generate executive brief with Gemma", type="primary", key="brief_btn"):
        conn = build_index(posts, campaigns)
        with st.spinner("Gemma 4 is writing the brief…"):
            brief = executive_brief(opportunities, posts, conn, client=client, tc=tc)
        if brief is None:
            st.warning("No validated opportunity meets the evidence bar yet.")
        else:
            badge = {"high": "🟢 High", "medium": "🟡 Medium", "low": "🔴 Low"}[brief.confidence]
            st.markdown(f"## {brief.headline}")
            st.markdown(f"**Confidence (deterministic): {badge}**  ·  Top need: **{brief.top_label}**  ·  Runner-up: {brief.runner_up}")
            st.markdown(f"**Why:** {brief.rationale}")
            st.success(f"**Recommended next step:** {brief.recommended_action}")
            if brief.watchouts:
                st.markdown("**Watch-outs (incl. data-reliability):**")
                for w in brief.watchouts:
                    st.markdown(f"- ⚠ {w}")
            top = opp_by_theme.get(brief.top_theme)
            if top:
                c, ss = top.components, top.sample_sizes
                with st.expander("🔎 How we arrived at this — audit trail (every number is traceable)"):
                    st.markdown(
                        f"**Opportunity score = {top.score:.2f}**, computed deterministically:\n\n"
                        f"`0.30×{c.consumer_freq} (demand) + 0.25×{c.momentum} (momentum) "
                        f"+ 0.25×{c.underrepresentation} (underrep) + 0.20×{c.evidence_quality} (evidence) "
                        f"− penalties`"
                    )
                    st.markdown(
                        f"- **Consumer demand** = {ss.get('support',0)} / {ss.get('posts',0)} posts "
                        f"= {top.consumer_share:.0%}\n"
                        f"- **Competitor coverage** = {len(top.counter_ids)} / {ss.get('campaigns',0)} campaigns "
                        f"= {top.competitor_coverage:.0%}\n"
                        f"- **Confidence** `{brief.confidence}` is a fixed rule over score + sample size + contradictions"
                    )
                    st.markdown("**Supporting records:** " + ", ".join(f"`[{i}]`" for i in top.support_ids[:12]))
                    st.caption(
                        "We do NOT output an arbitrary magnitude (e.g. 'cut spend 20%'). We output an "
                        "evidence-ranked gap and the experiment to measure the real effect — objective "
                        "and reproducible, not a guess."
                    )

    st.divider()
    st.markdown("### 🧪 Campaign experiment to test")
    supported = [o for o in opportunities if o.status == "ok"]
    if not supported:
        st.warning("No supported opportunities right now.")
    else:
        pick = st.selectbox("Opportunity to test", [o.theme for o in supported],
                            format_func=lambda t: tc.label(t), key="exp_pick")
        opp = opp_by_theme[pick]
        if st.button("Generate A/B experiment with Gemma", type="primary", key="exp_btn"):
            conn = build_index(posts, campaigns)
            summary, _ = explain_opportunity(opp, conn, client=client, tc=tc)
            with st.spinner("Gemma 4 is drafting the experiment…"):
                exp = generate_experiment(opp, summary, client=client, tc=tc)
            st.markdown(f"**Hypothesis:** {exp.hypothesis}")
            c1, c2 = st.columns(2)
            c1.markdown(f"**A (control)**\n\n{exp.a_copy}")
            c2.markdown(f"**B (variant)**\n\n{exp.b_copy}")
            st.markdown(f"**Audience:** {exp.audience}")
            st.markdown(f"**Success metric:** {exp.metric}")
            if exp.limitations:
                st.caption("Limitations: " + " · ".join(exp.limitations))
