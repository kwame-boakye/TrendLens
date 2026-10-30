"""TrendLens Streamlit app.

Three views: Opportunity Board, Evidence Inspector (with the live counterevidence
falsification demo), and Experiment Builder. The board reads cached/keyword labels
so it never blocks; Gemma 4 runs LIVE for the counterevidence reclassification, the
grounded explanation, and the experiment draft. Run from repo root:

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
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
from trendlens.schemas import ClassifiedRecord, CompetitorCampaign  # noqa: E402
from trendlens.scoring import rank_opportunities  # noqa: E402

st.set_page_config(page_title="TrendLens", page_icon="🔭", layout="wide")


@st.cache_data(show_spinner="Loading data and classifying (cached)…")
def load_base():
    tc = load_theme_config()
    posts = load_posts()
    campaigns = load_campaigns()
    # live=False: instant board (cached Gemma labels or keyword), never blocks on the UI.
    cls_posts = classify_records([(p.id, p.text) for p in posts], tc=tc, live=False)
    cls_camps = classify_records([(c.id, c.message) for c in campaigns], tc=tc, live=False)
    return tc, posts, campaigns, cls_posts, cls_camps


tc, posts, base_campaigns, cls_posts, base_cls_camps = load_base()
client = GemmaClient()
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
    "We don't tell you what people say — we test whether customer needs are missing "
    "from competitor messaging. Powered by **Gemma 4** (open-weight) via the Google Gemini API."
)
mode = "🟢 Live Gemma 4 ready" if client.available else "⚪ Offline (deterministic fallback)"
st.markdown(
    f"**Vertical:** {tc.vertical}  ·  **Brands:** {', '.join(tc.brands)}  ·  "
    f"**Data:** {len(posts)} posts, {len(campaigns)} campaigns (synthetic, labeled)  ·  {mode}"
)
st.info(
    "All records are clearly-labeled synthetic demonstration data. Scores are "
    "deterministic heuristics over a small sample — not statistically comparable "
    "market share.",
    icon="⚠️",
)

board, inspector, builder = st.tabs(
    ["📊 Opportunity Board", "🔍 Evidence & Counterevidence", "🧪 Experiment Builder"]
)

# --- Opportunity Board ---------------------------------------------------- #
with board:
    st.subheader("Opportunity Board")
    st.write(
        "Each card is a theme. **Gap** = consumers care but competitors aren't messaging "
        "it. Every score component and sample size is shown; tiny samples are flagged."
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

# --- Evidence Inspector + Counterevidence (the signature demo) ------------ #
with inspector:
    st.subheader("Evidence & Counterevidence")
    themes = [o.theme for o in opportunities]
    chosen = st.selectbox("Theme", themes, format_func=lambda t: tc.label(t))
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

        post_by_id = {p.id: p for p in posts}
        st.markdown("**Supporting consumer posts (real record ids):**")
        for pid in opp.support_ids[:6]:
            p = post_by_id.get(pid)
            if p:
                st.markdown(f"- `[{pid}]` ({p.brand}, {p.date}) — {p.text}")
        if not opp.support_ids:
            st.caption("No supporting records.")

    with colB:
        st.markdown("#### 🧪 Add counterevidence")
        st.caption(
            "Add a competitor campaign that emphasizes this theme. **Gemma 4 reclassifies "
            "it live**, competitor coverage rises, and the gap score should fall — new "
            "evidence can falsify an opportunity."
        )
        default_msg = f"Taste the difference — bold, refreshing {tc.label(chosen).lower()} you'll love."
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

# --- Experiment Builder --------------------------------------------------- #
with builder:
    st.subheader("Experiment Builder")
    st.write(
        "Turn a **supported** opportunity into a testable A/B experiment with Gemma 4. "
        "It proposes audience, a falsifiable hypothesis, A/B copy, and a success metric — "
        "and never predicts which variant wins."
    )
    supported = [o for o in opportunities if o.status == "ok"]
    if not supported:
        st.warning("No supported opportunities right now.")
    else:
        pick = st.selectbox("Opportunity", [o.theme for o in supported],
                            format_func=lambda t: tc.label(t), key="exp_pick")
        opp = opp_by_theme[pick]
        if st.button("Generate experiment with Gemma", type="primary"):
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
