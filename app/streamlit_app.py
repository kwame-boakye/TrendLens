"""TrendLens — turn any company's customer + competitor data into marketing decisions.

Industry-agnostic. A company uploads customer feedback + competitor data (+ optional
past campaign results), TrendLens analyses, and returns ranked, evidence-backed
opportunities with a recommended next step and how to measure it. The music/streaming
example is just sample data — swap it for any industry.

Workflow: Upload → Analyse (Gemma extracts needs) → Opportunities (needs vs competitor
messaging) → Validate (evidence + live counterevidence) → Recommend (next step + A/B test).

Run from repo root:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from trendlens.classify import classify_records  # noqa: E402
from trendlens.config import ROOT, load_theme_config  # noqa: E402
from trendlens.data_io import build_index, load_campaigns, load_posts  # noqa: E402
from trendlens.discover import build_dynamic_config, classify_constrained, open_extract  # noqa: E402
from trendlens.evidence import explain_opportunity  # noqa: E402
from trendlens.experiment import generate_experiment  # noqa: E402
from trendlens.inference import GemmaClient  # noqa: E402
from trendlens.ingest import parse_campaign_results, parse_competitors, parse_feedback  # noqa: E402
from trendlens.recommend import confidence_label, executive_brief  # noqa: E402
from trendlens.schemas import CompetitorCampaign  # noqa: E402
from trendlens.scoring import rank_opportunities  # noqa: E402

st.set_page_config(page_title="TrendLens", page_icon="🔭", layout="wide")

LOGO = ROOT / "assets" / "trendlens_logo.jpg"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Montserrat:wght@600;700;800&display=swap');
    html, body, [class*="css"], .stMarkdown, p, div, span, label, input, textarea, button,
    [data-testid="stMetricValue"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4, h5, [data-testid="stHeading"] {
        font-family: 'Montserrat', sans-serif !important; font-weight: 700; letter-spacing: -0.01em;
    }
    [data-testid="stSidebar"] h2 { font-family: 'Montserrat', sans-serif !important; }
    [data-testid="stSidebar"] { background: #6E1322; }
    [data-testid="stSidebar"] * { color: #F7E9EC !important; }
    .tl-step { background:#FBEEF0; border-radius:12px; padding:14px 16px; height:100%; }
    .tl-step b { color:#8B1A2B; }
    .tl-badge-high { background:#8B1A2B; color:#fff; padding:2px 10px; border-radius:20px; font-size:12px; }
    .tl-badge-med { background:#E7C3CB; color:#6E1322; padding:2px 10px; border-radius:20px; font-size:12px; }
    .tl-stat { background:#FBEEF0; border-radius:12px; padding:14px; }
    .tl-stat .big { font-size:26px; font-weight:700; color:#8B1A2B; }
    </style>
    """,
    unsafe_allow_html=True,
)

client = GemmaClient()
CLIENT_READY = client.available

st.session_state.setdefault("analysis", None)
st.session_state.setdefault("extra", [])
st.session_state.setdefault("sel", None)


# ------------------------------------------------------------------ helpers --
def run_analysis(posts, campaigns, tc, company, campaign_perf, live, dynamic=False, progress=None):
    if not posts or not campaigns:
        return None, "Need at least some customer feedback AND competitor data."
    cap_p, cap_c = 40, 24
    note = ""
    if len(posts) > cap_p or len(campaigns) > cap_c:
        note = f"Analysed the first {cap_p} feedback + {cap_c} competitor rows (demo cap)."
        posts, campaigns = posts[:cap_p], campaigns[:cap_c]

    if dynamic and live:
        # Industry-agnostic path: discover the themes from the uploaded data itself.
        raw = open_extract([(p.id, p.text) for p in posts], client, progress=progress)
        tc, cls_posts = build_dynamic_config(raw, tc, company)
        if not tc.vocab:
            return None, "Couldn't extract clear needs from the feedback — try more comments."
        cls_camps = classify_constrained([(c.id, c.message) for c in campaigns], tc.vocab,
                                         client, progress=progress)
    else:
        cls_posts = classify_records([(p.id, p.text) for p in posts], client=client, tc=tc, live=live)
        cls_camps = classify_records([(c.id, c.message) for c in campaigns], client=client, tc=tc, live=live)
    return {
        "company": company, "tc": tc, "posts": posts, "campaigns": campaigns,
        "cls_posts": cls_posts, "cls_camps": cls_camps, "campaign_perf": campaign_perf,
        "note": note,
    }, None


def friendly_title(label: str) -> str:
    return f"Focus on {label.lower()}"


def badge_html(conf: str) -> str:
    return (f"<span class='tl-badge-high'>High opportunity</span>" if conf == "high"
            else f"<span class='tl-badge-med'>{conf.title()} opportunity</span>")


# -------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🔭 TrendLens")
    st.caption("From market research to marketing decisions")
    st.markdown("---")
    if st.session_state.pop("_goto_results", False):
        st.session_state["navradio"] = "📊 Results"  # set before the widget is created
    nav = st.radio("Navigate", ["🏠 Home", "📊 Results"], key="navradio", label_visibility="collapsed")
    st.markdown("---")
    st.caption("🟢 Live Gemma 4" if CLIENT_READY else "⚪ Offline fallback")


# ==================================================================== HOME ===
def render_home():
    st.image(str(LOGO), width=440)
    st.write("Turn customer feedback and competitor insights into clear marketing recommendations — for **any company, any industry**.")

    cols = st.columns(4)
    steps = [
        ("1", "Upload your data", "Customer reviews, competitor data, campaign results (if available)"),
        ("2", "We analyse", "Find patterns, opportunities and gaps"),
        ("3", "Get recommendations", "See what to test and why"),
        ("4", "Take action", "Run a campaign and measure results"),
    ]
    for col, (n, t, d) in zip(cols, steps):
        col.markdown(f"<div class='tl-step'><b>{n} · {t}</b><br><small>{d}</small></div>", unsafe_allow_html=True)

    st.markdown("### Upload your data")
    st.caption("Add the information you want TrendLens to analyse. CSV with a text/message column works best.")
    company = st.text_input("Company name", value="", placeholder="e.g. your company")

    u1, u2, u3 = st.columns(3)
    with u1:
        st.markdown("**💬 Customer feedback**  \nReviews, surveys, social media")
        fb_file = st.file_uploader("Customer feedback CSV", type=["csv"], key="fb", label_visibility="collapsed")
    with u2:
        st.markdown("**📊 Competitor data**  \nAds, campaigns, product updates")
        comp_file = st.file_uploader("Competitor CSV", type=["csv"], key="comp", label_visibility="collapsed")
    with u3:
        st.markdown("**📄 Campaign results** *(optional)*  \nYour past marketing performance")
        camp_file = st.file_uploader("Campaign results CSV", type=["csv"], key="camp", label_visibility="collapsed")

    deep = st.checkbox(
        "🔬 Deep theme discovery with Gemma (slower; needs live quota). "
        "Off = instant analysis using a cross-industry theme library.",
        value=False,
    )
    a1, a2 = st.columns([1, 1])
    analyse = a1.button("Analyse data →", type="primary", use_container_width=True)
    example = a2.button("Use example data", use_container_width=True)

    if analyse:
        if not fb_file or not comp_file:
            st.error("Please upload at least customer feedback AND competitor data (campaign results optional).")
            return
        posts = parse_feedback(pd.read_csv(fb_file), company or "Your company")
        camps = parse_competitors(pd.read_csv(comp_file))
        perf = parse_campaign_results(pd.read_csv(camp_file)) if camp_file else []
        if deep and CLIENT_READY:
            # Gemma discovers themes from the data (slower; needs quota).
            tc = load_theme_config()
            prog = st.progress(0.0, text="Gemma 4 is discovering themes…")
            analysis, err = run_analysis(
                posts, camps, tc, company or "Your company", perf,
                live=True, dynamic=True,
                progress=lambda d, t: prog.progress(min(1.0, d / max(t, 1)),
                                                    text=f"Gemma 4 is analysing… {d}/{t}"),
            )
            prog.empty()
        else:
            # Default: instant, reliable — keyword match against the cross-industry library.
            tc = load_theme_config(str(ROOT / "config" / "generic.yaml"))
            with st.spinner("Analysing…"):
                analysis, err = run_analysis(posts, camps, tc, company or "Your company", perf,
                                             live=False, dynamic=False)
        if err:
            st.error(err)
            return
        st.session_state.analysis = analysis
        st.session_state.extra = []
        st.session_state.sel = None
        st.session_state["_goto_results"] = True
        st.rerun()

    if example:
        tc = load_theme_config()
        posts, camps = load_posts(), load_campaigns()
        analysis, _ = run_analysis(posts, camps, tc, f"Example ({tc.vertical})", [], live=False)
        st.session_state.analysis = analysis
        st.session_state.extra = []
        st.session_state.sel = None
        st.session_state["_goto_results"] = True
        st.rerun()

    with st.expander("What does the example data look like? (brand-neutral — Spotify was only a reference)"):
        st.write(
            "The example is a fictional music/streaming industry (brands Streamly, Wavecast, "
            "Tunefox) — but TrendLens is **not** a streaming tool. Upload a retail, fintech, SaaS, "
            "or FMCG dataset and the same Discover → Connect → Validate → Recommend pipeline runs."
        )


# ================================================================= RESULTS ===
def render_results():
    a = st.session_state.analysis
    if not a:
        st.info("No analysis yet. Go to **Home**, upload data (or click *Use example data*), then Analyse.")
        return

    tc = a["tc"]
    posts, base_camps = a["posts"], a["campaigns"]
    cls_posts, base_cls = a["cls_posts"], a["cls_camps"]
    extra = st.session_state.extra
    campaigns = base_camps + [c for c, _ in extra]
    cls_camps = base_cls + [r for _, r in extra]
    opportunities = rank_opportunities(cls_posts, cls_camps, posts, campaigns, tc=tc)
    opp_by_theme = {o.theme: o for o in opportunities}
    post_by_id = {p.id: p for p in posts}
    cls_post_by_id = {c.id: c for c in cls_posts}
    top = [o for o in opportunities if o.status == "ok"]

    st.image(str(LOGO), width=240)
    if a.get("note"):
        st.caption("ℹ " + a["note"])
    st.title(f"Top Opportunities for {a['company']}")
    st.caption(f"Based on {len(posts)} customer comments, {len(campaigns)} competitor messages"
               + (", and uploaded campaign results." if a["campaign_perf"] else "."))
    if st.button("＋ New analysis"):
        st.session_state.analysis = None
        st.rerun()

    if not top:
        st.warning("No opportunity met the evidence bar. Try more customer feedback.")
        return

    if st.session_state.sel not in [o.theme for o in top]:
        st.session_state.sel = top[0].theme

    left, center, right = st.columns([1.1, 2.3, 1.2])

    # ---- left: ranked opportunity list
    with left:
        st.markdown("#### Opportunities")
        for i, o in enumerate(top, 1):
            conf = confidence_label(o)
            label = "🟥 High" if conf == "high" else "🟪 Med"
            if st.button(f"{i}. {friendly_title(o.label)}  ·  {label}", key=f"sel_{o.theme}",
                         use_container_width=True):
                st.session_state.sel = o.theme

    opp = opp_by_theme[st.session_state.sel]
    conn = build_index(posts, campaigns)

    # ---- center: the selected opportunity in detail
    with center:
        st.markdown(f"### {friendly_title(opp.label)} &nbsp; {badge_html(confidence_label(opp))}",
                    unsafe_allow_html=True)
        # live=False -> instant deterministic summary (never blocks the page).
        exp_text, grounded = explain_opportunity(opp, conn, client=client, tc=tc, live=False)
        st.write(exp_text if grounded else "Customers raise this need more than competitors emphasize it.")

        st.markdown("**Why we recommend this**")
        s1, s2, s3 = st.columns(3)
        s1.markdown(f"<div class='tl-stat'><span class='big'>{opp.consumer_share:.0%}</span><br>"
                    f"<small>of customer comments mention this need</small></div>", unsafe_allow_html=True)
        s2.markdown(f"<div class='tl-stat'><span class='big'>{opp.competitor_coverage:.0%}</span><br>"
                    f"<small>of competitor messages focus on it</small></div>", unsafe_allow_html=True)
        perf_val = next((r["value"] for r in a["campaign_perf"]
                         if opp.theme in r["label"].lower() or opp.label.lower() in r["label"].lower()), None)
        s3.markdown(f"<div class='tl-stat'><span class='big'>{(str(perf_val)+'%') if perf_val is not None else '—'}</span><br>"
                    f"<small>past-campaign conversion (if uploaded)</small></div>", unsafe_allow_html=True)

        st.markdown("**Key customer comments**")
        for pid in opp.support_ids[:3]:
            p = post_by_id.get(pid)
            if p:
                st.markdown(f"> “{p.text}”  \n<small>`[{pid}]` · {p.brand}</small>", unsafe_allow_html=True)

        st.markdown("**Competitor comparison** — share of each competitor's messages on this need")
        per_brand = {}
        for c in campaigns:
            themes = next((r.themes for r in cls_camps if r.id == c.id), [])
            per_brand.setdefault(c.brand, [0, 0])
            per_brand[c.brand][1] += 1
            if opp.theme in themes:
                per_brand[c.brand][0] += 1
        brands = list(per_brand.keys())
        vals = [(per_brand[b][0] / per_brand[b][1]) if per_brand[b][1] else 0 for b in brands]
        fig = go.Figure(go.Bar(x=brands, y=vals, marker_color="#8B1A2B"))
        fig.update_layout(height=240, yaxis_tickformat=".0%", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True, key="compchart")

        with st.expander("🔍 Validate — challenge this opportunity (add a competitor message)"):
            st.caption("Add a competitor message on this need. Gemma reclassifies it live and the "
                       "opportunity reassesses — new evidence can falsify it.")
            msg = st.text_area("New competitor message",
                               value=f"Never settle — {opp.label.lower()} like never before.", height=70)
            if st.button("Add & re-score with Gemma", type="primary", key="ce"):
                cid = f"cuser{len(extra)+1:02d}"
                with st.spinner("Gemma 4 is reclassifying…"):
                    cls = classify_records([(cid, msg)], client=client, tc=tc, live=True)[0]
                camp = CompetitorCampaign(id=cid, date="2026-10-30", brand="New competitor",
                                          channel="demo", message=msg, url="", is_synthetic=False)
                st.session_state.extra.append((camp, cls))
                st.rerun()
            for camp, cls in extra:
                st.markdown(f"- `[{camp.id}]` → {cls.themes or '[none]'} — {camp.message}")
            if extra and st.button("Reset", key="rst"):
                st.session_state.extra = []
                st.rerun()

    # ---- right: recommended next step + how to measure + export
    with right:
        st.markdown("#### Recommended next step")
        if st.button("Generate A/B test with Gemma", type="primary", use_container_width=True):
            summary, _ = explain_opportunity(opp, conn, client=client, tc=tc, live=False)
            with st.spinner("Gemma 4 is drafting the experiment…"):
                st.session_state[f"exp_{opp.theme}"] = generate_experiment(opp, summary, client=client, tc=tc)
        exp = st.session_state.get(f"exp_{opp.theme}")
        if exp:
            st.markdown(f"**Test:** {exp.hypothesis}")
            st.markdown(f"**A:** {exp.a_copy}")
            st.markdown(f"**B:** {exp.b_copy}")
            st.markdown(f"**Audience:** {exp.audience}")
            st.markdown(f"**Metric:** {exp.metric}")
        else:
            st.info(f"Test a campaign emphasizing **{opp.label.lower()}**. Click above for Gemma's A/B copy.")

        st.markdown("**How to measure success**")
        metric = exp.metric if exp else "conversion rate"
        for item in [f"Track {metric}", "Compare cost per conversion vs other campaigns",
                     "Run the test for 4–8 weeks", "Use the same audience & budget for a fair comparison"]:
            st.markdown(f"- ✅ {item}")

        report = build_report(a["company"], opp, exp, posts)
        st.download_button("⬇ Export full report", report,
                           file_name=f"trendlens_{opp.theme}.md", use_container_width=True)


def build_report(company, opp, exp, posts) -> str:
    post_by_id = {p.id: p for p in posts}
    lines = [
        f"# TrendLens report — {company}", "",
        f"## Opportunity: {opp.label}", f"- Confidence: {confidence_label(opp)}",
        f"- Consumer demand: {opp.consumer_share:.0%} of comments",
        f"- Competitor coverage: {opp.competitor_coverage:.0%} of messages",
        f"- Supporting records: {', '.join(opp.support_ids[:15])}", "",
        "## Evidence (customer comments)",
    ]
    for pid in opp.support_ids[:5]:
        p = post_by_id.get(pid)
        if p:
            lines.append(f"- [{pid}] \"{p.text}\"")
    if exp:
        lines += ["", "## Recommended A/B test", f"- Hypothesis: {exp.hypothesis}",
                  f"- A: {exp.a_copy}", f"- B: {exp.b_copy}",
                  f"- Audience: {exp.audience}", f"- Metric: {exp.metric}"]
    if opp.limitations:
        lines += ["", "## Limitations"] + [f"- {l}" for l in opp.limitations]
    lines += ["", "_Scores are deterministic heuristics over a limited sample, not market share. "
              "TrendLens recommends a test, not a guaranteed outcome._"]
    return "\n".join(lines)


# -------------------------------------------------------------------- router
if nav == "🏠 Home":
    render_home()
else:
    render_results()
