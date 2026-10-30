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
    .eyebrow { color:#8B1A2B; font-size:12px; letter-spacing:.14em; font-weight:700; }
    .kpi { background:#fff; border:1px solid #F0DADE; border-radius:14px; padding:16px 18px; height:100%; }
    .kpi .lbl { font-size:11px; letter-spacing:.08em; color:#8a6b70; font-weight:700; }
    .kpi .num { font-size:30px; font-weight:800; color:#8B1A2B; line-height:1.15;
                font-family:'Montserrat',sans-serif; }
    .kpi .sub { font-size:12px; color:#6b5458; }
    .hero { background:#FBEEF0; border:1px solid #F0C9D1; border-radius:18px; padding:22px 24px; }
    .hero .tag { background:#6E1322; color:#fff; font-size:11px; letter-spacing:.08em;
                 padding:4px 12px; border-radius:20px; font-weight:700; }
    .scorebox { background:#fff; border-radius:14px; padding:16px; }
    .scorebox .s { font-size:34px; font-weight:800; color:#8B1A2B; font-family:'Montserrat',sans-serif; }
    .barbg { background:#F0DADE; border-radius:6px; height:10px; width:100%; overflow:hidden; }
    .barfg { background:#8B1A2B; height:10px; border-radius:6px; }
    .barfg.light { background:#E7A9B4; }
    .pill-high { background:#8B1A2B; color:#fff; padding:2px 12px; border-radius:20px; font-size:12px; }
    .pill-med { background:#E7C3CB; color:#6E1322; padding:2px 12px; border-radius:20px; font-size:12px; }
    .pill-low { background:#F3E4E7; color:#9b7b80; padding:2px 12px; border-radius:20px; font-size:12px; }
    .otable { width:100%; border-collapse:collapse; }
    .otable td, .otable th { padding:10px 8px; border-bottom:1px solid #F0DADE; font-size:13px; vertical-align:middle; }
    .otable th { text-align:left; color:#8a6b70; font-size:11px; letter-spacing:.04em; }
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
        st.session_state["navradio"] = "📊 Analysis Results"  # set before the widget is created
    nav = st.radio("Navigate", ["🏠 Home", "📊 Analysis Results"], key="navradio", label_visibility="collapsed")
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

    if not top:
        st.title("Your Marketing Opportunity Report")
        st.warning("No opportunity met the evidence bar yet. Try more customer feedback.")
        if st.button("＋ New analysis"):
            st.session_state.analysis = None
            st.rerun()
        return

    # ---- header
    ch, cb = st.columns([4, 1])
    with ch:
        st.markdown("<div class='eyebrow'>ANALYSIS RESULTS</div>", unsafe_allow_html=True)
        st.title("Your Marketing Opportunity Report")
        st.write(f"We analysed {a['company']}'s customer feedback, competitor messaging"
                 + (", and campaign results" if a["campaign_perf"] else "")
                 + " to uncover the biggest opportunities for your next marketing campaign.")
    with cb:
        st.download_button("⬇ Export Report",
                           build_report(a["company"], top[0],
                                        st.session_state.get(f"exp_{top[0].theme}"), posts),
                           file_name=f"trendlens_{a['company']}.md".replace(" ", "_"),
                           use_container_width=True)
        if st.button("＋ New analysis", use_container_width=True):
            st.session_state.analysis = None
            st.rerun()
    if a.get("note"):
        st.caption("ℹ " + a["note"])

    # ---- KPI cards
    brands = sorted({c.brand for c in base_camps})
    perf = a["campaign_perf"]
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='kpi'><div class='lbl'>CUSTOMER FEEDBACK</div>"
                f"<div class='num'>{len(posts):,}</div>"
                f"<div class='sub'>customer comments analysed</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><div class='lbl'>COMPETITOR MESSAGING</div>"
                f"<div class='num'>{len(brands)}</div>"
                f"<div class='sub'>competitors: {', '.join(brands[:5]) or '—'}</div></div>",
                unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'><div class='lbl'>CAMPAIGN RESULTS</div>"
                f"<div class='num'>{len(perf)}</div>"
                f"<div class='sub'>past campaigns analysed</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'><div class='lbl'>TOTAL DATA POINTS</div>"
                f"<div class='num'>{len(posts)+len(campaigns)+len(perf):,}</div>"
                f"<div class='sub'>insights processed by TrendLens</div></div>", unsafe_allow_html=True)
    st.write("")

    # ---- hero: top opportunity
    hero = top[0]
    score100 = int(round(hero.score * 100))
    checks = [("High customer demand", hero.consumer_share >= 0.30),
              ("Low competitor coverage", hero.competitor_coverage <= 0.15),
              ("Strong potential for differentiation",
               (hero.consumer_share - hero.competitor_coverage) >= 0.20)]
    check_html = "".join(f"<div>{'✅' if ok else '▫️'} {t}</div>" for t, ok in checks)
    h1, h2 = st.columns([2.2, 1])
    with h1:
        st.markdown(
            f"<div class='hero'><span class='tag'>TOP OPPORTUNITY</span>"
            f"<h2 style='margin:.4rem 0'>{hero.label}</h2>"
            f"<p>Customers frequently mention {hero.label.lower()}, but competitor messaging "
            f"{'and past campaigns ' if perf else ''}do not emphasize it. This creates a potential "
            f"opportunity to differentiate {a['company']} in your next marketing campaign.</p>"
            f"<div style='display:flex; gap:32px; margin-top:12px'>"
            f"<div><span style='font-size:28px;font-weight:800;color:#8B1A2B'>{hero.consumer_share:.0%}</span>"
            f"<br><small>of customer comments mention {hero.label.lower()}</small></div>"
            f"<div><span style='font-size:28px;font-weight:800;color:#8B1A2B'>{hero.competitor_coverage:.0%}</span>"
            f"<br><small>of competitor messages focus on it</small></div></div></div>",
            unsafe_allow_html=True)
    with h2:
        st.markdown(
            f"<div class='scorebox'><div style='color:#8a6b70;font-size:12px;font-weight:700'>Opportunity Score</div>"
            f"<div class='s'>{score100}<span style='font-size:16px;color:#9b7b80'>/100</span></div>"
            f"<div class='barbg' style='margin:8px 0 12px'><div class='barfg' style='width:{score100}%'></div></div>"
            f"<div style='font-size:12.5px;color:#4a3b3e;line-height:1.7'>{check_html}</div></div>",
            unsafe_allow_html=True)
    st.write("")

    # ---- bottom: ranked table + grouped chart
    bl, br = st.columns([1.1, 1])
    with bl:
        st.markdown("#### All Opportunities Identified")
        rows = ""
        for i, o in enumerate(top, 1):
            conf = confidence_label(o)
            cls_ = "high" if conf == "high" else "med" if conf == "medium" else "low"
            txt = "High" if conf == "high" else "Medium" if conf == "medium" else "Lower"
            d, cov = int(round(o.consumer_share * 100)), int(round(o.competitor_coverage * 100))
            rows += (f"<tr><td>{i}</td><td><b>{o.label}</b></td>"
                     f"<td>{d}%<div class='barbg'><div class='barfg' style='width:{d}%'></div></div></td>"
                     f"<td>{cov}%<div class='barbg'><div class='barfg light' style='width:{cov}%'></div></div></td>"
                     f"<td><b>{int(round(o.score*100))}</b> &nbsp;<span class='pill-{cls_}'>{txt}</span></td></tr>")
        st.markdown(
            "<table class='otable'><tr><th>#</th><th>Opportunity</th><th>Customer Demand</th>"
            "<th>Competitor Coverage</th><th>Opportunity Score</th></tr>" + rows + "</table>",
            unsafe_allow_html=True)
    with br:
        st.markdown("#### Customer Demand vs. Competitor Coverage")
        labels = [o.label for o in top]
        fig = go.Figure()
        fig.add_bar(name="Customer demand", x=labels, y=[o.consumer_share for o in top],
                    marker_color="#6E1322")
        fig.add_bar(name="Competitor coverage", x=labels, y=[o.competitor_coverage for o in top],
                    marker_color="#E7A9B4")
        fig.update_layout(barmode="group", height=330, yaxis_tickformat=".0%",
                          margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.18),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, key="demandchart")

    # ---- explore an opportunity (keeps all interactive features)
    st.divider()
    st.markdown("### 🔎 Explore an opportunity")
    sel = st.selectbox("Opportunity", [o.theme for o in top],
                       format_func=lambda t: opp_by_theme[t].label, key="explore_sel")
    opp = opp_by_theme[sel]
    conn = build_index(posts, campaigns)
    d1, d2 = st.columns([1.4, 1])
    with d1:
        exp_text, grounded = explain_opportunity(opp, conn, client=client, tc=tc, live=False)
        st.write(exp_text if grounded else "Customers raise this need more than competitors emphasize it.")
        st.markdown("**Key customer comments**")
        for pid in opp.support_ids[:3]:
            p = post_by_id.get(pid)
            if p:
                st.markdown(f"> “{p.text}”  \n<small>`[{pid}]` · {p.brand}</small>", unsafe_allow_html=True)
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
    with d2:
        st.markdown("**Recommended next step**")
        if st.button("Generate A/B test with Gemma", type="primary", use_container_width=True):
            summary, _ = explain_opportunity(opp, conn, client=client, tc=tc, live=False)
            with st.spinner("Gemma 4 is drafting the experiment…"):
                st.session_state[f"exp_{opp.theme}"] = generate_experiment(opp, summary, client=client, tc=tc)
        exp = st.session_state.get(f"exp_{opp.theme}")
        if exp:
            st.markdown(f"**Test:** {exp.hypothesis}")
            st.markdown(f"**A:** {exp.a_copy}")
            st.markdown(f"**B:** {exp.b_copy}")
            st.markdown(f"**Audience:** {exp.audience}  \n**Metric:** {exp.metric}")
        else:
            st.info(f"Test a campaign emphasizing **{opp.label.lower()}**. Click above for Gemma's A/B copy.")
        st.markdown("**How to measure success**")
        metric = exp.metric if exp else "conversion rate"
        for item in [f"Track {metric}", "Compare cost per conversion vs other campaigns",
                     "Run the test for 4–8 weeks", "Use the same audience & budget for a fair comparison"]:
            st.markdown(f"- ✅ {item}")


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
