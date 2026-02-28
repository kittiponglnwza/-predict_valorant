"""
ui/page_analysis.py — Model Analysis & Insights page for Football AI Nexus Engine
Dark minimal design with custom Streamlit CSS injection
"""
import streamlit as st
import pandas as pd

from src.analysis import run_monte_carlo, backtest_roi, analyze_draw_calibration, run_feature_importance
from utils import silent


# ── CSS ─────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #080c10 !important;
}
[data-testid="stAppViewContainer"] > .main {
    background: #080c10;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: #0e1318 !important; border-right: 1px solid rgba(255,255,255,0.05); }

/* Grid background */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.012) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.012) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

/* ── Typography ── */
*, p, span, div, label, li {
    font-family: 'DM Mono', monospace !important;
    color: #e8edf2;
}
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

/* ── Page header ── */
.nexus-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding: 8px 0 28px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 8px;
}
.nexus-brand {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #5a6880 !important;
    margin-bottom: 6px;
}
.nexus-brand span { color: #63dc8c !important; }
.nexus-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 30px !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #e8edf2 0%, #63dc8c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0 !important;
    padding: 0 !important;
}
.nexus-badge {
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 100px;
    border: 1px solid rgba(99,220,140,0.3);
    color: #63dc8c !important;
    background: rgba(99,220,140,0.06);
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    gap: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    color: #5a6880 !important;
    border-bottom: 2px solid transparent !important;
    padding: 12px 22px 14px !important;
    transition: color 0.2s !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #e8edf2 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #63dc8c !important;
    border-bottom-color: #63dc8c !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background: #63dc8c !important;
    height: 2px !important;
}

/* ── Section heading ── */
.section-head {
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin: 28px 0 20px;
}
.section-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
    color: #e8edf2 !important;
}
.section-desc {
    font-size: 10px !important;
    letter-spacing: 0.08em;
    color: #5a6880 !important;
}

/* ── Slider ── */
[data-testid="stSlider"] label {
    font-size: 10px !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #5a6880 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #63dc8c !important;
    border-color: #63dc8c !important;
    box-shadow: 0 0 10px rgba(99,220,140,0.4) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[class*="Track"] {
    background: #141b22 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[class*="Track"]:first-of-type {
    background: #63dc8c !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    background: #63dc8c !important;
    color: #080c10 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 28px !important;
    transition: all 0.2s !important;
    box-shadow: none !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(99,220,140,0.28) !important;
    background: #7deaa0 !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #0e1318 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    padding: 18px 20px !important;
    text-align: center !important;
    transition: border-color 0.2s, transform 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(99,220,140,0.2) !important;
    transform: translateY(-2px);
}
[data-testid="stMetric"] label {
    font-size: 9px !important;
    font-weight: 500 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #5a6880 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 26px !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    color: #63dc8c !important;
}
[data-testid="stMetricDelta"] {
    font-size: 11px !important;
    color: #5a6880 !important;
}

/* ── Dataframe / table ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] table {
    background: #0e1318 !important;
}
[data-testid="stDataFrame"] th {
    background: #141b22 !important;
    font-size: 9px !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #5a6880 !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stDataFrame"] td {
    font-size: 12px !important;
    color: #e8edf2 !important;
    border-bottom: 1px solid rgba(255,255,255,0.03) !important;
}
[data-testid="stDataFrame"] tr:hover td {
    background: rgba(255,255,255,0.02) !important;
}

/* ── Code block ── */
[data-testid="stCode"] {
    background: #060a0e !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
}
[data-testid="stCode"] code {
    font-size: 11.5px !important;
    color: #8da4be !important;
    line-height: 1.7 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #5a6880 !important;
    font-size: 12px !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(255,255,255,0.06) !important;
    margin: 24px 0 !important;
}

/* ── Card wrapper ── */
.nexus-card {
    background: #0e1318;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.nexus-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,220,140,0.18), transparent);
}

/* ── Feature bar ── */
.feat-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}
.feat-name {
    font-size: 11px;
    color: #5a6880;
    width: 200px;
    flex-shrink: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.feat-track {
    flex: 1;
    height: 4px;
    background: #141b22;
    border-radius: 2px;
    overflow: hidden;
}
.feat-fill {
    height: 100%;
    border-radius: 2px;
}
.feat-pct {
    font-size: 11px;
    font-weight: 500;
    width: 42px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Calibration gauge ── */
.cal-gauge-wrap {
    display: flex;
    justify-content: center;
    margin: 8px 0;
}

/* ── Columns gap fix ── */
[data-testid="column"] { padding: 0 8px !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child { padding-right: 0 !important; }
</style>
"""


def _header():
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown("""
    <div class="nexus-header">
      <div>
        <div class="nexus-brand">⚽ <span>Football AI</span> Nexus Engine</div>
        <div class="nexus-title">Model Analysis</div>
      </div>
      <div class="nexus-badge">● Live Model</div>
    </div>
    """, unsafe_allow_html=True)


def _section(title: str, desc: str = ""):
    desc_html = f'<span class="section-desc">{desc}</span>' if desc else ""
    st.markdown(f"""
    <div class="section-head">
      <span class="section-title">{title}</span>
      {desc_html}
    </div>
    """, unsafe_allow_html=True)


def _feat_bar(name: str, val: float, max_val: float, color: str = "#63dc8c"):
    pct = int(val / max_val * 100)
    st.markdown(f"""
    <div class="feat-row">
      <div class="feat-name">{name}</div>
      <div class="feat-track">
        <div class="feat-fill" style="width:{pct}%;background:{color}"></div>
      </div>
      <div class="feat-pct" style="color:{color}">{val*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)


def _gauge_html(predicted: float, actual: float) -> str:
    """SVG gauge comparing predicted vs actual draw rate."""
    pred_pct = int(predicted * 100)
    act_pct  = int(actual  * 100)
    pred_deg = predicted * 180 * 0.85   # map 0-1 → 0-153°
    act_deg  = actual    * 180 * 0.85

    def arc(r, deg, color, sw=10):
        import math
        start_x = 100 + r * math.cos(math.radians(180))
        start_y = 100 + r * math.sin(math.radians(180))
        end_x   = 100 + r * math.cos(math.radians(180 - deg))
        end_y   = 100 + r * math.sin(math.radians(180 - deg))
        laf = 1 if deg > 180 else 0
        return (f'<path d="M {start_x:.1f} {start_y:.1f} '
                f'A {r} {r} 0 {laf} 1 {end_x:.1f} {end_y:.1f}" '
                f'fill="none" stroke="{color}" stroke-width="{sw}" '
                f'stroke-linecap="round"/>')

    return f"""
    <svg viewBox="0 20 200 120" width="220" style="display:block;margin:0 auto">
      <!-- Tracks -->
      <path d="M 28 100 A 72 72 0 0 1 172 100" fill="none" stroke="#141b22" stroke-width="10" stroke-linecap="round"/>
      <path d="M 38 100 A 62 62 0 0 1 162 100" fill="none" stroke="#141b22" stroke-width="8"  stroke-linecap="round"/>
      <!-- Arcs -->
      {arc(72, pred_deg, '#63dc8c', 10)}
      {arc(62, act_deg,  '#3be0c4',  8)}
      <!-- Labels -->
      <text x="100" y="92" text-anchor="middle" font-family="DM Mono,monospace" font-size="14" font-weight="500" fill="#63dc8c">{pred_pct}%</text>
      <text x="100" y="108" text-anchor="middle" font-family="DM Mono,monospace" font-size="12" fill="#3be0c4">{act_pct}%</text>
      <!-- Legend -->
      <rect x="28" y="118" width="8" height="6" fill="#63dc8c" rx="1"/>
      <text x="40" y="124" font-family="DM Mono,monospace" font-size="8" fill="#5a6880">Predicted</text>
      <rect x="112" y="118" width="8" height="6" fill="#3be0c4" rx="1"/>
      <text x="124" y="124" font-family="DM Mono,monospace" font-size="8" fill="#5a6880">Actual</text>
    </svg>
    """


# ── Tabs ─────────────────────────────────────────────────────────────────────

def _tab_monte_carlo(ctx):
    _section("Monte Carlo Simulation", "Stochastic season path projection")

    n_sim = st.slider("Number of Simulations", 100, 2000, 500, 100)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("▶  Run Monte Carlo", key="btn_mc"):
        with st.spinner(f"Simulating {n_sim:,} paths..."):
            mc = silent(run_monte_carlo, ctx, n_simulations=n_sim)

        if mc:
            st.markdown("---")
            df = (
                pd.DataFrame(mc).T
                .sort_values("expected_pts", ascending=False)
                .reset_index()
                .rename(columns={"index": "Team"})
            )
            # Rename columns for display
            rename = {
                "expected_pts":   "Exp Pts",
                "expected_wins":  "Exp W",
                "expected_draws": "Exp D",
                "expected_losses":"Exp L",
                "title_pct":      "Title %",
                "top4_pct":       "Top 4 %",
                "relegation_pct": "Rel %",
            }
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:40px;color:#3a4555;
                        border:1px dashed rgba(255,255,255,0.06);border-radius:10px;font-size:12px">
              <div style="font-size:28px;margin-bottom:10px;opacity:.4">🎲</div>
              No simulation data returned — check ctx.matches
            </div>""", unsafe_allow_html=True)


def _tab_roi_backtest(ctx):
    _section("Betting ROI Backtest", "Historical simulation with Kelly staking")

    c1, c2 = st.columns(2)
    with c1:
        me = st.slider("Minimum Edge (%)", 1, 10, 3) / 100
    with c2:
        kf = st.slider("Kelly Fraction (%)", 5, 30, 15) / 100

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("▶  Run Backtest", key="btn_roi"):
        with st.spinner("Running historical backtest..."):
            roi = silent(backtest_roi, ctx, min_edge=me, kelly_fraction=kf)

        if roi:
            st.markdown("---")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("ROI",          f"{roi['roi']:+.1f}%")
            r2.metric("Win Rate",     f"{roi['win_rate']:.1f}%")
            r3.metric("Net P&L",      f"£{roi['net_pnl']:+,.0f}")
            r4.metric("Max Drawdown", f"{roi['max_dd']:.1f}%", delta_color="inverse")
        else:
            st.warning("No backtest data returned — check ctx.matches & odds data.", icon="⚠️")


def _tab_calibration(ctx):
    _section("Draw Probability Calibration", "Predicted vs actual draw rates")

    if st.button("▶  Analyze Calibration", key="btn_cal"):
        with st.spinner("Analyzing draw rates..."):
            cal = silent(analyze_draw_calibration, ctx)

        if cal:
            st.markdown("---")
            col_left, col_right = st.columns([1.4, 1])

            with col_left:
                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted Draw Rate", f"{cal['predicted_rate']:.1%}")
                m2.metric("Actual Draw Rate",    f"{cal['actual_rate']:.1%}")
                bias = cal['bias']
                m3.metric("Bias",                f"{bias:+.1f}%", delta_color="inverse")

                st.markdown("<br>", unsafe_allow_html=True)

                # Bias bar
                bias_pct = min(abs(bias) / 10 * 50, 50)
                color    = "#e05c7a" if bias < 0 else "#63dc8c"
                side     = "right:50%" if bias < 0 else "left:50%"
                label    = "Model overestimates draws" if bias > 0 else "Model underestimates draws"
                st.markdown(f"""
                <div style="font-size:9px;letter-spacing:.12em;text-transform:uppercase;
                            color:#5a6880;margin-bottom:8px">{label}</div>
                <div style="height:6px;background:#141b22;border-radius:3px;position:relative;margin-bottom:6px">
                  <div style="position:absolute;top:0;bottom:0;{side};width:{bias_pct}%;
                              background:{color};border-radius:3px"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:9px;
                            color:#3a4555;letter-spacing:.1em;text-transform:uppercase">
                  <span>−10%</span><span>0</span><span>+10%</span>
                </div>
                """, unsafe_allow_html=True)

            with col_right:
                st.markdown('<div class="cal-gauge-wrap">', unsafe_allow_html=True)
                st.markdown(
                    _gauge_html(cal["predicted_rate"], cal["actual_rate"]),
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No calibration data returned.", icon="⚠️")


def _tab_feature_importance(ctx):
    _section("Feature Importance", "Top 20 — tree-based weight analysis")

    if st.button("▶  Show Importance", key="btn_fi"):
        with st.spinner("Calculating tree weights..."):
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_feature_importance(ctx, max_display=20)
            raw = buf.getvalue()

        if raw.strip():
            st.markdown("---")

            # Try to parse lines like: "feature_name    0.142"
            rows = []
            for line in raw.strip().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        rows.append((parts[0], float(parts[-1])))
                    except ValueError:
                        pass

            if rows:
                max_val = max(v for _, v in rows)
                tier_colors = ["#63dc8c", "#63dc8c", "#3be0c4", "#3be0c4",
                               "#3be0c4", "#f5c842", "#f5c842", "#f5c842",
                               "#5a6880", "#5a6880"]

                for i, (name, val) in enumerate(rows[:20]):
                    color = tier_colors[min(i, len(tier_colors)-1)]
                    _feat_bar(name, val, max_val, color)
            else:
                # Fallback: show raw code output
                st.code(raw, language="text")
        else:
            st.warning("Feature importance returned no output.", icon="⚠️")


# ── Entry point ──────────────────────────────────────────────────────────────

def page_analysis(ctx):
    _header()

    t1, t2, t3, t4 = st.tabs([
        "🎲  Monte Carlo",
        "💰  ROI Backtest",
        "⚖️  Calibration",
        "🧠  Feature Importance",
    ])

    with t1: _tab_monte_carlo(ctx)
    with t2: _tab_roi_backtest(ctx)
    with t3: _tab_calibration(ctx)
    with t4: _tab_feature_importance(ctx)