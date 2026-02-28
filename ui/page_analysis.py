"""
ui/page_analysis.py — Model Analysis & Insights
Style matched to page_season.py  (football-data.org crests, same HTML table design)
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from src.analysis import run_monte_carlo, analyze_draw_calibration, run_feature_importance
from utils import silent


# ═══════════════════════════════════════════════════════════════
# CRESTS  (football-data.org CDN — same as page_season.py)
# ═══════════════════════════════════════════════════════════════
_CR = "https://crests.football-data.org/{}.png"
_IDS = {
    # ── Premier League current + recent ──────────────────────────
    "Arsenal":57, "Aston Villa":58, "Bournemouth":1044, "Brentford":402,
    "Brighton":397, "Brighton & Hove Albion":397, "Chelsea":61,
    "Crystal Palace":354, "Everton":62, "Fulham":63,
    "Ipswich":57218, "Ipswich Town":57218,
    "Leicester":338, "Leicester City":338, "Liverpool":64,
    "Man City":65, "Manchester City":65,
    "Man United":66, "Manchester United":66,
    "Newcastle":67, "Newcastle United":67,
    "Nottm Forest":351, "Nottingham Forest":351,
    "Nott'm Forest":351, "Nott'm Forest":351, "Nott Forest":351,
    "Nottm. Forest":351, "Nott. Forest":351,
    "Southampton":340,
    "Spurs":73, "Tottenham":73, "Tottenham Hotspur":73,
    "West Ham":563, "West Ham United":563,
    "Wolves":76, "Wolverhampton":76, "Wolverhampton Wanderers":76,
    # ── Championship / recently promoted ─────────────────────────
    "Leeds":341, "Leeds United":341,
    "Burnley":328,
    "Sheffield United":356, "Sheffield Utd":356,
    "Luton":389, "Luton Town":389,
    "Sunderland":71,
    "Norwich":68, "Norwich City":68,
    "Watford":346,
    "West Brom":74, "West Bromwich Albion":74,
    "Middlesbrough":343,
    "Hull":322, "Hull City":322,
    "Derby":334, "Derby County":334,
    "Blackburn":59, "Blackburn Rovers":59,
    "Coventry":333, "Coventry City":333,
    "Plymouth":1295, "Plymouth Argyle":1295,
    "Oxford":1333, "Oxford United":1333,
    "Cardiff":715, "Cardiff City":715,
    "Swansea":72, "Swansea City":72,
    "Stoke":70, "Stoke City":70,
    "QPR":69, "Queens Park Rangers":69,
    "Millwall":1062,
    "Preston":1081, "Preston North End":1081,
    "Bristol City":387,
    "Blackpool":386,
    "Huddersfield":394,
    "Wigan":75, "Wigan Athletic":75,
    "Rotherham":1427, "Rotherham United":1427,
}


def _crest(name: str) -> str:
    t = _IDS.get(name)
    if t:
        return _CR.format(t)
    nl = name.lower()
    for k, v in _IDS.items():
        if k.lower() in nl or nl in k.lower():
            return _CR.format(v)
    return ""


def _img_tag(name: str, sz: int = 28) -> str:
    url = _crest(name)
    if url:
        return (
            f'<img src="{url}" width="{sz}" height="{sz}" '
            f'style="object-fit:contain;vertical-align:middle;'
            f'border-radius:4px;flex-shrink:0;" '
            f'onerror="this.style.display=\'none\'">'
        )
    return (
        f'<span style="width:{sz}px;height:{sz}px;display:inline-block;'
        f'flex-shrink:0;"></span>'
    )


# ═══════════════════════════════════════════════════════════════
# ZONE COLOURS  (identical to page_season.py)
# ═══════════════════════════════════════════════════════════════
_ZM = {
    "ch":  {"c":"#FFD700","g":"rgba(255,215,0,.45)",   "bg":"rgba(255,215,0,.09)",   "b":"rgba(255,215,0,.45)",   "l":"Champion"},
    "ucl": {"c":"#38BDF8","g":"rgba(56,189,248,.4)",   "bg":"rgba(56,189,248,.08)",  "b":"rgba(56,189,248,.4)",   "l":"UCL"},
    "eu":  {"c":"#34D399","g":"rgba(52,211,153,.35)",  "bg":"rgba(52,211,153,.08)",  "b":"rgba(52,211,153,.35)",  "l":"Europa"},
    "co":  {"c":"#C084FC","g":"rgba(192,132,252,.35)", "bg":"rgba(192,132,252,.08)", "b":"rgba(192,132,252,.35)", "l":"Conference"},
    "re":  {"c":"#F87171","g":"rgba(248,113,113,.4)",  "bg":"rgba(248,113,113,.09)", "b":"rgba(248,113,113,.4)",  "l":"Relegation"},
    "sa":  {"c":"rgba(148,187,233,.35)","g":"transparent","bg":"transparent","b":"transparent","l":""},
}


def _zone_mc(title_pct: float, top4_pct: float, rel_pct: float) -> dict:
    """Map Monte Carlo probabilities → zone colour dict."""
    if title_pct >= 30:  return _ZM["ch"]
    if top4_pct  >= 50:  return _ZM["ucl"]
    if top4_pct  >= 20:  return _ZM["eu"]
    if rel_pct   >= 30:  return _ZM["re"]
    return _ZM["sa"]


# ═══════════════════════════════════════════════════════════════
# GOOGLE FONTS  (inside iframe HTML)
# ═══════════════════════════════════════════════════════════════
_GF = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Syne:wght@700;800&"
    "family=DM+Sans:wght@400;500;600;700&"
    "family=JetBrains+Mono:wght@600;700&display=swap');"
)


# ═══════════════════════════════════════════════════════════════
# PAGE CSS  (same design tokens as page_season.py)
# ═══════════════════════════════════════════════════════════════
_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@600;700&display=swap');

.main .block-container { padding-top:1.5rem; max-width:1200px; }

.pg-eyebrow {
    font-family:'DM Sans',sans-serif; font-size:.8rem; font-weight:700;
    letter-spacing:5px; text-transform:uppercase; color:#38BDF8; margin-bottom:10px;
}
.pg-title {
    font-family:'Syne',sans-serif; font-size:3.2rem; font-weight:800;
    color:#F0F6FF; letter-spacing:-.5px; line-height:1.05; margin-bottom:6px;
}
.pg-title em { font-style:normal; color:#38BDF8; }
.pg-sub {
    font-family:'DM Sans',sans-serif; font-size:1.05rem;
    color:rgba(148,187,233,.44); line-height:1.6;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap:0 !important; background:transparent !important;
    border-bottom:1px solid rgba(255,255,255,.08) !important;
}
.stTabs [data-baseweb="tab"] {
    font-family:'DM Sans',sans-serif !important; font-size:.85rem !important;
    font-weight:700 !important; letter-spacing:2.5px !important;
    text-transform:uppercase !important; color:rgba(148,187,233,.38) !important;
    background:transparent !important; border:none !important;
    border-radius:0 !important; padding:14px 28px !important;
    border-bottom:3px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color:#F0F6FF !important;
    border-bottom:3px solid #38BDF8 !important;
}

/* Buttons */
.stButton > button {
    font-family:'DM Sans',sans-serif !important; font-weight:700 !important;
    letter-spacing:2.5px !important; font-size:.85rem !important;
    text-transform:uppercase !important; border-radius:10px !important;
    padding:13px 22px !important; transition:all .2s !important;
}
[data-testid="baseButton-primary"] {
    background:linear-gradient(135deg,#0d5a74,#38BDF8) !important;
    border:none !important; color:#fff !important;
    box-shadow:0 4px 22px rgba(56,189,248,.22) !important;
}
[data-testid="baseButton-primary"]:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 10px 36px rgba(56,189,248,.38) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background:rgba(255,255,255,.03) !important;
    border:1px solid rgba(255,255,255,.08) !important;
    border-radius:14px !important; padding:22px 24px !important;
}
[data-testid="stMetricLabel"] {
    font-family:'DM Sans',sans-serif !important; font-size:.72rem !important;
    letter-spacing:2.5px !important; text-transform:uppercase !important;
    color:rgba(148,187,233,.36) !important;
}
[data-testid="stMetricValue"] {
    font-family:'Syne',sans-serif !important;
    font-size:1.3rem !important; color:#F0F6FF !important;
}

/* Legend */
.lg-wrap {
    display:flex; flex-wrap:wrap; gap:10px;
    margin-top:18px; padding-top:16px;
    border-top:1px solid rgba(255,255,255,.06);
}
.lg-pill {
    font-family:'DM Sans',sans-serif; font-size:.7rem; font-weight:700;
    letter-spacing:1.8px; text-transform:uppercase;
    padding:5px 16px; border-radius:99px; border:1px solid; white-space:nowrap;
}
.lg-dot {
    display:inline-block; width:8px; height:8px;
    border-radius:50%; margin-right:7px; vertical-align:middle;
}

/* Slider */
.stSlider label {
    font-family:'DM Sans',sans-serif !important; font-size:.75rem !important;
    letter-spacing:2px !important; text-transform:uppercase !important;
    color:rgba(148,187,233,.38) !important; font-weight:700 !important;
}

/* Spinner */
.stSpinner > div { border-top-color:#38BDF8 !important; }
</style>"""


# ═══════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════
_LGND = [
    ("#FFD700", "Champion"),
    ("#38BDF8", "UCL Top 4"),
    ("#34D399", "Europa"),
    ("#C084FC", "Conference"),
    ("#F87171", "Relegation"),
]


def _legend():
    pills = "".join(
        f'<span class="lg-pill" style="color:{c};border-color:{c}33;background:{c}0D;">'
        f'<span class="lg-dot" style="background:{c};"></span>{l}</span>'
        for c, l in _LGND
    )
    st.markdown(f'<div class="lg-wrap">{pills}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HTML TABLE — Monte Carlo results
# ═══════════════════════════════════════════════════════════════
_ROW_H  = 56
_HEAD_H = 50
_WRAP_H = 4


def _pct_bar(val: float, color: str, max_val: float = 100.0) -> str:
    """Mini inline bar + number — visual representation of a percentage."""
    w = max(3, int(val / max_val * 80))   # bar width in px, max 80px
    return (
        f'<span style="display:inline-flex;align-items:center;gap:7px;">'
        f'<span style="display:inline-block;width:{w}px;height:5px;border-radius:99px;'
        f'background:{color};opacity:.8;flex-shrink:0;"></span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:.9rem;'
        f'font-weight:700;color:{color};">{val:.1f}%</span>'
        f'</span>'
    )


def _build_mc_table(mc: dict) -> tuple[str, int]:
    """
    Columns: # | Club | Exp PTS | Title % | Top 4 % | Rel % | Zone
    Uses only data that run_monte_carlo actually returns.
    """
    rows_sorted = sorted(
        mc.items(),
        key=lambda x: x[1].get("expected_pts", 0),
        reverse=True,
    )
    n = len(rows_sorted)

    # ── Find max values for bar scaling ──────────────────────────
    all_title = [v.get("title_pct", 0) * 100 for _, v in rows_sorted]
    all_top4  = [v.get("top4_pct",  0) * 100 for _, v in rows_sorted]
    all_rel   = [v.get("relegation_pct", 0) * 100 for _, v in rows_sorted]
    max_title = max(all_title) or 1
    max_top4  = max(all_top4)  or 1
    max_rel   = max(all_rel)   or 1

    TH = (
        "font-family:'DM Sans',sans-serif;font-size:.75rem;font-weight:700;"
        "letter-spacing:2.5px;text-transform:uppercase;color:rgba(148,187,233,.32);"
        f"padding:0 18px;height:{_HEAD_H}px;white-space:nowrap;"
        "border-bottom:1px solid rgba(255,255,255,.07);"
    )
    TH_C = TH + "text-align:center;"
    TH_L = TH + "text-align:left;"
    TH_R = TH + "text-align:right;"

    thead = (
        f'<thead><tr>'
        f'<th style="{TH_C}">#</th>'
        f'<th style="{TH_L}">Club</th>'
        f'<th style="{TH_C}">Exp&nbsp;PTS</th>'
        f'<th style="{TH_R}">Title&nbsp;%</th>'
        f'<th style="{TH_R}">Top&nbsp;4&nbsp;%</th>'
        f'<th style="{TH_R}">Rel&nbsp;%</th>'
        f'<th style="{TH_C}">Zone</th>'
        f'</tr></thead>'
    )

    TD  = f"padding:0 18px;text-align:center;height:{_ROW_H}px;font-family:'DM Sans',sans-serif;"
    TDL = TD.replace("text-align:center", "text-align:left")
    TDR = TD.replace("text-align:center", "text-align:right")

    body_parts = []
    for pos, (team, stats) in enumerate(rows_sorted, start=1):
        exp_pts  = float(stats.get("expected_pts",   0))
        t_pct    = float(stats.get("title_pct",      0)) * 100
        top4_pct = float(stats.get("top4_pct",       0)) * 100
        rel_pct  = float(stats.get("relegation_pct", 0)) * 100

        z   = _zone_mc(t_pct, top4_pct, rel_pct)
        c   = z["c"]; g = z["g"]; bg = z["bg"]; b = z["b"]; lbl = z["l"]

        left_border = (
            f"border-left:4px solid {c};"
            if lbl else
            "border-left:4px solid rgba(255,255,255,.04);"
        )
        row_bg = (
            bg if lbl else
            ("rgba(255,255,255,.02)" if pos % 2 == 0 else "transparent")
        )
        pos_style = (
            f"font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:700;color:{c};"
            if lbl else
            "font-family:'JetBrains Mono',monospace;font-size:.95rem;font-weight:600;"
            "color:rgba(148,187,233,.3);"
        )

        # ── Club ─────────────────────────────────────────────────
        club_td = (
            f'<td style="{TDL}">'
            f'<span style="display:inline-flex;align-items:center;gap:14px;">'
            f'{_img_tag(team, 28)}'
            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:1.1rem;font-weight:700;'
            f'color:#F0F6FF;white-space:nowrap;">{team}</span>'
            f'</span></td>'
        )

        # ── Expected points (glowing) ─────────────────────────────
        pts_td = (
            f'<td style="{TD}">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:1.25rem;'
            f'font-weight:700;color:#F0F6FF;text-shadow:0 0 20px {g};">'
            f'{exp_pts:.1f}</span></td>'
        )

        # ── % columns with bars ───────────────────────────────────
        title_td = f'<td style="{TDR}">{_pct_bar(t_pct,    "#FFD700", max_title)}</td>'
        top4_td  = f'<td style="{TDR}">{_pct_bar(top4_pct, "#38BDF8", max_top4 )}</td>'
        rel_td   = f'<td style="{TDR}">{_pct_bar(rel_pct,  "#F87171", max_rel  )}</td>'

        # ── Zone badge ────────────────────────────────────────────
        zone_badge = (
            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:.68rem;font-weight:800;'
            f'letter-spacing:1.5px;text-transform:uppercase;color:{c};background:{bg};'
            f'border:1px solid {b};border-radius:6px;padding:4px 10px;">{lbl}</span>'
            if lbl else ""
        )
        zone_td = f'<td style="{TD}">{zone_badge}</td>'

        body_parts.append(
            f'<tr style="border-bottom:1px solid rgba(255,255,255,.05);'
            f'background:{row_bg};{left_border}transition:background .12s;" '
            f'onmouseover="this.style.background=\'rgba(56,189,248,.06)\'" '
            f'onmouseout="this.style.background=\'{row_bg}\'">'
            f'<td style="{TD}"><span style="{pos_style}">{pos}</span></td>'
            f'{club_td}{pts_td}{title_td}{top4_td}{rel_td}{zone_td}'
            f'</tr>'
        )

    tbody   = f'<tbody>{"".join(body_parts)}</tbody>'
    exact_h = _HEAD_H + n * _ROW_H + _WRAP_H

    html = (
        f'<html><head>'
        f'<style>{_GF}'
        f'*{{margin:0;padding:0;box-sizing:border-box;}}'
        f'html,body{{background:#060F1C;overflow:hidden;width:100%;}}'
        f'table{{width:100%;border-collapse:collapse;table-layout:auto;}}'
        f'</style></head><body>'
        f'<div style="background:rgba(255,255,255,.025);'
        f'border:1px solid rgba(255,255,255,.08);border-radius:14px;overflow:hidden;">'
        f'<table>{thead}{tbody}</table>'
        f'</div>'
        f'</body></html>'
    )
    return html, exact_h


# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════

def _tab_monte_carlo(ctx):
    st.write("")
    st.info(
        "สุ่มจำลองผลการแข่งขันที่เหลือของฤดูกาลหลายรอบ "
        "เพื่อประเมิน Expected Points, Title %, Top 4 % และ Relegation %"
    )

    n_sim = st.slider("Number of Simulations", 100, 2000, 500, 100)

    if st.button("🔮  Run Monte Carlo", type="primary", key="btn_mc"):
        with st.spinner("Simulating season paths…"):
            mc = silent(run_monte_carlo, ctx, n_simulations=n_sim)

        if mc:
            html, h = _build_mc_table(mc)
            components.html(html, height=h, scrolling=False)
            _legend()
        else:
            st.warning("No simulation data returned.")


def _tab_calibration(ctx):
    st.write("")
    st.info(
        "ตรวจสอบว่าเปอร์เซ็นต์ Draw ที่โมเดลทำนาย "
        "สอดคล้องกับผลการแข่งขันจริงหรือไม่"
    )

    if st.button("⚖️  Analyze Calibration", type="primary", key="btn_cal"):
        with st.spinner("Evaluating draw probabilities…"):
            cal = silent(analyze_draw_calibration, ctx)

        if cal:
            col1, col2, col3 = st.columns(3, gap="medium")
            col1.metric("Predicted Draw Rate", f"{cal['predicted_rate']:.1%}")
            col2.metric("Actual Draw Rate",    f"{cal['actual_rate']:.1%}")
            col3.metric("Bias",                f"{cal['bias']:+.1f}%")
            st.markdown("---")
            if abs(cal["bias"]) < 2:
                st.success("✅ Model draw probability is well calibrated.")
            elif cal["bias"] > 0:
                st.warning("⚠️ Model tends to overestimate draw probability.")
            else:
                st.warning("⚠️ Model tends to underestimate draw probability.")
        else:
            st.warning("No calibration data returned.")


def _tab_feature_importance(ctx):
    st.write("")
    st.info("แสดง 20 Features ที่มีผลต่อการตัดสินใจของโมเดลมากที่สุด")

    if st.button("🧠  Show Feature Importance", type="primary", key="btn_fi"):
        with st.spinner("Calculating feature importance…"):
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_feature_importance(ctx, max_display=20)
            raw = buf.getvalue()

        if raw.strip():
            rows = []
            for line in raw.strip().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        rows.append((parts[0], float(parts[-1])))
                    except Exception:
                        pass
            if rows:
                df = pd.DataFrame(rows, columns=["Feature", "Importance"])
                df = df.sort_values("Importance", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.code(raw)
        else:
            st.warning("Feature importance returned no output.")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════
def page_analysis(ctx):
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown("""
        <div style="margin-bottom:32px;">
            <div class="pg-eyebrow">⚡ Nexus Engine · Premier League</div>
            <div class="pg-title">Model <em>Analysis</em></div>
            <div class="pg-sub">Live Model Evaluation · Season Simulation · Probability Calibration</div>
        </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "  🎲 Monte Carlo  ",
        "  ⚖️ Calibration  ",
        "  🧠 Feature Importance  ",
    ])

    with tab1:
        _tab_monte_carlo(ctx)

    with tab2:
        _tab_calibration(ctx)

    with tab3:
        _tab_feature_importance(ctx)