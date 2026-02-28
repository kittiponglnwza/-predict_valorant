"""
ui/page_season.py — Season Table · Football AI Nexus Engine v9
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import glob as _glob

from src.config import TODAY, DATA_DIR
from src.predict import (
    run_season_simulation,
    update_season_csv_from_api,
    get_pl_standings_from_api,
)
from utils import silent, zone_label

# ═══════════════════════════════════════════════════════════════
# CRESTS  (football-data.org CDN — reliable, always available)
# ═══════════════════════════════════════════════════════════════
_CR = "https://crests.football-data.org/{}.png"
_IDS = {
    "Arsenal":57,"Aston Villa":58,"Bournemouth":1044,"Brentford":402,
    "Brighton":397,"Brighton & Hove Albion":397,"Chelsea":61,
    "Crystal Palace":354,"Everton":62,"Fulham":63,
    "Ipswich":57218,"Ipswich Town":57218,
    "Leicester":338,"Leicester City":338,"Liverpool":64,
    "Man City":65,"Manchester City":65,
    "Man United":66,"Manchester United":66,
    "Newcastle":67,"Newcastle United":67,
    "Nottm Forest":351,"Nottingham Forest":351,
    "Southampton":340,"Spurs":73,"Tottenham":73,"Tottenham Hotspur":73,
    "West Ham":563,"West Ham United":563,
    "Wolves":76,"Wolverhampton":76,"Wolverhampton Wanderers":76,
    "Leeds":341,"Leeds United":341,"Burnley":328,
    "Sheffield United":356,"Luton":389,"Luton Town":389,
    "Sunderland":71,"Norwich":68,"Norwich City":68,"Watford":346,
    "West Brom":74,"West Bromwich Albion":74,"Middlesbrough":343,
    "Hull":322,"Hull City":322,"Derby":334,"Derby County":334,
    "Blackburn":59,"Blackburn Rovers":59,"Coventry":333,"Coventry City":333,
    "Plymouth":1295,"Plymouth Argyle":1295,"Oxford":1333,"Oxford United":1333,
}

def _crest(name: str) -> str:
    t = _IDS.get(name)
    if t: return _CR.format(t)
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
    return f'<span style="width:{sz}px;height:{sz}px;display:inline-block;flex-shrink:0;"></span>'

# ═══════════════════════════════════════════════════════════════
# ZONE COLOURS
# ═══════════════════════════════════════════════════════════════
_ZM = {
    "ch":  {"c":"#FFD700","g":"rgba(255,215,0,.45)",  "bg":"rgba(255,215,0,.09)",  "b":"rgba(255,215,0,.45)",  "l":"Champion"},
    "ucl": {"c":"#38BDF8","g":"rgba(56,189,248,.4)",  "bg":"rgba(56,189,248,.08)", "b":"rgba(56,189,248,.4)",  "l":"UCL"},
    "eu":  {"c":"#34D399","g":"rgba(52,211,153,.35)", "bg":"rgba(52,211,153,.08)", "b":"rgba(52,211,153,.35)", "l":"Europa"},
    "co":  {"c":"#C084FC","g":"rgba(192,132,252,.35)","bg":"rgba(192,132,252,.08)","b":"rgba(192,132,252,.35)","l":"Conference"},
    "re":  {"c":"#F87171","g":"rgba(248,113,113,.4)", "bg":"rgba(248,113,113,.09)","b":"rgba(248,113,113,.4)", "l":"Relegation"},
    "sa":  {"c":"rgba(148,187,233,.35)","g":"transparent","bg":"transparent","b":"transparent","l":""},
}

def _zone(pos: int, n: int = 20) -> dict:
    if pos == 1:    return _ZM["ch"]
    if pos <= 4:    return _ZM["ucl"]
    if pos <= 6:    return _ZM["eu"]
    if pos == 7:    return _ZM["co"]
    if pos > n - 3: return _ZM["re"]
    return _ZM["sa"]

# ═══════════════════════════════════════════════════════════════
# FORM BADGES
# ═══════════════════════════════════════════════════════════════
def _form(s: str) -> str:
    if not s: return ""
    M = {"W":"#22C55E","D":"#F59E0B","L":"#EF4444"}
    out = []
    for c in str(s).replace(" ","").split(","):
        c = c.strip().upper()
        if c in M:
            col = M[c]
            out.append(
                f'<span style="display:inline-flex;align-items:center;justify-content:center;'
                f'width:24px;height:24px;border-radius:6px;background:{col}1A;'
                f'border:1.5px solid {col}55;font-size:.8rem;font-weight:900;color:{col};">{c}</span>'
            )
    return '<span style="display:inline-flex;gap:4px;">' + "".join(out) + '</span>'

# Google Fonts import string (used inside each HTML iframe)
_GF = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Syne:wght@700;800&"
    "family=DM+Sans:wght@400;500;600;700&"
    "family=JetBrains+Mono:wght@600;700&display=swap');"
)

# ═══════════════════════════════════════════════════════════════
# LOAD TEAM STATS FROM CSV  (W/D/L/GD — not in final_table)
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=60, show_spinner=False)
def _load_stats() -> dict:
    """Compute per-team played/W/D/L/GF/GA from season CSV files."""
    stats: dict = {}
    try:
        csvs = [f for f in _glob.glob(os.path.join(DATA_DIR, "*.csv"))
                if "backup" not in f.lower()]
        s25  = [f for f in csvs if "2025" in os.path.basename(f)]
        use  = s25 if s25 else csvs
        if not use:
            return stats
        raw = pd.concat([pd.read_csv(f) for f in use], ignore_index=True)
        raw = raw.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
        raw["FTHG"] = pd.to_numeric(raw["FTHG"], errors="coerce")
        raw["FTAG"] = pd.to_numeric(raw["FTAG"], errors="coerce")
        played = raw.dropna(subset=["FTHG", "FTAG"])
        for _, m in played.iterrows():
            ht, at = str(m["HomeTeam"]), str(m["AwayTeam"])
            hg, ag = int(m["FTHG"]), int(m["FTAG"])
            for t in (ht, at):
                if t not in stats:
                    stats[t] = {"mp": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0}
            stats[ht]["mp"] += 1;  stats[at]["mp"] += 1
            stats[ht]["gf"] += hg; stats[ht]["ga"] += ag
            stats[at]["gf"] += ag; stats[at]["ga"] += hg
            if   hg > ag: stats[ht]["w"] += 1; stats[at]["l"] += 1
            elif hg < ag: stats[at]["w"] += 1; stats[ht]["l"] += 1
            else:         stats[ht]["d"] += 1; stats[at]["d"] += 1
    except Exception:
        pass
    return stats

# ═══════════════════════════════════════════════════════════════
# TABLE HTML BUILDER
# key fix: height = header_px + n_rows * row_px  (no scrolling)
# ═══════════════════════════════════════════════════════════════
_ROW_H   = 56   # px per data row
_HEAD_H  = 50   # px for thead
_WRAP_H  = 4    # extra wrapper padding

def _table_html(rows: list, sim: bool = False) -> tuple[str, int]:
    """
    Build a full self-contained HTML table.
    Returns (html_string, exact_pixel_height).
    scrolling=False in components.html + exact height = NO SCROLL.
    """
    # ── thead ────────────────────────────────────────────────────
    TH = (
        "font-family:'DM Sans',sans-serif;font-size:.75rem;font-weight:700;"
        "letter-spacing:2.5px;text-transform:uppercase;color:rgba(148,187,233,.32);"
        f"padding:0 18px;height:{_HEAD_H}px;white-space:nowrap;"
        "border-bottom:1px solid rgba(255,255,255,.07);"
    )
    TH_C = TH + "text-align:center;"
    TH_L = TH + "text-align:left;"

    if sim:
        pts_h = (
            f'<th style="{TH_C}">Now</th>'
            f'<th style="{TH_C}">+Proj</th>'
            f'<th style="{TH_C}">Final</th>'
        )
    else:
        pts_h = f'<th style="{TH_C}">Pts</th>'

    thead = (
        f'<thead><tr>'
        f'<th style="{TH_C}">#</th>'
        f'<th style="{TH_L}" colspan="1">Club</th>'
        f'<th style="{TH_C}">MP</th>'
        f'<th style="{TH_C}">W</th>'
        f'<th style="{TH_C}">D</th>'
        f'<th style="{TH_C}">L</th>'
        f'<th style="{TH_C}">GD</th>'
        f'{pts_h}'
        f'<th style="{TH_L}">Form</th>'
        f'<th style="{TH_C}">Zone</th>'
        f'</tr></thead>'
    )

    # ── tbody ────────────────────────────────────────────────────
    TD  = (
        f"padding:0 18px;text-align:center;height:{_ROW_H}px;"
        "font-family:'DM Sans',sans-serif;"
    )
    TDL = TD.replace("text-align:center", "text-align:left")

    body_parts = []
    for r in rows:
        pos = r["pos"]
        z   = r["zc"]
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

        # position number
        pos_style = (
            f"font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:700;color:{c};"
            if lbl else
            "font-family:'JetBrains Mono',monospace;font-size:.95rem;font-weight:600;color:rgba(148,187,233,.3);"
        )

        # club cell — logo + name
        club_td = (
            f'<td style="{TDL}">'
            f'<span style="display:inline-flex;align-items:center;gap:14px;">'
            f'{_img_tag(r["team"], 28)}'
            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:1.1rem;font-weight:700;'
            f'color:#F0F6FF;white-space:nowrap;">{r["team"]}</span>'
            f'</span></td>'
        )

        # GD colouring
        gd  = int(r.get("gd", 0))
        gds = f"+{gd}" if gd > 0 else str(gd)
        gdc = "#34D399" if gd > 0 else ("#F87171" if gd < 0 else "rgba(148,187,233,.4)")

        stat_tds = (
            f'<td style="{TD}"><span style="font-size:1rem;color:rgba(148,187,233,.5);">{r.get("played", 0)}</span></td>'
            f'<td style="{TD}"><span style="font-size:1rem;font-weight:700;color:#34D399;">{r.get("w", 0)}</span></td>'
            f'<td style="{TD}"><span style="font-size:1rem;color:#F59E0B;">{r.get("d", 0)}</span></td>'
            f'<td style="{TD}"><span style="font-size:1rem;color:#F87171;">{r.get("l", 0)}</span></td>'
            f'<td style="{TD}"><span style="font-size:1rem;font-weight:700;color:{gdc};">{gds}</span></td>'
        )

        # points cells
        if sim:
            rp = int(r.get("rp", 0))
            pp = int(r.get("pp", 0))
            fp = int(r.get("fp", 0))
            ppc = "#34D399" if pp > 0 else ("#F87171" if pp < 0 else "rgba(148,187,233,.4)")
            pps = f"+{pp}" if pp >= 0 else str(pp)
            pts_tds = (
                f'<td style="{TD}"><span style="font-family:\'JetBrains Mono\',monospace;'
                f'font-size:1rem;color:rgba(148,187,233,.55);">{rp}</span></td>'
                f'<td style="{TD}"><span style="font-family:\'JetBrains Mono\',monospace;'
                f'font-size:.95rem;font-weight:700;color:{ppc};">{pps}</span></td>'
                f'<td style="{TD}"><span style="font-family:\'JetBrains Mono\',monospace;'
                f'font-size:1.25rem;font-weight:700;color:#F0F6FF;'
                f'text-shadow:0 0 20px {g};">{fp}</span></td>'
            )
        else:
            pts = int(r.get("pts", 0))
            pts_tds = (
                f'<td style="{TD}"><span style="font-family:\'JetBrains Mono\',monospace;'
                f'font-size:1.25rem;font-weight:700;color:#F0F6FF;'
                f'text-shadow:0 0 20px {g};">{pts}</span></td>'
            )

        form_td = f'<td style="{TDL}">{_form(r.get("form", ""))}</td>'

        if lbl:
            zone_badge = (
                f'<span style="font-family:\'DM Sans\',sans-serif;font-size:.68rem;font-weight:800;'
                f'letter-spacing:1.5px;text-transform:uppercase;color:{c};background:{bg};'
                f'border:1px solid {b};border-radius:6px;padding:4px 10px;">{lbl}</span>'
            )
        else:
            zone_badge = ""
        zone_td = f'<td style="{TD}">{zone_badge}</td>'

        body_parts.append(
            f'<tr style="border-bottom:1px solid rgba(255,255,255,.05);'
            f'background:{row_bg};{left_border}transition:background .12s;" '
            f'onmouseover="this.style.background=\'rgba(56,189,248,.06)\'" '
            f'onmouseout="this.style.background=\'{row_bg}\'">'
            f'<td style="{TD}"><span style="{pos_style}">{pos}</span></td>'
            f'{club_td}{stat_tds}{pts_tds}{form_td}{zone_td}'
            f'</tr>'
        )

    tbody = f'<tbody>{"".join(body_parts)}</tbody>'

    # ── exact pixel height (NO SCROLLBAR) ────────────────────────
    n_rows = len(rows)
    exact_h = _HEAD_H + n_rows * _ROW_H + _WRAP_H

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
# STREAMLIT PAGE CSS
# ═══════════════════════════════════════════════════════════════
_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@600;700&display=swap');

.main .block-container { padding-top:1.5rem; max-width:1200px; }

/* ── Header ── */
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

/* ── Tabs ── */
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

/* ── Selectbox ── */
.stSelectbox label {
    font-family:'DM Sans',sans-serif !important; font-size:.75rem !important;
    letter-spacing:2px !important; text-transform:uppercase !important;
    color:rgba(148,187,233,.38) !important; font-weight:700 !important;
}
.stSelectbox [data-baseweb="select"] {
    font-family:'DM Sans',sans-serif !important; font-size:1rem !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family:'DM Sans',sans-serif !important; font-weight:700 !important;
    letter-spacing:2.5px !important; font-size:.85rem !important;
    text-transform:uppercase !important; border-radius:10px !important;
    padding:13px 22px !important; transition:all .2s !important;
}
[data-testid="baseButton-primary"] {
    background:linear-gradient(135deg, #0d5a74, #38BDF8) !important;
    border:none !important; color:#fff !important;
    box-shadow:0 4px 22px rgba(56,189,248,.22) !important;
}
[data-testid="baseButton-primary"]:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 10px 36px rgba(56,189,248,.38) !important;
}

/* ── Metrics ── */
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
[data-testid="stMetricDelta"] {
    font-family:'DM Sans',sans-serif !important; font-size:.85rem !important;
}

/* ── Legend ── */
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

/* ── Section divider ── */
.div-wrap { display:flex; align-items:center; gap:16px; margin:38px 0 22px; }
.div-line  { flex:1; height:1px; background:rgba(255,255,255,.06); }
.div-label {
    font-family:'DM Sans',sans-serif; font-size:.72rem; font-weight:700;
    letter-spacing:3px; text-transform:uppercase;
    color:rgba(148,187,233,.28); white-space:nowrap;
}

/* ── Sim ready ── */
.ready-wrap {
    text-align:center; padding:6rem 2rem;
    background:linear-gradient(135deg,rgba(56,189,248,.03),rgba(192,132,252,.03));
    border:1px dashed rgba(56,189,248,.15); border-radius:16px; margin-top:1.5rem;
}
.ready-icon  { font-size:4rem; opacity:.22; margin-bottom:20px; }
.ready-title {
    font-family:'Syne',sans-serif; font-size:1.9rem; font-weight:800;
    color:#F0F6FF; margin-bottom:12px;
}
.ready-sub {
    font-family:'DM Sans',sans-serif; font-size:1.05rem;
    color:rgba(148,187,233,.36); line-height:1.75; max-width:420px; margin:0 auto;
}

/* ── Error ── */
.err-box {
    background:rgba(248,113,113,.06); border:1px solid rgba(248,113,113,.22);
    border-radius:12px; padding:18px 24px;
    font-family:'DM Sans',sans-serif; font-size:1rem; color:rgba(252,165,165,.8);
}

/* ── Team analysis card ── */
.team-card {
    display:flex; align-items:center; gap:18px;
    background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08);
    border-radius:16px; padding:20px 24px; margin-bottom:18px;
}
.team-card-name {
    font-family:'Syne',sans-serif; font-size:1.55rem; font-weight:800; color:#F0F6FF;
}
.team-card-sub {
    font-family:'DM Sans',sans-serif; font-size:.8rem; font-weight:700;
    letter-spacing:2px; text-transform:uppercase; margin-top:4px;
}

/* ── Fixtures header ── */
.fix-header {
    display:flex; align-items:center; gap:14px; margin-bottom:18px; flex-wrap:wrap;
}
.fix-title {
    font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800; color:#F0F6FF;
}
.fix-badge {
    font-family:'DM Sans',sans-serif; font-size:.85rem; font-weight:700;
    border-radius:8px; padding:4px 14px; border:1px solid;
}

/* Spinner */
.stSpinner > div { border-top-color:#38BDF8 !important; }
</style>"""

# ─── Legend ──────────────────────────────────────────────────
_LGND = [
    ("#FFD700","Champion"),("#38BDF8","UCL Top 4"),
    ("#34D399","Europa"),("#C084FC","Conference"),("#F87171","Relegation"),
]
def _legend():
    pills = "".join(
        f'<span class="lg-pill" style="color:{c};border-color:{c}33;background:{c}0D;">'
        f'<span class="lg-dot" style="background:{c};"></span>{l}</span>'
        for c, l in _LGND
    )
    st.markdown(f'<div class="lg-wrap">{pills}</div>', unsafe_allow_html=True)

def _divider(label: str):
    st.markdown(
        f'<div class="div-wrap"><span class="div-line"></span>'
        f'<span class="div-label">{label}</span>'
        f'<span class="div-line"></span></div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════
# MAIN PAGE
# ═══════════════════════════════════════════════════════════════
def page_season(ctx):

    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown("""
        <div style="margin-bottom:32px;">
            <div class="pg-eyebrow">⚡ Nexus Engine · Premier League</div>
            <div class="pg-title">Season <em>Table</em></div>
            <div class="pg-sub">ตารางคะแนนปัจจุบัน · AI จำลองผล · วิเคราะห์โอกาสทุกตำแหน่ง</div>
        </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["  Current Standings  ", "  AI Simulation  "])

    # ════════════════════════════════════════════════════════════
    # TAB 1 — Current Standings
    # ════════════════════════════════════════════════════════════
    with tab1:
        st.write("")
        yr   = TODAY.year
        base = yr if TODAY.month >= 8 else yr - 1
        opts = list(range(base, base - 6, -1))
        fmt  = {y: f"{y}/{str(y+1)[2:]}" for y in opts}

        c1, _ = st.columns([1, 3])
        sel = c1.selectbox(
            "Season", opts,
            format_func=lambda y: fmt[y],
            index=0, key="tab1_season"
        )

        @st.cache_data(ttl=300, show_spinner=False)
        def _fetch(y): return silent(get_pl_standings_from_api, y)

        with st.spinner("Fetching standings…"):
            api_rows = _fetch(sel)

        if not api_rows:
            st.markdown(
                '<div class="err-box">⚠️ ดึงข้อมูลจาก API ไม่ได้ — ตรวจสอบ API Key หรืออินเทอร์เน็ต</div>',
                unsafe_allow_html=True,
            )
        else:
            tbl = (
                pd.DataFrame(api_rows)
                .drop(columns=["pos"], errors="ignore")
                .reset_index(drop=True)
            )
            if "Form" not in tbl.columns:
                tbl["Form"] = ""
            n = len(tbl)

            rows = []
            for i, r in tbl.iterrows():
                pos = i + 1
                rows.append({
                    "pos":    pos,
                    "team":   str(r.get("Club", r.get("Team", "?"))),
                    "pts":    int(r.get("PTS", r.get("Pts", 0))),
                    "played": int(r.get("MP",  r.get("P",   0))),
                    "w":      int(r.get("W",   0)),
                    "d":      int(r.get("D",   0)),
                    "l":      int(r.get("L",   0)),
                    "gd":     int(r.get("GD",  0)),
                    "form":   str(r.get("Form", "")),
                    "zc":     _zone(pos, n),
                })

            html, h = _table_html(rows, sim=False)
            components.html(html, height=h, scrolling=False)
            _legend()

    # ════════════════════════════════════════════════════════════
    # TAB 2 — AI Simulation
    # ════════════════════════════════════════════════════════════
    with tab2:
        st.write("")

        b1, b2, _ = st.columns([1.2, 1.2, 2], gap="medium")
        run_btn  = b1.button("🔮  Run Simulation", type="primary",
                             use_container_width=True, key="btn_run")
        sync_btn = b2.button("↺  Sync Data",
                             use_container_width=True, key="btn_sync")

        if sync_btn:
            with st.spinner("Syncing from API…"):
                silent(update_season_csv_from_api)
                _load_stats.clear()        # invalidate cache after sync
            st.success("✅ Data synced!")

        if run_btn:
            with st.spinner("Running AI simulation…"):
                ctx_new = silent(run_season_simulation, ctx)
            if ctx_new is not None:
                st.session_state["ctx"] = ctx_new
                ctx = ctx_new
                st.success("✅ Simulation complete!")
            else:
                st.markdown(
                    '<div class="err-box">❌ การจำลองล้มเหลว — ลอง Sync Data แล้วรันใหม่</div>',
                    unsafe_allow_html=True,
                )

        ft = ctx.get("final_table") if ctx else None

        if ft is None:
            st.markdown("""
                <div class="ready-wrap">
                    <div class="ready-icon">🔮</div>
                    <div class="ready-title">Simulation Ready</div>
                    <div class="ready-sub">
                        กด <strong style="color:#38BDF8;">Run Simulation</strong>
                        เพื่อให้ AI จำลองผลแมตช์ที่เหลือ
                        และวิเคราะห์โอกาส&nbsp;% ในทุกตำแหน่ง
                    </div>
                </div>
            """, unsafe_allow_html=True)
            return

        try:
            # ── FIX: final_table index IS the team name ──────────────────
            # predict.py line 655: final_table.index.name = 'Team'
            # columns are ONLY: RealPoints, PredictedPoints, FinalPoints
            df = ft.copy().reset_index()
            if "Team" not in df.columns:
                df = df.rename(columns={df.columns[0]: "Team"})
            if "FinalPoints" not in df.columns:
                df["FinalPoints"] = (
                    df.get("RealPoints", pd.Series(0, index=df.index)) +
                    df.get("PredictedPoints", pd.Series(0, index=df.index))
                )
            df = df.sort_values("FinalPoints", ascending=False).reset_index(drop=True)
            n_teams = len(df)

            # ── W/D/L/GD from CSV (NOT in final_table) ───────────────────
            tstats = _load_stats()

            # ── Build table rows ──────────────────────────────────────────
            rows = []
            for i, r in df.iterrows():
                pos  = i + 1
                team = str(r["Team"])
                rp   = int(r.get("RealPoints",      0))
                pp   = int(r.get("PredictedPoints", 0))
                fp   = int(r.get("FinalPoints",  rp + pp))
                s    = tstats.get(team, {})
                rows.append({
                    "pos":    pos,
                    "team":   team,
                    "played": s.get("mp", 0),
                    "w":      s.get("w",  0),
                    "d":      s.get("d",  0),
                    "l":      s.get("l",  0),
                    "gd":     s.get("gf", 0) - s.get("ga", 0),
                    "form":   str(r.get("Form", "")),
                    "zc":     _zone(pos, n_teams),
                    "rp": rp, "pp": pp, "fp": fp,
                })

            html, h = _table_html(rows, sim=True)
            components.html(html, height=h, scrolling=False)
            _legend()

            # ── Quick summary metrics ─────────────────────────────────────
            st.write("")
            champ = df.iloc[0]
            ucl4  = df.iloc[:4]
            rel3  = df.iloc[-3:]   # always exactly last 3

            m1, m2, m3 = st.columns(3, gap="medium")
            m1.metric(
                "🏆 Projected Champion",
                str(champ["Team"]),
                f"{int(champ['FinalPoints'])} pts",
            )
            ucl_str = "  ·  ".join(str(r["Team"]) for _, r in ucl4.iterrows())
            m2.metric("⚽ UCL Top 4", "4 clubs",
                      ucl_str[:44] + "…" if len(ucl_str) > 44 else ucl_str)
            rel_str = "  ·  ".join(str(r["Team"]) for _, r in rel3.iterrows())
            m3.metric("🔻 Relegation Zone", "Bottom 3",
                      rel_str[:44] + "…" if len(rel_str) > 44 else rel_str)

            # ════════════════════════════════════════════════════════════
            # TEAM ANALYSIS
            # ════════════════════════════════════════════════════════════
            _divider("Team Analysis · Select a Club")

            all_teams = sorted(str(r["Team"]) for _, r in df.iterrows())
            sel_team  = st.selectbox(
                "เลือกทีม", all_teams,
                key="team_sel", label_visibility="collapsed"
            )

            teams_list = [str(r["Team"]) for _, r in df.iterrows()]
            base_pts   = np.array([float(r["FinalPoints"]) for _, r in df.iterrows()])
            pts_map    = {str(r["Team"]): float(r["FinalPoints"]) for _, r in df.iterrows()}
            cur_pos    = teams_list.index(sel_team) + 1

            # Monte Carlo — rank distribution
            N_SIM = 600
            rng   = np.random.default_rng(42)
            rank_cnt = {t: np.zeros(n_teams, dtype=int) for t in teams_list}
            for _ in range(N_SIM):
                noisy = base_pts + rng.normal(0, 5.0, n_teams)
                for rk0, idx in enumerate(np.argsort(-noisy)):
                    rank_cnt[teams_list[idx]][rk0] += 1
            sel_pct = (rank_cnt[sel_team] / N_SIM * 100).round(1)

            # Remaining fixtures for selected team
            remaining     = ctx.get("remaining_fixtures", [])
            team_fix      = [
                m for m in remaining
                if m.get("HomeTeam") == sel_team or m.get("AwayTeam") == sel_team
            ]

            fix_res = []
            for m in team_fix:
                ht, at  = m["HomeTeam"], m["AwayTeam"]
                is_home = (ht == sel_team)
                opp     = at if is_home else ht
                hs      = pts_map.get(ht, 50.)
                as_     = pts_map.get(at, 50.)
                diff    = hs - as_
                ph = 1 / (1 + 10 ** (-diff / 30))
                pd_ = max(.05, .28 - abs(diff) * .003)
                pa  = max(.05, 1 - ph - pd_)
                tot = ph + pd_ + pa
                ph /= tot; pd_ /= tot; pa /= tot
                pw, pl = (ph, pa) if is_home else (pa, ph)

                if pw >= pd_ and pw >= pl:   pred, pc = "WIN",  "#34D399"
                elif pd_ >= pl:              pred, pc = "DRAW", "#F59E0B"
                else:                        pred, pc = "LOSS", "#F87171"

                fix_res.append({
                    "opp":   opp,
                    "venue": "H" if is_home else "A",
                    "pw":    round(pw  * 100, 1),
                    "pd":    round(pd_ * 100, 1),
                    "pl":    round(pl  * 100, 1),
                    "pred":  pred,
                    "pc":    pc,
                })

            # ── Render two-column analysis ────────────────────────────────
            col_prob, col_fix = st.columns([1, 1], gap="large")

            # ─── LEFT: Position probability bars ─────────────────────────
            with col_prob:
                z_cur = _zone(cur_pos, n_teams)
                logo  = _img_tag(sel_team, 44)

                st.markdown(
                    f'<div class="team-card">'
                    f'{logo}'
                    f'<div>'
                    f'<div class="team-card-name">{sel_team}</div>'
                    f'<div class="team-card-sub" style="color:{z_cur["c"]};">'
                    f'Projected #{cur_pos} · {int(pts_map.get(sel_team, 0))} pts</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

                # Build position-probability rows
                bar_rows = []
                for rk0, pct in enumerate(sel_pct):
                    rk   = rk0 + 1
                    zz   = _zone(rk, n_teams)
                    zc   = zz["c"];  zlbl = zz["l"]
                    cur  = (rk == cur_pos)
                    bw   = max(1, int(pct * 0.95))   # % width capped at 95%
                    op   = max(.28, min(1., pct / 30 + .25))
                    hlbg = f"background:linear-gradient(90deg,{zc}14,transparent);" if cur else ""
                    dot  = (
                        f'<span style="color:{zc};font-size:1rem;'
                        f'line-height:1;flex-shrink:0;">◆</span>'
                        if cur else
                        f'<span style="width:1rem;flex-shrink:0;"></span>'
                    )
                    badge = (
                        f'<span style="font-family:\'DM Sans\',sans-serif;font-size:.65rem;'
                        f'font-weight:800;letter-spacing:1.5px;text-transform:uppercase;'
                        f'color:{zc};background:{zc}14;border:1px solid {zc}30;'
                        f'border-radius:5px;padding:1px 8px;white-space:nowrap;'
                        f'margin-left:8px;">{zlbl}</span>'
                        if zlbl else ""
                    )
                    bar_rows.append(
                        f'<div style="display:flex;align-items:center;gap:10px;'
                        f'padding:7px 12px;border-radius:9px;{hlbg}margin-bottom:2px;">'
                        f'{dot}'
                        f'<span style="font-family:\'JetBrains Mono\',monospace;'
                        f'font-size:.95rem;font-weight:700;color:{zc};'
                        f'width:26px;text-align:right;flex-shrink:0;opacity:{op:.2f};">{rk}</span>'
                        f'<div style="flex:1;background:rgba(255,255,255,.07);'
                        f'border-radius:99px;height:9px;overflow:hidden;">'
                        f'<div style="width:{bw}%;height:100%;background:{zc};'
                        f'opacity:{op:.2f};border-radius:99px;"></div></div>'
                        f'<span style="font-family:\'JetBrains Mono\',monospace;'
                        f'font-size:.95rem;font-weight:700;color:{zc};'
                        f'width:46px;text-align:right;flex-shrink:0;'
                        f'opacity:{op:.2f};">{pct}%</span>'
                        f'{badge}'
                        f'</div>'
                    )

                prob_html = (
                    f'<html><head><style>{_GF}'
                    f'*{{margin:0;padding:0;box-sizing:border-box;}}'
                    f'html,body{{background:#060F1C;overflow:hidden;}}'
                    f'</style></head><body>'
                    f'<div style="background:rgba(255,255,255,.025);'
                    f'border:1px solid rgba(255,255,255,.08);border-radius:14px;'
                    f'padding:16px 14px;">'
                    f'<div style="font-family:\'DM Sans\',sans-serif;font-size:.75rem;'
                    f'font-weight:700;letter-spacing:2.5px;text-transform:uppercase;'
                    f'color:rgba(148,187,233,.28);margin-bottom:12px;padding:0 12px;">'
                    f'Finish Position Probability</div>'
                    f'{"".join(bar_rows)}'
                    f'</div></body></html>'
                )
                prob_h = 40 + n_teams * 38 + 16
                components.html(prob_html, height=prob_h, scrolling=False)

            # ─── RIGHT: Remaining fixtures ────────────────────────────────
            with col_fix:
                wins_  = sum(1 for f in fix_res if f["pred"] == "WIN")
                draws_ = sum(1 for f in fix_res if f["pred"] == "DRAW")
                loss_  = sum(1 for f in fix_res if f["pred"] == "LOSS")

                st.markdown(
                    f'<div class="fix-header">'
                    f'<span class="fix-title">Remaining Fixtures</span>'
                    f'<span class="fix-badge" style="color:#34D399;'
                    f'border-color:#34D39930;background:#34D39910;">W&nbsp;{wins_}</span>'
                    f'<span class="fix-badge" style="color:#F59E0B;'
                    f'border-color:#F59E0B30;background:#F59E0B10;">D&nbsp;{draws_}</span>'
                    f'<span class="fix-badge" style="color:#F87171;'
                    f'border-color:#F8717130;background:#F8717110;">L&nbsp;{loss_}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                if not fix_res:
                    st.markdown(
                        '<div style="font-family:\'DM Sans\',sans-serif;font-size:1.05rem;'
                        'color:rgba(148,187,233,.3);padding:28px 0;text-align:center;">'
                        'ไม่มีแมตช์ที่เหลือ</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    fix_parts = []
                    for f in fix_res:
                        opp  = f["opp"]
                        vc   = "#38BDF8" if f["venue"] == "H" else "#F59E0B"
                        pc   = f["pc"]
                        ologo = _img_tag(opp, 28)

                        def _mbar(pct: float, col: str) -> str:
                            w = max(2, int(pct * 0.8))
                            return (
                                f'<span style="display:inline-flex;align-items:center;gap:5px;">'
                                f'<span style="display:inline-block;width:{w}px;max-width:75px;'
                                f'height:5px;border-radius:99px;background:{col};opacity:.85;'
                                f'flex-shrink:0;"></span>'
                                f'<span style="font-family:\'JetBrains Mono\',monospace;'
                                f'font-size:.82rem;font-weight:700;color:{col};">{pct}%</span>'
                                f'</span>'
                            )

                        fix_parts.append(
                            f'<div style="display:flex;align-items:center;gap:12px;'
                            f'padding:12px 16px;border-bottom:1px solid rgba(255,255,255,.05);">'
                            # H/A badge
                            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:.82rem;'
                            f'font-weight:700;color:{vc};background:{vc}14;border:1px solid {vc}30;'
                            f'border-radius:6px;padding:4px 9px;flex-shrink:0;">{f["venue"]}</span>'
                            # opponent logo + name
                            f'<span style="display:inline-flex;align-items:center;gap:11px;'
                            f'flex:1;min-width:0;">'
                            f'{ologo}'
                            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:1.05rem;'
                            f'font-weight:700;color:#F0F6FF;white-space:nowrap;'
                            f'overflow:hidden;text-overflow:ellipsis;">{opp}</span>'
                            f'</span>'
                            # W/D/L bars
                            f'<span style="display:inline-flex;gap:10px;align-items:center;flex-shrink:0;">'
                            f'{_mbar(f["pw"],"#34D399")}'
                            f'{_mbar(f["pd"],"#F59E0B")}'
                            f'{_mbar(f["pl"],"#F87171")}'
                            f'</span>'
                            # prediction badge
                            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:.75rem;'
                            f'font-weight:800;letter-spacing:1.5px;color:{pc};'
                            f'background:{pc}14;border:1px solid {pc}33;'
                            f'border-radius:6px;padding:4px 11px;flex-shrink:0;">{f["pred"]}</span>'
                            f'</div>'
                        )

                    fix_html = (
                        f'<html><head><style>{_GF}'
                        f'*{{margin:0;padding:0;box-sizing:border-box;}}'
                        f'html,body{{background:#060F1C;overflow:hidden;}}'
                        f'div[data-row]:hover{{background:rgba(56,189,248,.05);}}'
                        f'</style></head><body>'
                        f'<div style="background:rgba(255,255,255,.025);'
                        f'border:1px solid rgba(255,255,255,.08);border-radius:14px;overflow:hidden;">'
                        f'{"".join(fix_parts)}'
                        f'</div></body></html>'
                    )
                    fix_h = len(fix_res) * 58 + 8
                    components.html(fix_html, height=fix_h, scrolling=False)

        except Exception as exc:
            import traceback
            st.markdown(
                f'<div class="err-box">❌ เกิดข้อผิดพลาด: {exc}</div>',
                unsafe_allow_html=True,
            )
            with st.expander("🐛 Debug traceback"):
                st.code(traceback.format_exc(), language="python")
            st.info("ลอง Sync Data แล้ว Run Simulation ใหม่")