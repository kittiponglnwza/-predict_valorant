"""
ui/page_season.py — Season Table page for Football AI Nexus Engine
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from src.config import TODAY
from src.predict import run_season_simulation, update_season_csv_from_api, get_pl_standings_from_api
from utils import silent, make_styled_table, zone_label, find_team_col


# ── Zone config ──────────────────────────────────────────────────────────────
ZONE_CONFIG = {
    "👑": {"label": "Champion",   "color": "#F59E0B", "bg": "rgba(245,158,11,0.12)",  "border": "rgba(245,158,11,0.35)"},
    "⚽": {"label": "UCL",        "color": "#38BDF8", "bg": "rgba(56,189,248,0.10)",  "border": "rgba(56,189,248,0.30)"},
    "🌍": {"label": "Europa",     "color": "#34D399", "bg": "rgba(52,211,153,0.10)",  "border": "rgba(52,211,153,0.28)"},
    "🏅": {"label": "Conference", "color": "#A78BFA", "bg": "rgba(167,139,250,0.10)", "border": "rgba(167,139,250,0.28)"},
    "🔻": {"label": "Relegation", "color": "#F87171", "bg": "rgba(248,113,113,0.10)", "border": "rgba(248,113,113,0.30)"},
}

FORM_COLOR = {"W": "#22C55E", "D": "#F59E0B", "L": "#EF4444"}


def _zone_style(zone_str):
    for emoji, cfg in ZONE_CONFIG.items():
        if emoji in str(zone_str):
            # ensure label key always present
            result = dict(cfg)
            if "label" not in result:
                result["label"] = ""
            return result
    return {"label": "", "color": "rgba(148,187,233,0.4)", "bg": "transparent", "border": "transparent"}


def _form_dots(form_str):
    if not form_str:
        return ""
    mapping = {"W": "#22C55E", "D": "#F59E0B", "L": "#EF4444"}
    dots = []
    for c in form_str.split(","):
        c = c.strip()
        if c in mapping:
            dots.append(f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
                        f'background:{mapping[c]};margin-right:3px;vertical-align:middle;"></span>')
    return "".join(dots)


def _row_html(pos, team, pts, played, w, d, l, gd, form_str, zone_str, proj=None, final=None):
    z = _zone_style(zone_str)
    lbl = z.get("label", "")
    c   = z.get("color", "rgba(148,187,233,0.4)")
    bg  = z.get("bg", "transparent")
    bd  = z.get("border", "transparent")

    zone_badge = (f'<span style="font-size:.6rem;font-weight:700;letter-spacing:1.5px;font-family:Rajdhani,sans-serif;color:{c};background:{bg};border:1px solid {bd};border-radius:3px;padding:2px 7px;text-transform:uppercase;">{lbl}</span>') if lbl else ""

    gd_color = "#34D399" if gd > 0 else ("#F87171" if gd < 0 else "rgba(148,187,233,.5)")
    gd_str   = f"+{gd}" if gd > 0 else str(gd)
    form_html = _form_dots(form_str) if form_str else ""

    proj_td = ""
    if proj is not None and final is not None:
        pc = "#22C55E" if proj > 0 else ("#F87171" if proj < 0 else "rgba(148,187,233,.4)")
        ps = f"+{proj}" if proj > 0 else str(proj)
        proj_td = (f'<td style="text-align:center;padding:0 8px;"><span style="font-family:Orbitron,sans-serif;font-size:1.05rem;font-weight:700;color:#F0F6FF;">{final}</span></td>'
                   f'<td style="text-align:center;padding:0 8px;"><span style="font-family:Rajdhani,sans-serif;font-size:.85rem;font-weight:700;color:{pc};">{ps}</span></td>')

    pts_td = f'<td style="text-align:center;padding:0 8px;"><span style="font-family:Orbitron,sans-serif;font-size:1.1rem;font-weight:700;color:#F0F6FF;text-shadow:0 0 12px rgba(56,189,248,.4);">{pts}</span></td>'

    return (
        f'<tr style="border-bottom:1px solid rgba(255,255,255,.05);">'
        f'<td style="padding:10px 12px;text-align:center;width:36px;"><span style="font-family:Orbitron,sans-serif;font-size:.85rem;font-weight:700;color:{c};">{pos}</span></td>'
        f'<td style="padding:10px 12px;min-width:160px;"><span style="font-family:Rajdhani,sans-serif;font-size:1.05rem;font-weight:700;color:#F0F6FF;">{team}</span></td>'
        f'<td style="text-align:center;padding:0 8px;font-family:Rajdhani,sans-serif;font-size:.9rem;color:rgba(148,187,233,.6);">{played}</td>'
        f'<td style="text-align:center;padding:0 8px;font-family:Rajdhani,sans-serif;font-size:.9rem;font-weight:600;color:#34D399;">{w}</td>'
        f'<td style="text-align:center;padding:0 8px;font-family:Rajdhani,sans-serif;font-size:.9rem;color:#F59E0B;">{d}</td>'
        f'<td style="text-align:center;padding:0 8px;font-family:Rajdhani,sans-serif;font-size:.9rem;color:#F87171;">{l}</td>'
        f'<td style="text-align:center;padding:0 8px;"><span style="font-family:Rajdhani,sans-serif;font-size:.9rem;font-weight:700;color:{gd_color};">{gd_str}</span></td>'
        f'{pts_td}{proj_td}'
        f'<td style="padding:0 12px;">{form_html}</td>'
        f'<td style="padding:0 12px;text-align:right;">{zone_badge}</td>'
        f'</tr>'
    )


def _table_html(rows_html, sim=False):
    th = ("font-family:'Rajdhani',sans-serif;font-size:0.6rem;font-weight:700;"
          "letter-spacing:2.5px;text-transform:uppercase;color:rgba(148,187,233,0.35);"
          "padding:8px 8px 12px;text-align:center;white-space:nowrap;")
    th_left = th.replace("text-align:center", "text-align:left")

    proj_headers = (
        f'<th style="{th}">FINAL</th><th style="{th}">+PROJ</th>'
        if sim else ""
    )
    rows_joined = "".join(rows_html)

    font_import = "@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;600;700&display=swap');"
    return (
        f'<html><head><style>{font_import}'
        f'*{{margin:0;padding:0;box-sizing:border-box;}}'
        f'body{{background:#0B1628;}}'
        f'table{{width:100%;border-collapse:collapse;font-family:Rajdhani,sans-serif;}}'
        f'tr:hover{{background:rgba(56,189,248,.04)!important;}}'
        f'</style></head><body>'
        f'<div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.09);border-radius:10px;overflow:hidden;">'
        f'<table>'
        f'<thead><tr style="border-bottom:1px solid rgba(255,255,255,.10);">'
        f'<th style="{th}">#</th>'
        f'<th style="{th_left}">Club</th>'
        f'<th style="{th}">MP</th>'
        f'<th style="{th}">W</th>'
        f'<th style="{th}">D</th>'
        f'<th style="{th}">L</th>'
        f'<th style="{th}">GD</th>'
        f'<th style="{th}">PTS</th>'
        f'{proj_headers}'
        f'<th style="{th_left}">Form</th>'
        f'<th style="{th}">Zone</th>'
        f'</tr></thead>'
        f'<tbody>{rows_joined}</tbody>'
        f'</table></div></body></html>'
    )


def page_season(ctx):

    # ── STYLES ────────────────────────────────────────────────────────────────
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;600;700&display=swap');

        .main .block-container { padding-top: 2rem; max-width: 1100px; }

        .sx-eyebrow {
            font-family: 'Rajdhani', sans-serif; font-size: 0.75rem; font-weight: 700;
            letter-spacing: 4px; text-transform: uppercase; color: #38BDF8; margin-bottom: 6px;
        }
        .sx-title {
            font-family: 'Orbitron', sans-serif; font-size: 2.8rem; font-weight: 900;
            color: #F0F6FF; letter-spacing: 1px; line-height: 1.1;
        }
        .sx-title em { font-style: normal; color: #38BDF8; }
        .sx-subtitle {
            font-family: 'Rajdhani', sans-serif; font-size: 1.0rem;
            color: rgba(148,187,233,0.5); letter-spacing: 1px; margin-top: 4px;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px !important;
            background: transparent !important;
            border-bottom: 1px solid rgba(255,255,255,0.08) !important;
            padding-bottom: 0 !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-family: 'Rajdhani', sans-serif !important;
            font-size: 0.85rem !important;
            font-weight: 700 !important;
            letter-spacing: 2px !important;
            text-transform: uppercase !important;
            color: rgba(148,187,233,0.5) !important;
            background: transparent !important;
            border: none !important;
            border-radius: 6px 6px 0 0 !important;
            padding: 10px 20px !important;
        }
        .stTabs [aria-selected="true"] {
            color: #F0F6FF !important;
            background: rgba(56,189,248,0.08) !important;
            border-bottom: 2px solid #38BDF8 !important;
        }

        /* Selectbox label */
        .stSelectbox label {
            font-family: 'Rajdhani', sans-serif !important;
            font-size: 0.7rem !important; letter-spacing: 2px !important;
            text-transform: uppercase !important; color: rgba(148,187,233,0.5) !important;
            font-weight: 700 !important;
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.04) !important;
            border: 1px solid rgba(255,255,255,0.09) !important;
            border-radius: 8px !important;
            padding: 16px !important;
        }
        [data-testid="stMetricLabel"] {
            font-family: 'Rajdhani', sans-serif !important;
            font-size: 0.65rem !important; letter-spacing: 2px !important;
            text-transform: uppercase !important; color: rgba(148,187,233,0.45) !important;
        }
        [data-testid="stMetricValue"] {
            font-family: 'Orbitron', sans-serif !important;
            font-size: 1.1rem !important; color: #F0F6FF !important;
        }
        [data-testid="stMetricDelta"] {
            font-family: 'Rajdhani', sans-serif !important;
            font-size: 0.78rem !important;
        }

        /* Button */
        .stButton > button {
            font-family: 'Rajdhani', sans-serif !important;
            font-weight: 700 !important; letter-spacing: 2px !important;
            font-size: 0.85rem !important; border-radius: 6px !important;
            text-transform: uppercase !important;
        }

        /* Legend */
        .sx-legend {
            display: flex; flex-wrap: wrap; gap: 10px;
            margin-top: 14px; padding-top: 14px;
            border-top: 1px solid rgba(255,255,255,0.06);
        }
        .sx-legend-item {
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.65rem; font-weight: 700; letter-spacing: 1.5px;
            text-transform: uppercase; padding: 3px 10px;
            border-radius: 3px; white-space: nowrap;
        }

        /* Simulation ready card */
        .sx-sim-ready {
            text-align: center; padding: 4rem 2rem;
            background: rgba(255,255,255,0.02);
            border: 1px dashed rgba(56,189,248,0.2);
            border-radius: 12px; margin-top: 1rem;
        }
        .sx-sim-ready-title {
            font-family: 'Orbitron', sans-serif; font-size: 1.6rem;
            font-weight: 900; color: #F0F6FF; letter-spacing: 2px; margin: 12px 0 8px;
        }
        .sx-sim-ready-sub {
            font-family: 'Rajdhani', sans-serif; font-size: 0.95rem;
            color: rgba(148,187,233,0.45); letter-spacing: 1px;
        }
        </style>
    """, unsafe_allow_html=True)

    # ── PAGE HEADER ───────────────────────────────────────────────────────────
    st.markdown("""
        <div style="margin-bottom:28px;">
            <div class="sx-eyebrow">⚡ Nexus Engine · Premier League</div>
            <div class="sx-title">Season <em>Table</em></div>
            <div class="sx-subtitle">ตารางคะแนนปัจจุบัน หรือให้ AI จำลองผลถึงสิ้นฤดูกาล</div>
        </div>
    """, unsafe_allow_html=True)

    tab_current, tab_sim = st.tabs(["  Current Standings  ", "  AI Simulation  "])

    # ════════════════════════════════════════════════════════════════
    # TAB 1 — Current Standings
    # ════════════════════════════════════════════════════════════════
    with tab_current:
        st.write("")
        current_year = TODAY.year
        current_season_year = current_year if TODAY.month >= 8 else current_year - 1
        season_options = list(range(current_season_year, current_season_year - 6, -1))
        season_labels  = {y: f"{y}/{str(y + 1)[2:]}" for y in season_options}

        sc1, _ = st.columns([1, 3])
        selected_year = sc1.selectbox(
            "Season",
            options=season_options,
            format_func=lambda y: season_labels[y],
            index=0,
            key="season_year_select"
        )


        @st.cache_data(ttl=300, show_spinner=False)
        def _fetch_standings(year):
            return silent(get_pl_standings_from_api, year)

        with st.spinner("Fetching standings..."):
            rows = _fetch_standings(selected_year)

        if rows is None or len(rows) == 0:
            st.markdown("""
                <div style="background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.25);
                     border-radius:8px;padding:16px 20px;font-family:'Rajdhani',sans-serif;
                     font-size:0.95rem;color:rgba(252,165,165,0.8);letter-spacing:0.5px;">
                    ⚠️ &nbsp; ดึงข้อมูลจาก API ไม่ได้ — ตรวจสอบ API Key หรือการเชื่อมต่ออินเทอร์เน็ต
                </div>
            """, unsafe_allow_html=True)
        else:
            tbl = pd.DataFrame(rows)
            tbl = tbl.drop(columns=['pos'], errors='ignore')
            if 'Form' not in tbl.columns:
                tbl['Form'] = ""
            tbl.index = range(1, len(tbl) + 1)
            tbl['Zone'] = [zone_label(i) for i in tbl.index]

            row_htmls = []
            for pos in tbl.index:
                r = tbl.loc[pos]
                row_htmls.append(_row_html(
                    pos=pos,
                    team=str(r.get('Team', r.get('Club', '?'))),
                    pts=int(r.get('PTS', r.get('Pts', 0))),
                    played=int(r.get('MP', r.get('P', 0))),
                    w=int(r.get('W', 0)),
                    d=int(r.get('D', 0)),
                    l=int(r.get('L', 0)),
                    gd=int(r.get('GD', 0)),
                    form_str=str(r.get('Form', '')),
                    zone_str=str(r.get('Zone', '')),
                ))

            components.html(_table_html(row_htmls, sim=False), height=760, scrolling=True)

            # Legend
            legend_items = "".join([
                f'<span class="sx-legend-item" style="color:{cfg["color"]};background:{cfg["bg"]};border:1px solid {cfg["border"]};">'
                f'{emoji} {cfg["label"]}</span>'
                for emoji, cfg in ZONE_CONFIG.items()
            ])
            st.markdown(f'<div class="sx-legend">{legend_items}</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 2 — AI Simulation
    # ════════════════════════════════════════════════════════════════
    with tab_sim:
        st.write("")

        b1, b2, _ = st.columns([1, 1, 2], gap="medium")
        sim_clicked = b1.button("Run Simulation", type="primary", use_container_width=True, key="run_sim")
        sync2       = b2.button("↺  Sync Data",   use_container_width=True, key="sync_sim")

        if sync2:
            with st.spinner("Syncing..."):
                silent(update_season_csv_from_api)
            st.success("✅ Synced!")

        if sim_clicked:
            with st.spinner("Simulating remaining fixtures..."):
                ctx_new = silent(run_season_simulation, ctx)
                st.session_state['ctx'] = ctx_new
                ctx = ctx_new

        ft = ctx.get('final_table')

        if ft is not None:
            df = ft.sort_values('FinalPoints', ascending=False).reset_index(drop=False)
            df.index = range(1, len(df) + 1)
            team_col = find_team_col(df)

            row_htmls = []
            for pos in df.index:
                r = df.loc[pos]
                real  = int(r.get('RealPoints', 0))
                final = int(r.get('FinalPoints', 0))
                row_htmls.append(_row_html(
                    pos=pos,
                    team=str(r.get(team_col, '?')),
                    pts=real,
                    played=int(r.get('MP', r.get('P', 0))),
                    w=int(r.get('W', 0)),
                    d=int(r.get('D', 0)),
                    l=int(r.get('L', 0)),
                    gd=int(r.get('GD', 0)),
                    form_str=str(r.get('Form', '')),
                    zone_str=zone_label(pos),
                    proj=final - real,
                    final=final,
                ))

            components.html(_table_html(row_htmls, sim=True), height=760, scrolling=True)

            # Legend
            legend_items = "".join([
                f'<span class="sx-legend-item" style="color:{cfg["color"]};background:{cfg["bg"]};border:1px solid {cfg["border"]};">'
                f'{emoji} {cfg["label"]}</span>'
                for emoji, cfg in ZONE_CONFIG.items()
            ])
            st.markdown(f'<div class="sx-legend">{legend_items}</div>', unsafe_allow_html=True)

            # Highlights
            st.write("")
            champion  = df.iloc[0]
            ucl_teams = df[df.index <= 4]
            relegated = df[df.index >= 18]
            s1, s2, s3 = st.columns(3, gap="medium")
            s1.metric("Champion",
                      str(champion.get(team_col, '—')),
                      f"{int(champion['FinalPoints'])} pts")
            if not ucl_teams.empty:
                names = ", ".join(str(r.get(team_col, '?')) for _, r in ucl_teams.iterrows())
                s2.metric("UCL Top 4", f"{len(ucl_teams)} clubs",
                          names[:32] + "…" if len(names) > 32 else names)
            if not relegated.empty:
                names = ", ".join(str(r.get(team_col, '?')) for _, r in relegated.iterrows())
                s3.metric("Relegated", f"{len(relegated)} clubs",
                          names[:32] + "…" if len(names) > 32 else names)
        else:
            st.markdown("""
                <div class="sx-sim-ready">
                    <div style="font-size:2.5rem;opacity:0.35;">🔮</div>
                    <div class="sx-sim-ready-title">SIMULATION READY</div>
                    <div class="sx-sim-ready-sub">
                        กด <strong style="color:#38BDF8;">Run Simulation</strong> เพื่อให้ AI คาดการณ์ผลจนจบฤดูกาล
                    </div>
                </div>
            """, unsafe_allow_html=True)