"""
ui/sidebar.py — Sidebar navigation (NEXUS ENGINE v9.0)
Redesigned to match the main dashboard's dark navy theme.
"""
import streamlit as st

def render_sidebar(ctx=None):
    """Render the sidebar — unified with main content color system."""

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;600;700&display=swap');

        /* ═══════════════════════════════════════
           SIDEBAR SHELL  —  match main bg exactly
        ═══════════════════════════════════════ */
        [data-testid="stSidebar"] {
            background: #0B1628 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0;
            padding-left: 0;
            padding-right: 0;
            display: flex !important;
            flex-direction: column !important;
            height: 100vh !important;
        }

        /* Nav wrap fills remaining space and centers content vertically */
        .nx-nav-wrap {
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        /* ═══════════════════════════════════════
           LOGO AREA
        ═══════════════════════════════════════ */
        .nx-logo-block {
            padding: 24px 20px 20px;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }

        .nx-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.6rem;
            font-weight: 700;
            letter-spacing: 3px;
            text-transform: uppercase;
            color: #38BDF8;
            background: rgba(56, 189, 248, 0.08);
            border: 1px solid rgba(56, 189, 248, 0.2);
            padding: 3px 10px;
            border-radius: 3px;
            margin-bottom: 10px;
        }

        .nx-eyebrow::before {
            content: '';
            width: 5px; height: 5px;
            border-radius: 50%;
            background: #38BDF8;
            box-shadow: 0 0 6px #38BDF8;
            animation: nx-ping 2.5s ease-in-out infinite;
        }

        .nx-logo-text {
            font-family: 'Orbitron', sans-serif;
            font-size: 2rem;
            font-weight: 900;
            color: #F0F6FF;
            letter-spacing: 1px;
            line-height: 1;
        }

        .nx-logo-text em {
            font-style: normal;
            color: #38BDF8;
        }

        .nx-version {
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.65rem;
            font-weight: 600;
            letter-spacing: 4px;
            color: rgba(148, 187, 233, 0.4);
            text-transform: uppercase;
            margin-top: 6px;
        }

        /* ═══════════════════════════════════════
           NAV SECTION
        ═══════════════════════════════════════ */
        .nx-nav-header {
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.58rem;
            font-weight: 700;
            letter-spacing: 3.5px;
            text-transform: uppercase;
            color: rgba(148, 187, 233, 0.35);
            padding: 20px 20px 8px;
        }

        [data-testid="stSidebar"] .stRadio {
            padding: 0 10px;
        }

        [data-testid="stSidebar"] .stRadio > div {
            gap: 2px !important;
        }

        /* ── Reset ALL Streamlit internal radio structures ── */
        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
            gap: 2px !important;
        }

        /* Kill every possible Streamlit-injected border/bg on the wrapper divs */
        [data-testid="stSidebar"] .stRadio div[data-baseweb="radio"],
        [data-testid="stSidebar"] .stRadio div[data-baseweb="block"],
        [data-testid="stSidebar"] .stRadio div[class*="st-"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }

        /* ── The label itself is our entire nav item ── */
        [data-testid="stSidebar"] .stRadio label {
            font-family: 'Rajdhani', sans-serif !important;
            font-size: 0.92rem !important;
            font-weight: 600 !important;
            letter-spacing: 1px !important;
            color: rgba(148, 187, 233, 0.55) !important;
            background: transparent !important;
            border: 1px solid transparent !important;
            border-radius: 6px !important;
            padding: 9px 14px !important;
            margin: 0 !important;
            width: 100% !important;
            box-sizing: border-box !important;
            cursor: pointer !important;
            transition: color 0.15s ease, background 0.15s ease, border-color 0.15s ease !important;
            position: relative !important;
            overflow: hidden !important;
            box-shadow: none !important;
            outline: none !important;
        }

        [data-testid="stSidebar"] .stRadio label:hover {
            color: rgba(240, 246, 255, 0.85) !important;
            background: rgba(56, 189, 248, 0.06) !important;
            border-color: rgba(56, 189, 248, 0.15) !important;
        }

        /* ── Active item ── */
        [data-testid="stSidebar"] .stRadio label:has(input:checked) {
            color: #F0F6FF !important;
            background: rgba(56, 189, 248, 0.10) !important;
            border-color: rgba(56, 189, 248, 0.30) !important;
        }

        /* Left accent stripe on active only */
        [data-testid="stSidebar"] .stRadio label:has(input:checked)::before {
            content: '';
            position: absolute;
            left: 0; top: 4px; bottom: 4px;
            width: 3px;
            background: linear-gradient(180deg, #38BDF8 0%, #0EA5E9 100%);
            border-radius: 0 2px 2px 0;
        }

        /* Hide default radio circle */
        [data-testid="stSidebar"] .stRadio label > div:first-child {
            display: none !important;
        }

        /* Kill focus rings */
        [data-testid="stSidebar"] .stRadio label:focus-within,
        [data-testid="stSidebar"] .stRadio label:focus {
            outline: none !important;
            box-shadow: none !important;
            border-color: transparent !important;
        }

        [data-testid="stSidebar"] .stRadio label:has(input:checked):focus-within {
            border-color: rgba(56, 189, 248, 0.30) !important;
        }

        /* ═══════════════════════════════════════
           STATUS FOOTER
        ═══════════════════════════════════════ */
        .nx-footer {
            padding: 16px 20px;
            border-top: 1px solid rgba(255,255,255,0.06);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .nx-status-dot {
            width: 6px; height: 6px;
            border-radius: 50%;
            background: #22C55E;
            box-shadow: 0 0 6px rgba(34, 197, 94, 0.7);
            flex-shrink: 0;
            animation: nx-ping 2.5s ease-in-out infinite;
        }

        .nx-status-text {
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.62rem;
            font-weight: 600;
            letter-spacing: 2px;
            color: rgba(148, 187, 233, 0.35);
            text-transform: uppercase;
        }

        @keyframes nx-ping {
            0%, 100% { opacity: 1; }
            50%       { opacity: 0.25; }
        }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:

        # ── LOGO ──
        st.markdown("""
            <div class="nx-logo-block">
                <div ></div>
                <div class="nx-logo-text">แมนยู<em>5</em>พอลล</div>
                <div class="nx-version">⚡ &nbsp;Nexus Engine &nbsp;v9.0</div>
            </div>
        """, unsafe_allow_html=True)

        # ── NAV WRAP open (fills space, centers nav vertically) ──
        st.markdown('<div class="nx-nav-wrap">', unsafe_allow_html=True)

        # ── NAV HEADER ──
        st.markdown('<div class="nx-nav-header">Navigation</div>', unsafe_allow_html=True)

        # ── NAV ITEMS ──
        nav_icons = {
            "Overview":      "  Overview",
            "Predict Match": "  Predict Match",
            "Next Fixtures": "  Next Fixtures",
            "Season Table":  "  Season Table",
            "Analysis":      "  Analysis",
            "Update Data":   "  Update Data",
        }

        st.radio(
            "Navigation",
            list(nav_icons.keys()),
            key="nav_page",
            label_visibility="collapsed",
            format_func=lambda x: nav_icons.get(x, x),
        )

        st.markdown('</div>', unsafe_allow_html=True)

        # ── STATUS FOOTER ──
        st.markdown("""
            <div class="nx-footer">
                <div class="nx-status-dot"></div>
                <span class="nx-status-text">System Online &nbsp;·&nbsp; v9.0.1</span>
            </div>
        """, unsafe_allow_html=True)

        stabilize_on = bool(ctx and ctx.get('stabilize_connected'))
        if stabilize_on:
            cfg = ctx.get('stabilize_settings', {})
            t_home = cfg.get('t_home')
            t_draw = cfg.get('t_draw')
            selected_profile = ctx.get('stabilize_selected_profile')
            st.caption('STABILIZE linked (monitoring only).')
            if selected_profile:
                st.caption(f"profile: {selected_profile}")
            if t_home is not None and t_draw is not None:
                st.caption(f"report thresholds: home={float(t_home):.3f}, draw={float(t_draw):.3f}")
        else:
            st.caption('STABILIZE report not found.')


