"""
ui/sidebar.py — Sidebar navigation & system status for Football AI Nexus Engine
"""
import streamlit as st


def render_sidebar(ctx):
    """Render the sidebar with navigation and system status."""
    from sklearn.metrics import accuracy_score

    with st.sidebar:
        st.title("⚡ Nexus Engine")
        st.caption("FOOTBALL AI v9.0")
        st.divider()

        # Navigation — เชื่อม Sidebar กับ session_state
        st.radio("Navigation", [
            "Overview", "Predict Match", "Next Fixtures",
            "Season Table", "Analysis", "Update Data",
        ], key="nav_page", label_visibility="collapsed")

        st.divider()
        st.markdown("### 📊 System Status")
        acc = round(accuracy_score(ctx['y_test'], ctx['y_pred_final']) * 100, 1)
        st.metric("Model Accuracy", f"{acc}%")

        c1, c2 = st.columns(2)
        c1.metric("Hybrid Mode", "ON" if ctx['POISSON_HYBRID_READY'] else "OFF")
        c2.metric("α Value", f"{ctx['best_alpha']:.2f}")

        st.write("")
        st.caption(f"**Features:** `{len(ctx['FEATURES'])}` | **xG:** `{'Active' if ctx['XG_AVAILABLE'] else 'Inactive'}`")
        st.caption(f"**T_Home:** `{ctx['OPT_T_HOME']:.2f}` | **T_Draw:** `{ctx['OPT_T_DRAW']:.2f}`")