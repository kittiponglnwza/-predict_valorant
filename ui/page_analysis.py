"""
ui/page_analysis.py — Model Analysis & Insights page for Football AI Nexus Engine
"""
import streamlit as st
import pandas as pd

from src.analysis import run_monte_carlo, backtest_roi, analyze_draw_calibration, run_feature_importance
from utils import silent


def page_analysis(ctx):
    st.title("Model Analysis & Insights")
    st.write("")

    t1, t2, t3, t4 = st.tabs([
        "🎲 Monte Carlo", "💰 ROI Backtest", "⚖️ Calibration", "🧠 Feature Importance"
    ])

    with t1:
        st.subheader("Monte Carlo Simulation")
        n_sim = st.slider("Number of Simulations", 100, 2000, 500, 100)
        if st.button("Run Monte Carlo", type="primary", key="btn_mc"):
            with st.spinner(f"Simulating {n_sim:,} paths..."):
                mc = silent(run_monte_carlo, ctx, n_simulations=n_sim)
            if mc:
                df = pd.DataFrame(mc).T.sort_values('expected_pts', ascending=False)
                st.dataframe(df, use_container_width=True)

    with t2:
        st.subheader("Betting ROI Backtest")
        c1, c2 = st.columns(2)
        me = c1.slider("Minimum Edge (%)", 1, 10, 3) / 100
        kf = c2.slider("Kelly Fraction (%)", 5, 30, 15) / 100

        if st.button("Run Backtest", type="primary", key="btn_roi"):
            with st.spinner("Running historical backtest..."):
                roi = silent(backtest_roi, ctx, min_edge=me, kelly_fraction=kf)
            if roi:
                st.write("")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("ROI", f"{roi['roi']:+.1f}%")
                r2.metric("Win Rate", f"{roi['win_rate']:.1f}%")
                r3.metric("Net P&L", f"£{roi['net_pnl']:+,.0f}")
                r4.metric("Max Drawdown", f"{roi['max_dd']:.1f}%")

    with t3:
        st.subheader("Draw Probability Calibration")
        if st.button("Analyze Calibration", type="primary", key="btn_cal"):
            with st.spinner("Analyzing draw rates..."):
                cal = silent(analyze_draw_calibration, ctx)
            if cal:
                st.write("")
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted Draw Rate", f"{cal['predicted_rate']:.1%}")
                c2.metric("Actual Draw Rate", f"{cal['actual_rate']:.1%}")
                c3.metric("Bias", f"{cal['bias']:+.1f}%", delta_color="inverse")

    with t4:
        st.subheader("Feature Importance (Top 20)")
        if st.button("Show Importance", type="primary", key="btn_fi"):
            with st.spinner("Calculating tree weights..."):
                import io, contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    run_feature_importance(ctx, max_display=20)
                st.code(buf.getvalue(), language="text")