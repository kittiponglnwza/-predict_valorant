"""
ui/page_update.py — Data Management page for Football AI Nexus Engine
"""
import os
import glob
from pathlib import Path
import streamlit as st
import pandas as pd

from src.config import DATA_DIR, MODEL_PATH
from src.predict import update_season_csv_from_api
from utils import silent

STABILIZE_REPORT_PATH = Path(DATA_DIR).parent / "artifacts" / "reports" / "stabilize_backtest_report.json"


def page_update(ctx):
    st.title("Data Management")
    st.caption(f"**DATA_DIR:** `{DATA_DIR}`  |  **MODEL_PATH:** `{MODEL_PATH}`")
    st.divider()

    st.subheader("System Update")
    if st.button("☁️ Sync Season 2025 via API", type="primary"):
        with st.spinner("Connecting to API..."):
            df_new = silent(update_season_csv_from_api)
        if df_new is not None:
            st.success(f"Update successful — {len(df_new):,} matches indexed.")
            st.dataframe(df_new.head(10), use_container_width=True)
        else:
            st.error("Failed to fetch update.")

    st.divider()
    st.subheader("STABILIZE Backtest Link")
    st.caption(f"**Report Path:** `{STABILIZE_REPORT_PATH}`")

    if ctx.get('stabilize_connected'):
        st.success("STABILIZE report linked (monitoring only, not overriding inference).")
        sm = ctx.get('stabilize_summary', {})
        if sm:
            c1, c2, c3 = st.columns(3)
            c1.metric("Val Accuracy", f"{sm.get('avg_val_accuracy_after', 0):.3f}")
            c2.metric("Holdout Accuracy", f"{sm.get('final_holdout_accuracy_after', 0):.3f}")
            c3.metric("Holdout Macro-F1", f"{sm.get('final_holdout_macro_f1_after', 0):.3f}")
    else:
        st.warning("STABILIZE report not linked yet.")

    if st.button("Run STABILIZE Backtest (2020-2025)"):
        with st.spinner("Running rolling-origin backtest..."):
            from pipelines.train_pipeline import main as run_stabilize_backtest
            silent(run_stabilize_backtest)
        st.success("STABILIZE backtest complete. Reloading UI context...")
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.subheader("Local Datasets (`/data`)")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if files:
        for f in files:
            try:
                df_tmp = pd.read_csv(f)
                st.markdown(f"- 📄 **{os.path.basename(f)}** — `{len(df_tmp):,}` rows")
            except Exception:
                st.markdown(f"- ⚠️ **{os.path.basename(f)}** — *Unable to read*")
    else:
        st.info("No CSV files found in data directory.")
