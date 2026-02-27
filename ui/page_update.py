"""
ui/page_update.py — Data Management page for Football AI Nexus Engine
"""
import os
import glob
import streamlit as st
import pandas as pd

from src.config import DATA_DIR, MODEL_PATH
from src.predict import update_season_csv_from_api
from utils import silent


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