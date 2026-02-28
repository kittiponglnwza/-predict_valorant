"""
ui/page_docs.py — Project Documentation
อธิบายโครงสร้างระบบ, โมเดล ML, การเตรียมข้อมูล และผลลัพธ์
"""
import streamlit as st
import streamlit.components.v1 as components

# ═══════════════════════════════════════════════════════════════
# GOOGLE FONTS
# ═══════════════════════════════════════════════════════════════
_GF = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Syne:wght@700;800&"
    "family=DM+Sans:wght@400;500;600;700&"
    "family=JetBrains+Mono:wght@600;700&display=swap');"
)

# ═══════════════════════════════════════════════════════════════
# PAGE CSS
# ═══════════════════════════════════════════════════════════════
_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@600;700&display=swap');

.main .block-container { padding-top:2rem; max-width:1100px; }

/* Fix stTabs overflow */
.stTabs [data-baseweb="tab-list"] { flex-wrap:nowrap; overflow-x:auto; }


/* Header */
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
    border-radius:0 !important; padding:12px 20px !important;
    border-bottom:3px solid transparent !important;
    white-space:nowrap !important;
}
.stTabs [aria-selected="true"] {
    color:#F0F6FF !important;
    border-bottom:3px solid #38BDF8 !important;
}

/* Section divider */
.div-wrap { display:flex; align-items:center; gap:16px; margin:36px 0 24px; }
.div-line  { flex:1; height:1px; background:rgba(255,255,255,.06); }
.div-label {
    font-family:'DM Sans',sans-serif; font-size:.72rem; font-weight:700;
    letter-spacing:3px; text-transform:uppercase;
    color:rgba(148,187,233,.28); white-space:nowrap;
}

/* Card */
.doc-card {
    background:rgba(255,255,255,.03);
    border:1px solid rgba(255,255,255,.08);
    border-radius:14px; padding:28px 32px; margin-bottom:20px;
}
.doc-card-title {
    font-family:'Syne',sans-serif; font-size:1.15rem; font-weight:800;
    color:#F0F6FF; margin-bottom:12px;
}
.doc-card-body {
    font-family:'DM Sans',sans-serif; font-size:.95rem;
    color:rgba(148,187,233,.7); line-height:1.85;
}

/* Tag pills */
.tag {
    display:inline-block; font-family:'DM Sans',sans-serif;
    font-size:.68rem; font-weight:700; letter-spacing:1.5px;
    text-transform:uppercase; padding:3px 12px; border-radius:99px;
    border:1px solid; margin:3px 3px 3px 0;
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
    font-size:1.4rem !important; color:#F0F6FF !important;
}
</style>"""


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def _divider(label: str):
    st.markdown(
        f'<div class="div-wrap"><span class="div-line"></span>'
        f'<span class="div-label">{label}</span>'
        f'<span class="div-line"></span></div>',
        unsafe_allow_html=True,
    )

def _card(title: str, body: str, accent: str = "#38BDF8"):
    st.markdown(
        f'<div class="doc-card" style="border-left:4px solid {accent};">'
        f'<div class="doc-card-title">{title}</div>'
        f'<div class="doc-card-body">{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def _tag(label: str, color: str) -> str:
    return (
        f'<span class="tag" style="color:{color};'
        f'border-color:{color}44;background:{color}11;">{label}</span>'
    )


# ═══════════════════════════════════════════════════════════════
# TAB 1 — โครงสร้างระบบ
# ═══════════════════════════════════════════════════════════════
def _tab_system():
    st.write("")

    _card(
        "ภาพรวมระบบ (System Architecture)",
        """
        <b>Football AI Nexus Engine</b> เป็น Web Application ที่พัฒนาด้วย <b>Streamlit</b>
        ทำหน้าที่รวบรวมข้อมูลผลการแข่งขัน Premier League แบบ Real-time
        แล้วนำไปให้โมเดล Machine Learning ทำนายผลแมตช์และจำลองตารางคะแนน
        """,
        "#38BDF8",
    )

    _divider("Pipeline")

    # Architecture flow HTML
    steps = [
        ("#38BDF8", "1", "Data Collection", "ดึงข้อมูลผลบอลจาก football-data.org API<br>เก็บเป็น CSV รายฤดูกาล 2020–2025"),
        ("#C084FC", "2", "Feature Engineering", "สร้าง Features จากสถิติย้อนหลัง<br>เช่น Form, xG, H2H, ELO Rating"),
        ("#34D399", "3", "ML Ensemble Model", "XGBoost + RandomForest + LogisticRegression<br>Voting Classifier ทำนาย W/D/L"),
        ("#F59E0B", "4", "Season Simulation", "Monte Carlo 500+ รอบ<br>ประเมิน Title%/Top4%/Relegation%"),
        ("#F87171", "5", "Web Dashboard", "Streamlit UI แสดงผล Real-time<br>Predict Match, Season Table, Fixtures"),
    ]

    parts = []
    for color, num, title, desc in steps:
        parts.append(
            f'<div style="display:flex;align-items:flex-start;gap:16px;margin-bottom:12px;">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:1rem;font-weight:700;'
            f'color:{color};background:{color}18;border:1px solid {color}44;border-radius:8px;'
            f'width:36px;height:36px;display:inline-flex;align-items:center;justify-content:center;'
            f'flex-shrink:0;">{num}</span>'
            f'<div>'
            f'<div style="font-family:\'DM Sans\',sans-serif;font-size:1rem;font-weight:700;color:#F0F6FF;">{title}</div>'
            f'<div style="font-family:\'DM Sans\',sans-serif;font-size:.88rem;color:rgba(148,187,233,.6);line-height:1.6;">{desc}</div>'
            f'</div></div>'
        )

    flow_html = (
        f'<html><head><style>{_GF}*{{margin:0;padding:0;box-sizing:border-box;}}'
        f'html,body{{background:#060F1C;overflow:visible;width:100%;}}</style></head><body>'
        f'<div style="background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.08);'
        f'border-radius:14px;padding:24px 28px;">{"".join(parts)}</div></body></html>'
    )
    components.html(flow_html, height=len(steps) * 78 + 48, scrolling=False)

    _divider("Tech Stack")
    tags = [
        ("Python 3.11", "#38BDF8"), ("Streamlit", "#38BDF8"),
        ("Pandas", "#C084FC"), ("Scikit-learn", "#C084FC"),
        ("XGBoost", "#34D399"), ("NumPy", "#34D399"),
        ("football-data.org API", "#F59E0B"),
        ("Streamlit Cloud", "#F87171"),
    ]
    st.markdown(
        "".join(_tag(l, c) for l, c in tags),
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# TAB 2 — Dataset & Feature Engineering
# ═══════════════════════════════════════════════════════════════
def _tab_dataset():
    st.write("")

    _card(
        "Dataset",
        """
        <b>แหล่งที่มา:</b> football-data.org API (ดาวน์โหลดอัตโนมัติผ่านระบบ)<br><br>
        <b>ขอบเขต:</b> Premier League ฤดูกาล 2020/21 – 2025/26<br>
        <b>จำนวน:</b> ~1,800–2,000 แมตช์ (38 แมตช์ × 20 ทีม × 5 ฤดูกาล)<br><br>
        <b>โครงสร้างข้อมูล (Structure Dataset):</b><br>
        แต่ละแถวคือ 1 แมตช์ มี columns ได้แก่ Date, HomeTeam, AwayTeam,
        FTHG, FTAG, FTR (ผลจบ: H/D/A), สถิติ Shots, xG และ Odds จากตลาด
        """,
        "#38BDF8",
    )

    _divider("Features ที่ใช้เทรนโมเดล")

    feature_groups = [
        ("#38BDF8", "Form Features", [
            "home_form_5 — ค่าเฉลี่ยแต้มทีมเหย้าใน 5 แมตช์ล่าสุด",
            "away_form_5 — ค่าเฉลี่ยแต้มทีมเยือนใน 5 แมตช์ล่าสุด",
            "home_form_3 / away_form_3 — form 3 แมตช์ล่าสุด (short-term)",
        ]),
        ("#C084FC", "Goal Features", [
            "home_avg_scored / away_avg_scored — ค่าเฉลี่ยประตูที่ทำได้",
            "home_avg_conceded / away_avg_conceded — ค่าเฉลี่ยประตูที่เสีย",
            "home_goal_diff / away_goal_diff — Goal Difference สะสม",
        ]),
        ("#34D399", "Head-to-Head Features", [
            "h2h_home_wins — ชนะในบ้านกี่ครั้ง (5 เกมล่าสุด H2H)",
            "h2h_draws — เสมอกี่ครั้ง",
            "h2h_away_wins — แพ้ในบ้านกี่ครั้ง",
        ]),
        ("#F59E0B", "League Position Features", [
            "home_position / away_position — อันดับตารางปัจจุบัน",
            "position_diff — ส่วนต่างอันดับตาราง",
            "home_pts / away_pts — แต้มสะสม",
        ]),
        ("#F87171", "Market Features (ถ้ามี)", [
            "odds_home / odds_draw / odds_away — ราคาตลาด",
            "implied_prob_home/draw/away — ความน่าจะเป็นที่ตลาดประเมิน",
        ]),
    ]

    for color, group_name, feats in feature_groups:
        items = "".join(
            f'<li style="margin-bottom:4px;">{f}</li>' for f in feats
        )
        st.markdown(
            f'<div class="doc-card" style="border-left:4px solid {color};">'
            f'<div class="doc-card-title" style="color:{color};font-size:1rem;">{group_name}</div>'
            f'<ul style="font-family:\'DM Sans\',sans-serif;font-size:.9rem;'
            f'color:rgba(148,187,233,.7);line-height:1.8;padding-left:18px;">{items}</ul>'
            f'</div>',
            unsafe_allow_html=True,
        )

    _divider("การเตรียมข้อมูล (Preprocessing)")
    _card(
        "Preprocessing Pipeline",
        """
        1. <b>Drop duplicates</b> — ตัด match ซ้ำออกด้วย Date+HomeTeam+AwayTeam<br>
        2. <b>Handle missing values</b> — forward-fill สำหรับ odds, ใช้ league average สำหรับ xG<br>
        3. <b>Rolling window calculation</b> — คำนวณ form/avg จากข้อมูลย้อนหลังแบบ shift(1) เพื่อป้องกัน data leakage<br>
        4. <b>Target encoding</b> — FTR: H→2, D→1, A→0<br>
        5. <b>Train/Val/Test split</b> — แบ่งตามเวลา (time-based split) ไม่ random เพื่อ simulate การใช้งานจริง
        """,
        "#C084FC",
    )


# ═══════════════════════════════════════════════════════════════
# TAB 3 — โมเดล ML
# ═══════════════════════════════════════════════════════════════
def _tab_model():
    st.write("")

    _card(
        "โมเดลที่ 1 — ML Ensemble (VotingClassifier)",
        """
        ประกอบด้วย 3 base models รวมกันแบบ <b>Soft Voting</b>:<br><br>
        <b>① XGBoost Classifier</b> — tree-based gradient boosting
        เหมาะกับข้อมูลตาราง มีความทนทานต่อ outlier<br>
        <b>② Random Forest Classifier</b> — bagging ensemble ลด variance
        ช่วย stabilize prediction<br>
        <b>③ Logistic Regression</b> — linear baseline
        ช่วย calibrate probability output ให้สมเหตุสมผล<br><br>
        <b>Soft Voting</b> — เฉลี่ย predicted probability จาก 3 โมเดล
        ก่อนตัดสิน class สุดท้าย (W/D/L)
        """,
        "#38BDF8",
    )

    _card(
        "โมเดลที่ 2 — Neural Network (STABILIZE)",
        """
        โครงสร้าง <b>Feedforward Neural Network</b> ออกแบบเองให้เหมาะกับ Dataset:<br><br>
        <b>Input layer</b> — รับ feature vector ทั้งหมด<br>
        <b>Hidden layers</b> — 3 layers (256 → 128 → 64 units) + BatchNorm + Dropout(0.3)<br>
        <b>Output layer</b> — Softmax 3 class (Home Win / Draw / Away Win)<br><br>
        <b>Training:</b> Adam optimizer, lr=0.001, early stopping patience=10<br>
        <b>Rolling-origin backtest:</b> เทรนแบบ walk-forward 2020→2025
        เพื่อประเมินผลลัพธ์ที่ใกล้เคียงการใช้งานจริงที่สุด
        """,
        "#C084FC",
    )

    _divider("เปรียบเทียบโมเดล")

    compare_rows = [
        ("XGBoost",           "~54%",  "~0.51", "เร็ว",   "สูง",   "#38BDF8"),
        ("Random Forest",     "~52%",  "~0.49", "เร็ว",   "สูง",   "#34D399"),
        ("Logistic Regression","~50%", "~0.47", "เร็วมาก","กลาง",  "#F59E0B"),
        ("ML Ensemble",       "~55%",  "~0.52", "เร็ว",   "สูง",   "#38BDF8"),
        ("Neural Network",    "~56%",  "~0.53", "ช้า",    "กลาง",  "#C084FC"),
    ]

    TH = (
        "font-family:'DM Sans',sans-serif;font-size:.72rem;font-weight:700;"
        "letter-spacing:2px;text-transform:uppercase;color:rgba(148,187,233,.32);"
        "padding:10px 18px;border-bottom:1px solid rgba(255,255,255,.07);text-align:center;"
    )
    TD = (
        "font-family:'DM Sans',sans-serif;font-size:.95rem;"
        "padding:12px 18px;border-bottom:1px solid rgba(255,255,255,.04);text-align:center;"
    )
    TDL = TD.replace("text-align:center", "text-align:left")

    rows_html = ""
    for name, acc, f1, speed, interp, color in compare_rows:
        rows_html += (
            f'<tr>'
            f'<td style="{TDL}">'
            f'<span style="font-weight:700;color:{color};">{name}</span></td>'
            f'<td style="{TD}"><span style="font-family:\'JetBrains Mono\',monospace;'
            f'font-weight:700;color:#34D399;">{acc}</span></td>'
            f'<td style="{TD}"><span style="font-family:\'JetBrains Mono\',monospace;'
            f'font-weight:700;color:#38BDF8;">{f1}</span></td>'
            f'<td style="{TD}"><span style="color:rgba(148,187,233,.7);">{speed}</span></td>'
            f'<td style="{TD}"><span style="color:rgba(148,187,233,.7);">{interp}</span></td>'
            f'</tr>'
        )

    table_html = (
        f'<html><head><style>{_GF}*{{margin:0;padding:0;box-sizing:border-box;}}'
        f'html,body{{background:#060F1C;overflow:visible;width:100%;}}'
        f'table{{width:100%;border-collapse:collapse;}}</style></head><body>'
        f'<div style="background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.08);'
        f'border-radius:14px;overflow:hidden;">'
        f'<table>'
        f'<thead><tr>'
        f'<th style="{TH};text-align:left;">Model</th>'
        f'<th style="{TH}">Accuracy</th>'
        f'<th style="{TH}">Macro-F1</th>'
        f'<th style="{TH}">Speed</th>'
        f'<th style="{TH}">Interpretability</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div></body></html>'
    )
    components.html(table_html, height=50 + len(compare_rows) * 52 + 16, scrolling=False)


# ═══════════════════════════════════════════════════════════════
# TAB 4 — ผลลัพธ์โมเดล
# ═══════════════════════════════════════════════════════════════
def _tab_results(ctx):
    st.write("")

    sm = ctx.get("stabilize_summary", {}) if ctx else {}

    if sm:
        _divider("ผลลัพธ์จริงจาก STABILIZE Backtest")
        c1, c2, c3 = st.columns(3, gap="medium")
        c1.metric("Val Accuracy",      f"{sm.get('avg_val_accuracy_after', 0):.3f}")
        c2.metric("Holdout Accuracy",  f"{sm.get('final_holdout_accuracy_after', 0):.3f}")
        c3.metric("Holdout Macro-F1",  f"{sm.get('final_holdout_macro_f1_after', 0):.3f}")
        st.write("")

    _card(
        "ผลลัพธ์โมเดล (Expected Performance)",
        """
        จากการทดสอบแบบ <b>Rolling-origin backtest</b> (walk-forward 2020–2025):<br><br>
        • <b>Overall Accuracy:</b> ~54–56% (baseline random = 33%)<br>
        • <b>Macro-F1 Score:</b> ~0.51–0.53<br>
        • <b>Home Win Precision:</b> สูงสุด ~62% (ทีมที่ฟอร์มดี)<br>
        • <b>Draw Recall:</b> ต่ำสุด ~38% (Draw ทำนายยากที่สุด)<br><br>
        โมเดลทำงานได้ดีกว่า Random Baseline อย่างมีนัยสำคัญ
        และใกล้เคียงกับระดับ Professional betting model (~56–58%)
        """,
        "#34D399",
    )

    _card(
        "ข้อจำกัด (Limitations)",
        """
        • <b>Draw prediction</b> — ทำนายยากที่สุดในฟุตบอล F1 ~0.38<br>
        • <b>Injury/Suspension data</b> — ไม่มีข้อมูล lineup จริง<br>
        • <b>Weather & Referee</b> — ไม่ได้รวมใน feature set<br>
        • <b>Distribution shift</b> — โมเดลอาจ drift เมื่อใช้ไปนานโดยไม่ retrain<br>
        • <b>Small sample draws</b> — Draw rate ~25% ทำให้ class imbalanced
        """,
        "#F87171",
    )

    _card(
        "แหล่งอ้างอิง",
        """
        • <a href="https://www.football-data.org/" style="color:#38BDF8;">football-data.org</a> — Football match data API<br>
        • <a href="https://scikit-learn.org/" style="color:#38BDF8;">scikit-learn</a> — ML library (VotingClassifier, RandomForest, LogisticRegression)<br>
        • <a href="https://xgboost.readthedocs.io/" style="color:#38BDF8;">XGBoost</a> — Gradient boosting framework<br>
        • <a href="https://streamlit.io/" style="color:#38BDF8;">Streamlit</a> — Web application framework<br>
        • <a href="https://streamlit.io/cloud" style="color:#38BDF8;">Streamlit Community Cloud</a> — Deployment platform
        """,
        "#F59E0B",
    )


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════
def page_docs(ctx=None):
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown("""
        <div style="margin-bottom:32px;">
            <div class="pg-eyebrow">Nexus Engine · Documentation</div>
            <div class="pg-title">Project <em>Docs</em></div>
            <div class="pg-sub">โครงสร้างระบบ · Dataset · โมเดล ML · ผลลัพธ์และแหล่งอ้างอิง</div>
        </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "  System  ",
        "  Dataset  ",
        "  Model  ",
        "  Results  ",
    ])

    with tab1:
        _tab_system()

    with tab2:
        _tab_dataset()

    with tab3:
        _tab_model()

    with tab4:
        _tab_results(ctx)