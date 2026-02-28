"""
ui/page_model_test.py — Model Testing & Evaluation
แสดงผลทดสอบโมเดลทั้ง 2 ประเภท: ML Ensemble และ Neural Network
"""
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd

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

.main .block-container { padding-top:1.5rem; max-width:1100px; }

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
    font-size:1.5rem !important; color:#F0F6FF !important;
}
[data-testid="stMetricDelta"] {
    font-family:'DM Sans',sans-serif !important; font-size:.85rem !important;
}

/* Section divider */
.div-wrap { display:flex; align-items:center; gap:16px; margin:32px 0 20px; }
.div-line  { flex:1; height:1px; background:rgba(255,255,255,.06); }
.div-label {
    font-family:'DM Sans',sans-serif; font-size:.72rem; font-weight:700;
    letter-spacing:3px; text-transform:uppercase;
    color:rgba(148,187,233,.28); white-space:nowrap;
}

/* Not-ready box */
.not-ready {
    text-align:center; padding:4rem 2rem;
    background:linear-gradient(135deg,rgba(56,189,248,.03),rgba(192,132,252,.03));
    border:1px dashed rgba(56,189,248,.15); border-radius:16px; margin-top:1rem;
    font-family:'DM Sans',sans-serif; font-size:1rem; color:rgba(148,187,233,.4);
}

/* Spinner */
.stSpinner > div { border-top-color:#38BDF8 !important; }
</style>"""


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
_LABELS = {0: "Away Win", 1: "Draw", 2: "Home Win"}
_COLORS = {"Home Win": "#34D399", "Draw": "#F59E0B", "Away Win": "#F87171"}


def _divider(label: str):
    st.markdown(
        f'<div class="div-wrap"><span class="div-line"></span>'
        f'<span class="div-label">{label}</span>'
        f'<span class="div-line"></span></div>',
        unsafe_allow_html=True,
    )


def _safe_metrics(y_true, y_pred):
    """Compute accuracy, per-class precision/recall/f1, macro-f1."""
    from collections import defaultdict
    classes = sorted(set(y_true) | set(y_pred))
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    correct = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct += 1
            tp[yt] += 1
        else:
            fp[yp] += 1
            fn[yt] += 1

    accuracy = correct / len(y_true) if y_true else 0
    per_class = {}
    for c in classes:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0
        rec  = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1,
                        "support": tp[c] + fn[c]}
    macro_f1 = np.mean([v["f1"] for v in per_class.values()]) if per_class else 0
    return accuracy, macro_f1, per_class


def _confusion_matrix(y_true, y_pred, classes):
    """Return confusion matrix as 2D list."""
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = [[0] * n for _ in range(n)]
    for yt, yp in zip(y_true, y_pred):
        if yt in idx and yp in idx:
            cm[idx[yt]][idx[yp]] += 1
    return cm


# ═══════════════════════════════════════════════════════════════
# CONFUSION MATRIX HTML
# ═══════════════════════════════════════════════════════════════
def _cm_html(cm, classes, title: str, accent: str) -> str:
    labels = [_LABELS.get(c, str(c)) for c in classes]
    n = len(classes)
    max_val = max(cm[r][c] for r in range(n) for c in range(n)) or 1

    # header row
    th_style = (
        "font-family:'DM Sans',sans-serif;font-size:.72rem;font-weight:700;"
        "letter-spacing:2px;text-transform:uppercase;color:rgba(148,187,233,.35);"
        "padding:10px 14px;text-align:center;"
    )
    td_label = (
        "font-family:'DM Sans',sans-serif;font-size:.85rem;font-weight:700;"
        "color:rgba(148,187,233,.55);padding:10px 16px;text-align:right;white-space:nowrap;"
    )

    header = (
        f'<tr><th style="{th_style}">Actual \\ Pred</th>'
        + "".join(f'<th style="{th_style}">{l}</th>' for l in labels)
        + "</tr>"
    )

    # Pre-defined rgba palettes — diagonal=green, off-diagonal=red
    _DIAG_BG   = "rgba(52,211,153,{op})"    # #34D399 green
    _DIAG_BDR  = "rgba(52,211,153,0.5)"
    _DIAG_TXT  = "#34D399"
    _OFF_BG    = "rgba(248,113,113,{op})"   # #F87171 red
    _OFF_BDR   = "rgba(248,113,113,0.35)"
    _OFF_TXT   = "#F87171"

    rows_html = ""
    for r in range(n):
        cells = f'<td style="{td_label}">{labels[r]}</td>'
        for c in range(n):
            val     = cm[r][c]
            op      = max(0.12, val / max_val * 0.85)   # 0.12–0.85
            is_diag = (r == c)
            if is_diag:
                bg_css  = _DIAG_BG.format(op=f"{op:.2f}")
                bdr_css = _DIAG_BDR
                txt_css = _DIAG_TXT
            else:
                bg_css  = _OFF_BG.format(op=f"{op:.2f}")
                bdr_css = _OFF_BDR
                txt_css = _OFF_TXT
            cells += (
                f'<td style="padding:10px 14px;text-align:center;">'
                f'<span style="display:inline-flex;align-items:center;justify-content:center;'
                f'min-width:60px;height:40px;padding:0 12px;border-radius:8px;'
                f'background:{bg_css};border:1px solid {bdr_css};'
                f'font-family:\'JetBrains Mono\',monospace;font-size:1rem;font-weight:700;'
                f'color:{txt_css};">{val}</span></td>'
            )
        rows_html += f'<tr style="border-bottom:1px solid rgba(255,255,255,.04);">{cells}</tr>'

    html = (
        f'<html><head><style>{_GF}'
        f'*{{margin:0;padding:0;box-sizing:border-box;}}'
        f'html,body{{background:#060F1C;overflow:hidden;width:100%;}}'
        f'table{{width:100%;border-collapse:collapse;}}</style></head><body>'
        f'<div style="background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.08);'
        f'border-radius:14px;overflow:hidden;">'
        f'<div style="padding:16px 20px 8px;font-family:\'Syne\',sans-serif;font-size:1rem;'
        f'font-weight:800;color:{accent};">{title}</div>'
        f'<table><thead>{header}</thead><tbody>{rows_html}</tbody></table>'
        f'</div></body></html>'
    )
    h = 60 + 50 + (n + 1) * 60
    return html, h


# ═══════════════════════════════════════════════════════════════
# PER-CLASS METRICS TABLE HTML
# ═══════════════════════════════════════════════════════════════
def _metrics_table_html(per_class: dict, title: str, accent: str) -> str:
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
    for cls, m in sorted(per_class.items()):
        lbl   = _LABELS.get(cls, str(cls))
        col   = _COLORS.get(lbl, "#38BDF8")
        prec  = m["precision"]
        rec   = m["recall"]
        f1    = m["f1"]
        sup   = m["support"]

        def _bar(v, color):
            w = max(3, int(v * 60))
            return (
                f'<span style="display:inline-flex;align-items:center;gap:6px;">'
                f'<span style="display:inline-block;width:{w}px;height:5px;border-radius:99px;'
                f'background:{color};opacity:.8;"></span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-weight:700;'
                f'color:{color};">{v:.3f}</span></span>'
            )

        rows_html += (
            f'<tr>'
            f'<td style="{TDL}"><span style="font-weight:700;color:{col};">{lbl}</span></td>'
            f'<td style="{TD}">{_bar(prec, "#38BDF8")}</td>'
            f'<td style="{TD}">{_bar(rec,  "#34D399")}</td>'
            f'<td style="{TD}">{_bar(f1,   accent)}</td>'
            f'<td style="{TD}"><span style="font-family:\'JetBrains Mono\',monospace;'
            f'color:rgba(148,187,233,.5);">{sup}</span></td>'
            f'</tr>'
        )

    html = (
        f'<html><head><style>{_GF}'
        f'*{{margin:0;padding:0;box-sizing:border-box;}}'
        f'html,body{{background:#060F1C;overflow:hidden;width:100%;}}'
        f'table{{width:100%;border-collapse:collapse;}}</style></head><body>'
        f'<div style="background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.08);'
        f'border-radius:14px;overflow:hidden;">'
        f'<div style="padding:16px 20px 8px;font-family:\'Syne\',sans-serif;font-size:1rem;'
        f'font-weight:800;color:{accent};">{title}</div>'
        f'<table>'
        f'<thead><tr>'
        f'<th style="{TH};text-align:left;">Class</th>'
        f'<th style="{TH}">Precision</th>'
        f'<th style="{TH}">Recall</th>'
        f'<th style="{TH}">F1-Score</th>'
        f'<th style="{TH}">Support</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody></table>'
        f'</div></body></html>'
    )
    h = 60 + 50 + len(per_class) * 52
    return html, h


# ═══════════════════════════════════════════════════════════════
# RENDER ONE MODEL'S FULL EVALUATION
# ═══════════════════════════════════════════════════════════════
def _render_evaluation(y_true, y_pred, proba, model_name: str, accent: str):
    classes = sorted(set(y_true))
    accuracy, macro_f1, per_class = _safe_metrics(list(y_true), list(y_pred))
    baseline = 1 / len(classes) if classes else 0.333

    # ── Key metrics ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.metric("Accuracy",    f"{accuracy:.3f}", f"+{accuracy - baseline:.3f} vs random")
    c2.metric("Macro F1",    f"{macro_f1:.3f}")
    c3.metric("Test Samples", f"{len(y_true):,}")
    c4.metric("Classes",     str(len(classes)))

    st.write("")

    # ── Confusion matrix ─────────────────────────────────────────
    _divider("Confusion Matrix")
    cm = _confusion_matrix(list(y_true), list(y_pred), classes)
    cm_html, cm_h = _cm_html(cm, classes, f"{model_name} — Confusion Matrix", accent)
    components.html(cm_html, height=cm_h, scrolling=False)

    # ── Per-class metrics ─────────────────────────────────────────
    _divider("Per-Class Metrics")
    mt_html, mt_h = _metrics_table_html(per_class, f"{model_name} — Classification Report", accent)
    components.html(mt_html, height=mt_h, scrolling=False)

    # ── Probability distribution ──────────────────────────────────
    if proba is not None:
        _divider("Predicted Probability Distribution")
        cols = st.columns(3, gap="medium")
        class_names = ["Away Win", "Draw", "Home Win"]
        prob_colors = ["#F87171", "#F59E0B", "#34D399"]
        for i, (cname, ccolor) in enumerate(zip(class_names, prob_colors)):
            if i < proba.shape[1]:
                p = proba[:, i]
                cols[i].metric(
                    f"{cname} — Mean Prob",
                    f"{p.mean():.3f}",
                    f"std ±{p.std():.3f}",
                )


# ═══════════════════════════════════════════════════════════════
# TAB 1 — ML Ensemble
# ═══════════════════════════════════════════════════════════════
def _tab_ensemble(ctx):
    st.write("")

    y_test = ctx.get("y_test")
    y_pred = ctx.get("y_pred_final")
    proba  = ctx.get("proba_hybrid")

    if y_test is None or y_pred is None:
        st.markdown(
            '<div class="not-ready">⚠️ ไม่พบข้อมูล y_test / y_pred_final ใน ctx<br>'
            'กรุณา Load หรือ Train model ก่อน</div>',
            unsafe_allow_html=True,
        )
        return

    y_true_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)

    _render_evaluation(y_true_arr, y_pred_arr, proba, "ML Ensemble", "#38BDF8")

    # ── Model components info ─────────────────────────────────────
    _divider("Model Components")
    mlp_ready   = ctx.get("MLP_MODEL_READY",  False)
    mlp2_ready  = ctx.get("MLP2_MODEL_READY", False)
    mlp_w       = ctx.get("mlp_blend_weight",  0.0)
    mlp2_w      = ctx.get("mlp2_blend_weight", 0.0)

    components_data = [
        ("XGBoost",            "#38BDF8", "✅ Active", f"{(1 - mlp_w - mlp2_w)*100:.0f}% weight"),
        ("Random Forest",      "#34D399", "✅ Active", "Stage-2 Calibration"),
        ("Logistic Regression","#F59E0B", "✅ Active", "Stage-1 Calibration"),
        ("MLP (Neural Net 1)", "#C084FC",
            "✅ Active" if mlp_ready else "⚠️ Not loaded",
            f"Blend weight: {mlp_w:.2f}" if mlp_ready else "—"),
        ("MLP (Neural Net 2)", "#F87171",
            "✅ Active" if mlp2_ready else "⚠️ Not loaded",
            f"Blend weight: {mlp2_w:.2f}" if mlp2_ready else "—"),
    ]

    TH = (
        "font-family:'DM Sans',sans-serif;font-size:.72rem;font-weight:700;"
        "letter-spacing:2px;text-transform:uppercase;color:rgba(148,187,233,.32);"
        "padding:10px 18px;border-bottom:1px solid rgba(255,255,255,.07);"
    )
    rows = ""
    for name, col, status, note in components_data:
        rows += (
            f'<tr style="border-bottom:1px solid rgba(255,255,255,.04);">'
            f'<td style="padding:12px 18px;font-family:\'DM Sans\',sans-serif;'
            f'font-weight:700;color:{col};">{name}</td>'
            f'<td style="padding:12px 18px;text-align:center;font-family:\'DM Sans\',sans-serif;'
            f'font-size:.9rem;color:rgba(148,187,233,.7);">{status}</td>'
            f'<td style="padding:12px 18px;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:.85rem;color:rgba(148,187,233,.5);">{note}</td>'
            f'</tr>'
        )
    tbl = (
        f'<html><head><style>{_GF}*{{margin:0;padding:0;box-sizing:border-box;}}'
        f'html,body{{background:#060F1C;overflow:hidden;width:100%;}}'
        f'table{{width:100%;border-collapse:collapse;}}</style></head><body>'
        f'<div style="background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.08);'
        f'border-radius:14px;overflow:hidden;">'
        f'<table><thead><tr>'
        f'<th style="{TH};text-align:left;">Component</th>'
        f'<th style="{TH};text-align:center;">Status</th>'
        f'<th style="{TH};text-align:left;">Note</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
        f'</div></body></html>'
    )
    components.html(tbl, height=60 + len(components_data) * 52, scrolling=False)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — Neural Network
# ═══════════════════════════════════════════════════════════════
def _tab_neural(ctx):
    st.write("")

    mlp_ready  = ctx.get("MLP_MODEL_READY",  False)
    mlp2_ready = ctx.get("MLP2_MODEL_READY", False)

    if not mlp_ready and not mlp2_ready:
        st.markdown(
            '<div class="not-ready">'
            '🧠 Neural Network ยังไม่ถูก load<br><br>'
            'MLP_MODEL_READY = False · MLP2_MODEL_READY = False<br><br>'
            'ลอง Run STABILIZE Backtest ในหน้า Update Data</div>',
            unsafe_allow_html=True,
        )
        return

    y_test = ctx.get("y_test")
    proba  = ctx.get("proba_hybrid")   # hybrid already blends MLP

    if y_test is None or proba is None:
        st.warning("ไม่พบ y_test หรือ proba ใน ctx")
        return

    y_true_arr = np.array(y_test)

    # Derive MLP-only prediction from hybrid proba
    # (best approximation without separate MLP proba stored in ctx)
    y_pred_hybrid = np.argmax(proba, axis=1)

    _render_evaluation(y_true_arr, y_pred_hybrid, proba, "Neural Network (Blended)", "#C084FC")

    _divider("Neural Network Architecture")
    arch_items = [
        ("#C084FC", "Input Layer",   "Feature vector — ขนาดตาม FEATURES list"),
        ("#38BDF8", "Hidden Layer 1","256 units · BatchNorm · ReLU · Dropout(0.3)"),
        ("#38BDF8", "Hidden Layer 2","128 units · BatchNorm · ReLU · Dropout(0.3)"),
        ("#38BDF8", "Hidden Layer 3","64 units  · BatchNorm · ReLU · Dropout(0.2)"),
        ("#34D399", "Output Layer",  "3 units · Softmax → P(Home Win), P(Draw), P(Away Win)"),
    ]
    parts = []
    for color, layer, desc in arch_items:
        parts.append(
            f'<div style="display:flex;align-items:center;gap:14px;'
            f'padding:10px 16px;border-bottom:1px solid rgba(255,255,255,.04);">'
            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:.78rem;font-weight:800;'
            f'color:{color};background:{color}18;border:1px solid {color}33;'
            f'border-radius:6px;padding:3px 10px;white-space:nowrap;flex-shrink:0;">{layer}</span>'
            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:.9rem;'
            f'color:rgba(148,187,233,.65);">{desc}</span>'
            f'</div>'
        )
    arch_html = (
        f'<html><head><style>{_GF}*{{margin:0;padding:0;box-sizing:border-box;}}'
        f'html,body{{background:#060F1C;overflow:hidden;width:100%;}}</style></head><body>'
        f'<div style="background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.08);'
        f'border-radius:14px;overflow:hidden;">{"".join(parts)}</div></body></html>'
    )
    components.html(arch_html, height=len(arch_items) * 52 + 8, scrolling=False)

    # blend weights
    _divider("Blend Weights")
    mlp_w  = ctx.get("mlp_blend_weight",  0.0)
    mlp2_w = ctx.get("mlp2_blend_weight", 0.0)
    ens_w  = max(0.0, 1.0 - mlp_w - mlp2_w)

    c1, c2, c3 = st.columns(3, gap="medium")
    c1.metric("Ensemble Weight", f"{ens_w:.2f}")
    c2.metric("MLP-1 Weight",    f"{mlp_w:.2f}")
    c3.metric("MLP-2 Weight",    f"{mlp2_w:.2f}")


# ═══════════════════════════════════════════════════════════════
# TAB 3 — เปรียบเทียบ
# ═══════════════════════════════════════════════════════════════
def _tab_compare(ctx):
    st.write("")

    y_test = ctx.get("y_test")
    y_pred = ctx.get("y_pred_final")
    proba  = ctx.get("proba_hybrid")
    p2s    = ctx.get("proba_2stage")

    if y_test is None or y_pred is None:
        st.markdown(
            '<div class="not-ready">⚠️ ไม่พบข้อมูลใน ctx</div>',
            unsafe_allow_html=True,
        )
        return

    y_true = np.array(y_test)

    # Ensemble (threshold-applied)
    acc_ens, f1_ens, pc_ens = _safe_metrics(list(y_true), list(y_pred))

    # Hybrid proba argmax (raw, no threshold)
    y_hybrid = np.argmax(proba, axis=1) if proba is not None else y_pred
    acc_hyb, f1_hyb, _ = _safe_metrics(list(y_true), list(y_hybrid))

    # 2-stage raw
    y_2s = np.argmax(p2s, axis=1) if p2s is not None else y_pred
    acc_2s, f1_2s, _ = _safe_metrics(list(y_true), list(y_2s))

    # ── Comparison table ─────────────────────────────────────────
    models = [
        ("ML Ensemble (threshold)", "#38BDF8", acc_ens, f1_ens),
        ("Hybrid Proba (argmax)",   "#C084FC", acc_hyb, f1_hyb),
        ("2-Stage Raw (argmax)",    "#34D399", acc_2s,  f1_2s),
    ]

    TH = (
        "font-family:'DM Sans',sans-serif;font-size:.72rem;font-weight:700;"
        "letter-spacing:2px;text-transform:uppercase;color:rgba(148,187,233,.32);"
        "padding:10px 18px;border-bottom:1px solid rgba(255,255,255,.07);"
    )
    TD = "font-family:'DM Sans',sans-serif;font-size:.95rem;padding:14px 18px;"

    best_acc = max(m[2] for m in models)
    best_f1  = max(m[3] for m in models)

    rows = ""
    for name, color, acc, f1 in models:
        star_a = " ★" if acc == best_acc else ""
        star_f = " ★" if f1 == best_f1  else ""
        rows += (
            f'<tr style="border-bottom:1px solid rgba(255,255,255,.04);">'
            f'<td style="{TD};text-align:left;font-weight:700;color:{color};">{name}</td>'
            f'<td style="{TD};text-align:center;">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:1.1rem;'
            f'font-weight:700;color:{"#34D399" if star_a else "#F0F6FF"};">'
            f'{acc:.4f}{star_a}</span></td>'
            f'<td style="{TD};text-align:center;">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:1.1rem;'
            f'font-weight:700;color:{"#34D399" if star_f else "#F0F6FF"};">'
            f'{f1:.4f}{star_f}</span></td>'
            f'</tr>'
        )
    tbl_html = (
        f'<html><head><style>{_GF}*{{margin:0;padding:0;box-sizing:border-box;}}'
        f'html,body{{background:#060F1C;overflow:hidden;width:100%;}}'
        f'table{{width:100%;border-collapse:collapse;}}</style></head><body>'
        f'<div style="background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.08);'
        f'border-radius:14px;overflow:hidden;">'
        f'<table><thead><tr>'
        f'<th style="{TH};text-align:left;">Model</th>'
        f'<th style="{TH};text-align:center;">Accuracy</th>'
        f'<th style="{TH};text-align:center;">Macro F1</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
        f'</div></body></html>'
    )
    components.html(tbl_html, height=60 + len(models) * 56, scrolling=False)

    st.markdown(
        '<div style="font-family:\'DM Sans\',sans-serif;font-size:.8rem;'
        'color:rgba(148,187,233,.3);margin-top:10px;">★ = Best in category</div>',
        unsafe_allow_html=True,
    )

    # ── Per-class comparison ──────────────────────────────────────
    _divider("Per-Class F1 — ML Ensemble")
    mt_html, mt_h = _metrics_table_html(pc_ens, "ML Ensemble — Per-Class Report", "#38BDF8")
    components.html(mt_html, height=mt_h, scrolling=False)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════
def page_model_test(ctx=None):
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown("""
        <div style="margin-bottom:32px;">
            <div class="pg-eyebrow">⚡ Nexus Engine · Evaluation</div>
            <div class="pg-title">Model <em>Testing</em></div>
            <div class="pg-sub">ผลทดสอบโมเดล · Confusion Matrix · Classification Report · Model Comparison</div>
        </div>
    """, unsafe_allow_html=True)

    if ctx is None:
        st.markdown(
            '<div class="not-ready">⚠️ ไม่พบ ctx — กรุณา Load model ก่อน</div>',
            unsafe_allow_html=True,
        )
        return

    tab1, tab2, tab3 = st.tabs([
        "  🤖 ML Ensemble  ",
        "  🧠 Neural Network  ",
        "  ⚖️ Compare  ",
    ])

    with tab1:
        _tab_ensemble(ctx)

    with tab2:
        _tab_neural(ctx)

    with tab3:
        _tab_compare(ctx)