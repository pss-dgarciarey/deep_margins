# ================================================================
# app.py — Deep Margins Dashboard (with login + theme integration)
# ================================================================

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------
# Page + Theme Config
# ------------------------------------------------------------
st.set_page_config(page_title="Deep Margins", page_icon="📊", layout="wide")

THEMES = {
    "Corporate": {"bg":"#e6e9ee","card":"#ffffff","text":"#0f172a","muted":"#6b7280","accent":"#2f6feb","template":"plotly_white",
        "font_import":"https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap",
        "font_family":"'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial"},
    "Dark": {"bg":"#0f172a","card":"#111827","text":"#e5e7eb","muted":"#94a3b8","accent":"#84ccff","template":"plotly_dark",
        "font_import":"https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&display=swap",
        "font_family":"'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial"},
    "Ocean": {"bg":"#e6f7fb","card":"#ffffff","text":"#0b2530","muted":"#517a88","accent":"#0ea5b7","template":"plotly_white",
        "font_import":"https://fonts.googleapis.com/css2?family=Rubik:wght@400;600;700&display=swap",
        "font_family":"'Rubik', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial"},
}

def apply_theme(name: str):
    t = THEMES[name]
    st.markdown(f"""
    <style>
    @import url('{t["font_import"]}');
    html, body, .stApp {{
        background-color: {t["bg"]} !important;
        color: {t["text"]} !important;
        font-family: {t["font_family"]};
    }}
    .accent-btn > button {{
        background: {t["accent"]} !important;
        color:white !important;
        border:none; border-radius:10px; font-weight:700;
        box-shadow: 0 6px 20px rgba(0,0,0,.25);
    }}
    </style>
    """, unsafe_allow_html=True)
    px.defaults.template = t["template"]

# ------------------------------------------------------------
# Login System
# ------------------------------------------------------------
USERS = {
    st.secrets["USER1"]: st.secrets["USER1_PWD"],
    st.secrets["USER2"]: st.secrets["USER2_PWD"],
    st.secrets["USER4"]: st.secrets["USER4_PWD"],
    st.secrets["USER5"]: st.secrets["USER5_PWD"],
    st.secrets["USER6"]: st.secrets["USER6_PWD"],
    st.secrets["USER7"]: st.secrets["USER7_PWD"],
    st.secrets["USER8"]: st.secrets["USER8_PWD"],
}
USER_NAMES = {
    "d.garcia":"Daniel Garcia Rey","t.held":"Thomas Held","b.arrieta":"Borja Arrieta",
    "c.bahn":"Cristoph Bahn","tgv":"Tomas Garcia Villanueva",
    "f.wittfeld":"Florian Wittfeld","m.peter":"Michel Peter"
}

if "auth" not in st.session_state: st.session_state["auth"] = False
if "theme" not in st.session_state: st.session_state["theme"] = "Corporate"

if not st.session_state["auth"]:
    st.title("🔐 Deep Margins Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login", use_container_width=True):
        if u in USERS and p == USERS[u]:
            st.session_state["auth"] = True
            st.session_state["user"] = u
            st.success(f"Welcome {USER_NAMES.get(u,u).split()[0]}!")
        else:
            st.error("Invalid credentials.")
    st.stop()

# Sidebar theme selector
st.sidebar.title("🎨 Theme")
sel_theme = st.sidebar.selectbox("Choose Theme", list(THEMES.keys()),
                                 index=list(THEMES.keys()).index(st.session_state["theme"]))
apply_theme(sel_theme)
st.session_state["theme"] = sel_theme
TEMPLATE = THEMES[sel_theme]["template"]

# ------------------------------------------------------------
# Page Title
# ------------------------------------------------------------
st.title("📊 Deep Margins — PSS Analytics Dashboard")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
SERVICE_PRETTY = {
    "tpm": "TPM",
    "cpm": "CPM",
    "eng": "Engineering",
    "qa_qc_exp": "QA/QC/Exp",
    "hse": "HSE",
    "constr": "Construction",
    "com": "Commissioning",
    "man": "Manufacturing",
    "proc": "Procurement",
}
SERVICE_BLOCKS = ["tpm", "cpm", "eng", "qa_qc_exp", "hse", "constr", "com", "man", "proc"]

# Exclude noisy flags from Drivers
EXCLUDED_DRIVER_FLAGS = {
    # budget_overrun
    "qa_qc_exp_b_o_flag", "cpm_b_o_flag", "tpm_b_o_flag", "hse_b_o_flag",
    # hours_overrun
    "proc_h_o_flag", "hse_h_o_flag", "man_h_o_flag",
    # delay
    "qa_qc_exp_delay_flag", "cpm_delay_flag", "tpm_delay_flag",
    "hse_delay_flag", "constr_delay_flag",
}

def normalize(c: str) -> str:
    c = str(c).strip().replace(" ", "_").replace("/", "_").replace("-", "_").replace("%", "pct")
    while "__" in c:
        c = c.replace("__", "_")
    return c.lower()

def safe_num(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def safe_int(v):
    try:
        return int(float(v))
    except Exception:
        return 0

def plotly_config(name: str):
    return {
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": f"{name}",
            "height": 1350,
            "width": 2400,
            "scale": 2,
        },
    }

def is_binary_series(s: pd.Series) -> bool:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return False
    uniq = pd.unique(s.astype(float))
    return np.all(np.isin(uniq, [0.0, 1.0]))

def to_flag(s: pd.Series, op: str, thr: float):
    s = pd.to_numeric(s, errors="coerce")
    if op == "≥": return (s >= thr).astype(int)
    if op == ">": return (s >  thr).astype(int)
    if op == "≤": return (s <= thr).astype(int)
    if op == "<": return (s <  thr).astype(int)
    if op == "=": return (s == thr).astype(int)
    return (s > thr).astype(int)

def humanize_col(c: str) -> str:
    s = c.lower()
    for k,v in SERVICE_PRETTY.items():
        s = s.replace(f"{k}_", f"{v} ")
    s = (s.replace("b_o", "budget overrun")
           .replace("h_o", "hours overrun")
           .replace("delay", "delay")
           .replace("cm2pct", "CM2%")
           .replace("cm2_", "CM2 "))
    s = s.replace("_", " ").strip()
    return s[:1].upper() + s[1:]

def find_col(df: pd.DataFrame, must_contain):
    tokens = [t.lower() for t in must_contain]
    for c in df.columns:
        lc = c.lower()
        if all(t in lc for t in tokens):
            return c
    return None

# ------------------------------------------------------------
# Data load + filters
# ------------------------------------------------------------
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
if not uploaded:
    st.info("Upload your 'Project List Main.xlsx' file.")
    st.stop()

df = pd.read_excel(uploaded)
df.columns = [normalize(c) for c in df.columns]

# Filters
if "country" in df.columns:
    countries = sorted(df["country"].dropna().unique())
    selected_countries = st.sidebar.multiselect("Countries", countries, default=countries)
    df = df[df["country"].isin(selected_countries)]

if "customer" in df.columns:
    customers = sorted(df["customer"].dropna().unique())
    selected_customers = st.sidebar.multiselect("Customers", customers, default=customers)
    df = df[df["customer"].isin(selected_customers)]

st.sidebar.success(f"✅ {len(df)} projects loaded after filters")

# Numeric coercions where meaningful
num_candidates = [
    "contract_value", "cash_received",
    "cm2_forecast", "cm2_actual", "cm2_budget",
    "cm2pct_forecast", "cm2pct_actual", "cm2pct_budget",
    "total_penalties", "total_o", "total_delays", "check_v"
]
for col in num_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Identify Real CM2% Deviation column once and reuse where we need deviation
REAL_DEV_COL = (
    find_col(df, ["real","cm2","pct","dev"]) or
    find_col(df, ["cm2pct","real","dev"]) or
    find_col(df, ["cm2pct","dev"])
)

# ------------------------------------------------------------
# KPIs
# (Only TEXT changed: label now says "Real CM2% Deviation".
#  Calculation stays as BEFORE: cm2_actual weighted by contract_value.)
# ------------------------------------------------------------
total_contract = df.get("contract_value", pd.Series(dtype=float)).sum()
total_cash = df.get("cash_received", pd.Series(dtype=float)).sum()
weighted_fore_pct = (df.get("cm2_forecast", 0).sum() / total_contract * 100) if total_contract else 0
weighted_real_pct = (df.get("cm2_actual", 0).sum() / total_contract * 100) if total_contract else 0  # unchanged

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Projects", len(df))
with c2: st.metric("Contract Value Σ (EUR)", f"{total_contract:,.0f}")
with c3: st.metric("Cash Received Σ (EUR)", f"{total_cash:,.0f}")
with c4: st.metric("Compounded CM2% (Forecast)", f"{weighted_fore_pct:,.1f}%")
with c5: st.metric("Real CM2% Deviation (weighted)", f"{weighted_real_pct:,.1f}%" if total_contract else "—")

# ------------------------------------------------------------
# Build service-level long table
# ------------------------------------------------------------
svc_rows = []
for s in SERVICE_BLOCKS:
    for field in ["budget", "forecast", "actual", "h_o", "b_o", "delay"]:
        col = f"{s}_{field}"
        if col not in df.columns:
            df[col] = np.nan
    for _, r in df.iterrows():
        svc_rows.append({
            "project_id": r.get("project_id"),
            "service": s.upper(),
            "budget": safe_num(r[f"{s}_budget"]),
            "forecast": safe_num(r[f"{s}_forecast"]),
            "actual": safe_num(r[f"{s}_actual"]),
            "h_o": safe_int(r.get(f"{s}_h_o")),
            "b_o": safe_int(r.get(f"{s}_b_o")),
            "delay": safe_int(r.get(f"{s}_delay"))
        })
svc = pd.DataFrame(svc_rows)
svc["inflation"] = np.where(svc["budget"] > 0, svc["actual"] / svc["budget"], np.nan)
PRETTY_UP = {k.upper(): v for k,v in SERVICE_PRETTY.items()}

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tabs = st.tabs([
    "Overview", "Internal Services Metrics",
    "Margin Bridge", "Probabilities", "Overrun Heatmap", "Drivers"
])

# ------------------------------------------------------------
# 1) Overview
# ------------------------------------------------------------
with tabs[0]:
    st.subheader("Portfolio Overview")

    # Correlation heatmap
    heat_cols = [c for c in [
        "contract_value", "cash_received",
        "cm2_budget", "cm2_forecast", "cm2_actual",
        "cm2pct_budget", "cm2pct_forecast", "cm2pct_actual",
        "total_penalties", "total_o", "total_delays"
    ] if c in df.columns]
    df_num = df[heat_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if not df_num.empty and df_num.shape[1] > 1:
        corr = df_num.corr("spearman")
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="tealrose",
            title="Correlation heatmap (incl. hours & budget overruns)"
        )
        st.plotly_chart(fig, use_container_width=True, config=plotly_config("correlation_heatmap"))

    # Bubble chart
    st.subheader("Contract Value vs CM2% Forecast (bubble = penalties)")
    df_bubble = df.copy()
    for c in ["contract_value", "cm2pct_forecast", "total_penalties"]:
        if c in df_bubble.columns:
            df_bubble[c] = pd.to_numeric(df_bubble[c], errors="coerce").fillna(0)
    df_bubble = df_bubble.dropna(subset=["contract_value", "cm2pct_forecast"])
    if not df_bubble.empty:
        df_bubble["has_penalty"] = df_bubble["total_penalties"] > 0
        fig = px.scatter(
            df_bubble,
            x="contract_value", y="cm2pct_forecast",
            size="total_penalties",
            color="country" if "country" in df_bubble.columns else None,
            hover_data=[c for c in ["project_id", "customer", "total_penalties", "cm2pct_forecast"] if c in df_bubble.columns],
            color_discrete_sequence=px.colors.qualitative.Set2,
            size_max=50
        )
        fig.update_traces(marker=dict(
            symbol=["square" if not p else "circle" for p in df_bubble["has_penalty"]],
            line=dict(width=0.4, color="rgba(0,0,0,0.3)")
        ))
        fig.update_layout(
            xaxis_title="Contract Value (EUR)",
            yaxis_title="CM2% Forecast",
            plot_bgcolor="white", paper_bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True, config=plotly_config("penalty_bubble"))

    # Margin Scatter
    st.subheader("Margin Scatter")
    ms_mode = st.radio("Y-axis:", ["CM2% Forecast", "CM2 Forecast (EUR)"], horizontal=True, key="ov_ms_mode")
    ms_y = "cm2pct_forecast" if ms_mode == "CM2% Forecast" else "cm2_forecast"
    if ms_y in df.columns and "contract_value" in df.columns:
        fig2 = px.scatter(
            df, x="contract_value", y=ms_y,
            color="country" if "country" in df.columns else None,
            hover_data=[c for c in ["project_id", "customer"] if c in df.columns],
            color_discrete_sequence=px.colors.qualitative.Set2,
            title=f"Contract Value vs {ms_y.replace('_',' ').title()}"
        )
        st.plotly_chart(fig2, use_container_width=True, config=plotly_config("margin_scatter"))

# ------------------------------------------------------------
# 2) Internal Services Metrics
# ------------------------------------------------------------
with tabs[1]:
    st.subheader("Internal Services Metrics")

    # Table: include ALL services
    svc_agg_all = svc.groupby("service").agg(
        projects=("project_id", "nunique"),
        budget=("budget", "sum"),
        actual=("actual", "sum"),
        forecast=("forecast", "sum"),
        h_overruns=("h_o", "sum"),
        b_overruns=("b_o", "sum"),
        delays=("delay", "sum"),
        median_inflation=("inflation", "median"),
    ).reset_index()
    svc_agg_all["Service"] = svc_agg_all["service"].map(PRETTY_UP)
    svc_agg_all["inflation_factor"] = svc_agg_all["actual"] / svc_agg_all["budget"]

    st.dataframe(
        svc_agg_all[[
            "Service", "projects", "budget", "actual", "forecast",
            "h_overruns", "b_overruns", "delays", "median_inflation", "inflation_factor"
        ]],
        use_container_width=True,
    )

    # Chart: exclude MAN & PROC (they’re n/a for some metrics)
    svc_chart = svc_agg_all[~svc_agg_all["service"].isin(["MAN", "PROC"])]
    fig3 = px.bar(
        svc_chart, x="Service", y=["budget", "actual", "forecast"],
        barmode="group", color_discrete_sequence=px.colors.sequential.Tealgrn,
        title="Budget vs Actual vs Forecast (hours)"
    )
    st.plotly_chart(fig3, use_container_width=True, config=plotly_config("services_baf"))

# ------------------------------------------------------------
# 3) Margin Bridge (Real CM2% Deviation from file)
# ------------------------------------------------------------
with tabs[2]:
    st.subheader("Real CM2% Deviation")

    real_dev_col = REAL_DEV_COL
    eur_delta_col = (
        find_col(df, ["cm2","eur","dev"]) or
        find_col(df, ["delta","cm2"]) or
        find_col(df, ["cm2","eur","delta"])
    )

    if real_dev_col is None:
        st.warning("Could not find a precomputed 'Real CM2% Deviation' column in the Excel.")
    else:
        bridge = df[[c for c in ["project_id", "customer", real_dev_col, eur_delta_col] if c in df.columns]].copy()
        bridge = bridge.dropna(subset=[real_dev_col])
        if not bridge.empty:
            fig4 = px.bar(
                bridge, x=("project_id" if "project_id" in bridge.columns else bridge.index),
                y=real_dev_col,
                color=np.where(bridge[real_dev_col] > 0, "Positive", "Negative"),
                hover_data=[c for c in ["customer", eur_delta_col] if c in bridge.columns],
                color_discrete_sequence=["#7fc8a9", "#e07a5f"],
                title="Real CM2% Deviation (hover shows € delta)"
            )
            fig4.update_layout(xaxis_title="Project", yaxis_title="Real CM2% Δ")
            st.plotly_chart(fig4, use_container_width=True, config=plotly_config("real_cm2_dev_bridge"))
        else:
            st.info("No data available for Real CM2% Deviation.")

# ------------------------------------------------------------
# 4) Probabilities
# ------------------------------------------------------------
with tabs[3]:
    st.subheader("Probabilities")

    def ensure_defaults():
        ss = st.session_state
        ss.setdefault("p_target_col",
            REAL_DEV_COL or
            find_col(df, ["cm2pct","forecast"]) or
            (df.columns[0] if len(df.columns) else "x")
        )
        ss.setdefault("p_logic", "AND")
        ss.setdefault("p_use_c2", False)
        ss.setdefault("p_c1_col", df.columns[0])
        ss.setdefault("p_c1_op", "≥")
        ss.setdefault("p_c1_thr", 0.0)
        ss.setdefault("p_c1_flag_val", 1)
        ss.setdefault("p_c2_col", df.columns[0])
        ss.setdefault("p_c2_op", "≥")
        ss.setdefault("p_c2_thr", 0.0)
        ss.setdefault("p_c2_flag_val", 1)

        if "penalty_flag" not in df.columns and "total_penalties" in df.columns:
            df["penalty_flag"] = (pd.to_numeric(df["total_penalties"], errors="coerce").fillna(0) > 0).astype(int)

        numeric_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        flag_candidates = [c for c in df.columns if is_binary_series(df[c])]
        ss.setdefault("b_num", numeric_candidates[0] if numeric_candidates else df.columns[0])
        ss.setdefault("b_op", "≥")
        ss.setdefault("b_thr", 0.0)
        ss.setdefault("b_flag", flag_candidates[0] if flag_candidates else df.columns[0])

        ss.setdefault("c_flag", flag_candidates[0] if flag_candidates else df.columns[0])
        ss.setdefault("c_num", numeric_candidates[0] if numeric_candidates else df.columns[0])
        ss.setdefault("c_op", "≥")
        ss.setdefault("c_thr", 0.0)

    ensure_defaults()

    def apply_preset(name: str):
        if name == "eng_delay_yes":
            col = "eng_delay" if "eng_delay" in df.columns else next((c for c in df.columns if "eng" in c and "delay" in c), df.columns[0])
            st.session_state.update({
                "p_use_c2": False,
                "p_c1_col": col,
                "p_c1_op": "=", "p_c1_thr": 1.0, "p_c1_flag_val": 1
            })
        elif name == "constr_bo_and_cpm_ho":
            c1 = "constr_b_o" if "constr_b_o" in df.columns else next((c for c in df.columns if "constr" in c and "b_o" in c), df.columns[0])
            c2 = "cpm_h_o" if "cpm_h_o" in df.columns else next((c for c in df.columns if "cpm" in c and "h_o" in c), df.columns[0])
            st.session_state.update({
                "p_use_c2": True, "p_logic": "AND",
                "p_c1_col": c1, "p_c1_op": ">", "p_c1_thr": 0.0,
                "p_c2_col": c2, "p_c2_op": ">", "p_c2_thr": 0.0
            })
        elif name == "penalty_given_proc_delay":
            proc_delay_col = "proc_delay" if "proc_delay" in df.columns else next((c for c in df.columns if "proc" in c and "delay" in c), df.columns[0])
            st.session_state.update({
                "b_num": "total_penalties" if "total_penalties" in df.columns else st.session_state["b_num"],
                "b_op": ">", "b_thr": 0.0,
                "b_flag": proc_delay_col
            })
        st.rerun()

    target_choices = [c for c in df.columns if "cm2" in c and "pct" in c] or list(df.columns)
    default_target = REAL_DEV_COL if REAL_DEV_COL in target_choices else target_choices[0]
    p_target_col = st.selectbox("Outcome metric that defines 'negative margin' (< 0):",
                                target_choices, index=target_choices.index(default_target), key="p_target_sel")
    target_flag = (pd.to_numeric(df[p_target_col], errors="coerce") < 0).astype(int)

    st.markdown("### A) Probability of **negative margin** given conditions")
    colA, colB = st.columns(2)

    with colA:
        c1_idx = list(df.columns).index(st.session_state.get("p_c1_col", df.columns[0]))
        c1_col = st.selectbox("Condition 1", df.columns, index=c1_idx, format_func=humanize_col, key="p_c1_col_sel")
        if is_binary_series(df[c1_col]):
            c1_val = st.selectbox("is", ["Yes (1)", "No (0)"], key="p_c1_flag_sel")
            c1_flag_val = 1 if "Yes" in c1_val else 0
            c1_desc = f"{humanize_col(c1_col)} = {'Yes' if c1_flag_val==1 else 'No'}"
            m1 = (pd.to_numeric(df[c1_col], errors="coerce") == c1_flag_val)
        else:
            ops = ["≥", ">", "≤", "<", "="]
            op = st.selectbox("operator", ops, index=ops.index(st.session_state.get("p_c1_op","≥")), key="p_c1_op_sel")
            thr = st.number_input("threshold", value=float(st.session_state.get("p_c1_thr",0.0)), step=1.0, key="p_c1_thr_num")
            c1_desc = f"{humanize_col(c1_col)} {op} {thr}"
            m1 = (to_flag(df[c1_col], op, float(thr)) == 1)

    with colB:
        use_c2 = st.checkbox("Add Condition 2", value=st.session_state.get("p_use_c2", False), key="p_use_c2_chk")
        logic = st.radio("Logic", ["AND", "OR"], horizontal=True, index=(0 if st.session_state.get("p_logic","AND")=="AND" else 1), key="p_logic_radio")
        if use_c2:
            c2_idx = list(df.columns).index(st.session_state.get("p_c2_col", df.columns[0]))
            c2_col = st.selectbox("Condition 2", df.columns, index=c2_idx, format_func=humanize_col, key="p_c2_col_sel")
            if is_binary_series(df[c2_col]):
                c2_val = st.selectbox("is", ["Yes (1)", "No (0)"], key="p_c2_flag_sel")
                c2_flag_val = 1 if "Yes" in c2_val else 0
                c2_desc = f"{humanize_col(c2_col)} = {'Yes' if c2_flag_val==1 else 'No'}"
                m2 = (pd.to_numeric(df[c2_col], errors="coerce") == c2_flag_val)
            else:
                ops = ["≥", ">", "≤", "<", "="]
                op2 = st.selectbox("operator", ops, index=ops.index(st.session_state.get("p_c2_op","≥")), key="p_c2_op_sel")
                thr2 = st.number_input("threshold", value=float(st.session_state.get("p_c2_thr",0.0)), step=1.0, key="p_c2_thr_num")
                c2_desc = f"{humanize_col(c2_col)} {op2} {thr2}"
                m2 = (to_flag(df[c2_col], op2, float(thr2)) == 1)
        else:
            m2 = pd.Series(True, index=df.index)
            c2_desc = "None"

    mask = (m1 | m2) if (use_c2 and logic == "OR") else (m1 & m2)
    cond_text = c1_desc if not use_c2 else f"{c1_desc} **{logic}** {c2_desc}"
    n = int(mask.sum())
    if n > 0:
        prob = target_flag[mask].mean() * 100
        st.markdown(
            f"""<div style="background:#eaf7ef;border:1px solid #bce3c7;padding:14px;border-radius:8px;font-size:1.05rem;">
            <b>Probability of a project ending with negative margin if {cond_text} = {prob:.1f}%</b>
            <span style="opacity:.65">(n={n})</span></div>""",
            unsafe_allow_html=True
        )
    else:
        st.info("No rows satisfy the chosen conditions.")

    st.markdown("#### Quick presets")
    p1,p2,p3 = st.columns(3)
    with p1:
        st.button("P(neg. margin | ENG delay=Yes)", use_container_width=True,
                  on_click=apply_preset, args=("eng_delay_yes",))
    with p2:
        st.button("P(neg. margin | CONSTR b_o>0 AND CPM h_o>0)", use_container_width=True,
                  on_click=apply_preset, args=("constr_bo_and_cpm_ho",))
    with p3:
        st.button("P(penalty>0 | PROC delay=Yes)", use_container_width=True,
                  on_click=apply_preset, args=("penalty_given_proc_delay",))

    st.divider()

    # B) Numeric ↔ Flag
    st.markdown("### B) Numeric ↔ Flag (and reverse)")
    numeric_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    flag_candidates = [c for c in df.columns if is_binary_series(df[c])]
    bn_idx = numeric_candidates.index(st.session_state.get("b_num", numeric_candidates[0])) if numeric_candidates else 0
    bf_idx = flag_candidates.index(st.session_state.get("b_flag", flag_candidates[0])) if flag_candidates else 0
    colN, colF = st.columns(2)
    with colN:
        b_num = st.selectbox("Numeric variable", numeric_candidates if numeric_candidates else df.columns,
                             index=bn_idx, format_func=humanize_col, key="p_b_num_sel")
        b_comp = st.selectbox("Compare", ["≥", ">", "≤", "<", "="], index=["≥",">","≤","<","="].index(st.session_state.get("b_op","≥")), key="p_b_op_sel")
        b_thr = st.number_input("Threshold (B)", value=float(st.session_state.get("b_thr",0.0)), step=1.0, key="p_b_thr_num")
    with colF:
        b_flag = st.selectbox("Flag variable (0/1)", flag_candidates if flag_candidates else df.columns,
                              index=bf_idx, format_func=humanize_col, key="p_b_flag_sel")

    num_flag = to_flag(df[b_num], b_comp, float(b_thr))
    flag_ser = pd.to_numeric(df[b_flag], errors="coerce").fillna(0).astype(int)

    mflag = flag_ser == 1
    if int(mflag.sum()) > 0:
        pA = num_flag[mflag].mean() * 100
        st.success(f"P({humanize_col(b_num)} {b_comp} {b_thr} | {humanize_col(b_flag)}=1) = {pA:.1f}%  (n={int(mflag.sum())})")
    else:
        st.info(f"No rows with {humanize_col(b_flag)}=1.")

    mnum = num_flag == 1
    if int(mnum.sum()) > 0:
        pB = flag_ser[mnum].mean() * 100
        st.success(f"P({humanize_col(b_flag)}=1 | {humanize_col(b_num)} {b_comp} {b_thr}) = {pB:.1f}%  (n={int(mnum.sum())})")
    else:
        st.info(f"No rows where {humanize_col(b_num)} {b_comp} {b_thr}.")

    st.divider()

    # C) Flag | Numeric→Flag
    st.markdown("### C) Flag | Numeric→Flag")
    ct_idx = flag_candidates.index(st.session_state.get("c_flag", flag_candidates[0])) if flag_candidates else 0
    cn_idx = numeric_candidates.index(st.session_state.get("c_num", numeric_candidates[0])) if numeric_candidates else 0
    colC1, colC2 = st.columns(2)
    with colC1:
        c_tgt = st.selectbox("Target flag", flag_candidates if flag_candidates else df.columns,
                             index=ct_idx, format_func=humanize_col, key="p_c_flag_sel")
    with colC2:
        c_num = st.selectbox("Numeric to convert → flag", numeric_candidates if numeric_candidates else df.columns,
                             index=cn_idx, format_func=humanize_col, key="p_c_num_sel")
        c_op = st.selectbox("Operator", ["≥", ">", "≤", "<", "="], index=["≥",">","≤","<","="].index(st.session_state.get("c_op","≥")), key="p_c_op_sel")
        c_thr = st.number_input("Threshold (C)", value=float(st.session_state.get("c_thr",0.0)), step=1.0, key="p_c_thr_num")

    conv_flag = to_flag(df[c_num], c_op, float(c_thr))
    tgt_ser = pd.to_numeric(df[c_tgt], errors="coerce").fillna(0).astype(int)
    m3 = conv_flag == 1
    if int(m3.sum()) > 0:
        p3 = tgt_ser[m3].mean() * 100
        st.success(f"P({humanize_col(c_tgt)}=1 | {humanize_col(c_num)} {c_op} {c_thr}) = {p3:.1f}%  (n={int(m3.sum())})")
    else:
        st.info("No rows match the numeric→flag condition.")

    st.divider()
    st.markdown("#### Averages & Medians (portfolio)")
    stats_cols = [c for c in ["total_penalties", "total_delays", "total_o"] if c in df.columns]
    if stats_cols:
        agg = pd.DataFrame({
            "mean": df[stats_cols].mean(numeric_only=True),
            "median": df[stats_cols].median(numeric_only=True)
        })
        st.dataframe(agg.style.format({"mean": "{:.2f}", "median": "{:.2f}"}), use_container_width=True)
    else:
        st.info("No standard totals found (penalties / delays / overruns).")

# ------------------------------------------------------------
# 5) Overrun Heatmap
# ------------------------------------------------------------
with tabs[4]:
    st.subheader("Overrun & Delay density heatmap")
    svc_filtered = svc[~svc["service"].isin(["MAN", "PROC"])]
    heat = svc_filtered.groupby("service")[["h_o", "b_o", "delay"]].mean().reset_index()
    heat["Service"] = heat["service"].map(PRETTY_UP)
    melt = heat.melt(id_vars="Service", var_name="Type", value_name="Rate")
    fig6 = px.density_heatmap(
        melt, x="Type", y="Service", z="Rate",
        color_continuous_scale="tealrose",
        title="Average overrun rate by service"
    )
    st.plotly_chart(fig6, use_container_width=True, config=plotly_config("overrun_heatmap"))

# ------------------------------------------------------------
# 6) Drivers (service-level odds ratios; excludes totals & check_v)
# ------------------------------------------------------------
with tabs[5]:
    st.subheader("Drivers of CM2% deviation (service-level)")

    real_dev_col = REAL_DEV_COL or find_col(df, ["cm2pct","forecast"])
    if real_dev_col is None:
        st.warning("No suitable CM2% real/forecast deviation column found.")
    else:
        y = (pd.to_numeric(df[real_dev_col], errors="coerce") < 0).astype(int)

        feat_cols = []
        feat_df = pd.DataFrame(index=df.index)
        for s in SERVICE_BLOCKS:
            for suffix, label in [("b_o","budget_overrun"), ("h_o","hours_overrun"), ("delay","delay")]:
                col = f"{s}_{suffix}"
                if col in df.columns:
                    fcol = f"{s}_{suffix}_flag"
                    if fcol in EXCLUDED_DRIVER_FLAGS:
                        continue
                    feat_df[fcol] = (pd.to_numeric(df[col], errors="coerce").fillna(0) > 0).astype(int)
                    feat_cols.append((fcol, s, label))

        if feat_df.shape[1] == 0 or y.nunique() < 2:
            st.info("Not enough variation to fit driver model.")
        else:
            X = feat_df.values
            pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", LogisticRegression(max_iter=400))])
            pipe.fit(X, y)
            odds = np.exp(pipe.named_steps["clf"].coef_[0])

            feature_table = pd.DataFrame({
                "Feature": [f for f,_,_ in feat_cols],
                "Odds_Ratio": odds,
                "Service": [SERVICE_PRETTY.get(s.lower(), s) for _,s,_ in feat_cols],
                "Type": [typ for _,_,typ in feat_cols],
            }).sort_values("Odds_Ratio", ascending=False)

            fig7 = px.bar(
                feature_table, x="Feature", y="Odds_Ratio", color="Type",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Odds ratios by overrun/delay flag (↑ = higher risk of negative margin)"
            )
            st.plotly_chart(fig7, use_container_width=True, config=plotly_config("drivers_features"))

            feature_table["Impact"] = np.abs(np.log(feature_table["Odds_Ratio"].replace(0, np.nan))).fillna(0)
            service_impact = feature_table.groupby("Service")["Impact"].sum().reset_index().sort_values("Impact", ascending=False)
            fig8 = px.bar(
                service_impact, x="Service", y="Impact", color="Service",
                title="Service impact score (sum |log-odds|)",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig8.update_layout(showlegend=False)
            st.plotly_chart(fig8, use_container_width=True, config=plotly_config("drivers_services"))
