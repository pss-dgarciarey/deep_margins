# app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------
# Streamlit setup
# ------------------------------------------------------------
st.set_page_config(page_title="PSS Analytics Dashboard", layout="wide")
st.title("📊 PSS Project Analytics")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
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

def find_col(df: pd.DataFrame, must_contain):
    tokens = [t.lower() for t in must_contain]
    for c in df.columns:
        lc = c.lower()
        if all(t in lc for t in tokens):
            return c
    return None

def is_binary_series(s: pd.Series) -> bool:
    s = pd.to_numeric(s, errors="coerce")
    s = s.dropna()
    if s.empty: return False
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

# nicify service bits in labels
SERVICE_PRETTY = {
    "tpm": "TPM", "cpm": "CPM", "eng": "Engineering", "qa_qc_exp": "QA/QC/Exp",
    "hse": "HSE", "constr": "Construction", "com": "Commissioning",
    "man": "Manufacturing", "proc": "Procurement"
}
def humanize_col(c: str) -> str:
    s = c.lower()
    for k,v in SERVICE_PRETTY.items():
        s = s.replace(f"{k}_", f"{v} ")
    s = s.replace("b_o", "budget overrun").replace("h_o", "hours overrun").replace("delay", "delay")
    s = s.replace("cm2pct", "CM2%").replace("cm2_", "CM2 ")
    s = s.replace("_", " ").strip()
    return s[:1].upper() + s[1:]

# ------------------------------------------------------------
# Load + Filters
# ------------------------------------------------------------
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
if not uploaded:
    st.info("Upload your 'Project List Main.xlsx' file.")
    st.stop()

df = pd.read_excel(uploaded)
df.columns = [normalize(c) for c in df.columns]

# Country + Customer filters
if "country" in df.columns:
    countries = sorted(df["country"].dropna().unique())
    selected_countries = st.sidebar.multiselect("Countries", countries, default=countries)
    df = df[df["country"].isin(selected_countries)]

if "customer" in df.columns:
    customers = sorted(df["customer"].dropna().unique())
    selected_customers = st.sidebar.multiselect("Customers", customers, default=customers)
    df = df[df["customer"].isin(selected_customers)]

st.sidebar.success(f"✅ {len(df)} projects loaded after filters")

# ------------------------------------------------------------
# Numeric coercions
# ------------------------------------------------------------
for col in [
    "contract_value", "cash_received",
    "cm2_forecast", "cm2_actual", "cm2_budget",
    "cm2pct_forecast", "cm2pct_actual", "cm2pct_budget",
    "total_penalties", "total_o", "total_delays"
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ------------------------------------------------------------
# Compounded high-level metrics
# ------------------------------------------------------------
total_contract = df.get("contract_value", pd.Series(dtype=float)).sum()
total_cash = df.get("cash_received", pd.Series(dtype=float)).sum()
weighted_fore_pct = (df.get("cm2_forecast", 0).sum() / total_contract * 100) if total_contract else 0
weighted_real_pct = (df.get("cm2_actual", 0).sum() / total_contract * 100) if total_contract else 0

col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Projects", len(df))
with col2: st.metric("Contract Value Σ (EUR)", f"{total_contract:,.0f}")
with col3: st.metric("Cash Received Σ (EUR)", f"{total_cash:,.0f}")
with col4: st.metric("Compounded CM2% (Forecast)", f"{weighted_fore_pct:,.1f}%")
with col5: st.metric("Real Compounded CM2% (Actual)", f"{weighted_real_pct:,.1f}%")

# ------------------------------------------------------------
# Build service-level table (hours/budget/delay)
# ------------------------------------------------------------
service_blocks = ["tpm", "cpm", "eng", "qa_qc_exp", "hse", "constr", "com", "man", "proc"]
svc_rows = []
for s in service_blocks:
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

pretty = {k.upper(): v for k,v in SERVICE_PRETTY.items()}

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

    heat_cols = [c for c in [
        "contract_value", "cash_received",
        "cm2_budget", "cm2_forecast", "cm2_actual",
        "cm2pct_budget", "cm2pct_forecast", "cm2pct_actual",
        "total_penalties", "total_o", "total_delays"
    ] if c in df.columns]
    df_num = df[heat_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if not df_num.empty and df_num.shape[1] > 1:
        corr = df_num.corr("spearman")
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale="tealrose",
                        title="Correlation heatmap (incl. hours & budget overruns)")
        st.plotly_chart(fig, use_container_width=True, config=plotly_config("correlation_heatmap"))

    st.subheader("Contract Value vs CM2% Forecast (bubble = penalties)")
    df_bubble = df.copy()
    for c in ["contract_value", "cm2pct_forecast", "total_penalties"]:
        if c in df_bubble.columns:
            df_bubble[c] = pd.to_numeric(df_bubble[c], errors="coerce").fillna(0)
    df_bubble = df_bubble.dropna(subset=["contract_value", "cm2pct_forecast"])
    if not df_bubble.empty:
        df_bubble["has_penalty"] = df_bubble["total_penalties"] > 0
        fig = px.scatter(
            df_bubble, x="contract_value", y="cm2pct_forecast",
            size="total_penalties",
            color="country" if "country" in df_bubble.columns else None,
            hover_data=[c for c in ["project_id", "customer", "total_penalties", "cm2pct_forecast"] if c in df_bubble.columns],
            color_discrete_sequence=px.colors.qualitative.Set2, size_max=50
        )
        fig.update_traces(marker=dict(
            symbol=["square" if not p else "circle" for p in df_bubble["has_penalty"]],
            line=dict(width=0.4, color="rgba(0,0,0,0.3)")
        ))
        fig.update_layout(xaxis_title="Contract Value (EUR)", yaxis_title="CM2% Forecast")
        st.plotly_chart(fig, use_container_width=True, config=plotly_config("penalty_bubble"))

    st.subheader("Margin Scatter")
    mode = st.radio("Y-axis:", ["CM2% Forecast", "CM2 Forecast (EUR)"], horizontal=True, key="scatter_y")
    yaxis = "cm2pct_forecast" if mode == "CM2% Forecast" else "cm2_forecast"
    if yaxis in df.columns:
        df[yaxis] = pd.to_numeric(df[yaxis], errors="coerce")
        fig2 = px.scatter(
            df, x="contract_value", y=yaxis,
            color="country" if "country" in df.columns else None,
            hover_data=[c for c in ["project_id", "customer"] if c in df.columns],
            color_discrete_sequence=px.colors.qualitative.Set2,
            title=f"Contract Value vs {yaxis.replace('_', ' ').title()}"
        )
        st.plotly_chart(fig2, use_container_width=True, config=plotly_config("margin_scatter"))

# ------------------------------------------------------------
# 2) Internal Services Metrics
# ------------------------------------------------------------
with tabs[1]:
    st.subheader("Internal Services Metrics")

    svc_agg = svc.groupby("service").agg(
        projects=("project_id", "nunique"),
        budget=("budget", "sum"),
        actual=("actual", "sum"),
        forecast=("forecast", "sum"),
        h_overruns=("h_o", "sum"),
        b_overruns=("b_o", "sum"),
        delays=("delay", "sum"),
        median_inflation=("inflation", "median"),
    ).reset_index()
    svc_agg["Service"] = svc_agg["service"].map(pretty)
    svc_agg["inflation_factor"] = svc_agg["actual"] / svc_agg["budget"]

    st.dataframe(
        svc_agg[[
            "Service", "projects", "budget", "actual", "forecast",
            "h_overruns", "b_overruns", "delays", "median_inflation", "inflation_factor"
        ]],
        use_container_width=True,
    )

    svc_chart = svc_agg[~svc_agg["service"].isin(["MAN", "PROC"])]
    fig3 = px.bar(
        svc_chart, x="Service", y=["budget", "actual", "forecast"],
        barmode="group", color_discrete_sequence=px.colors.sequential.Tealgrn,
        title="Budget vs Actual vs Forecast (hours)"
    )
    st.plotly_chart(fig3, use_container_width=True, config=plotly_config("services_baf"))

# ------------------------------------------------------------
# 3) Margin Bridge  (Real CM2% Deviation from Excel)
# ------------------------------------------------------------
with tabs[2]:
    st.subheader("Real CM2% Deviation")
    real_dev_col = (find_col(df, ["real","cm2","pct","dev"]) or
                    find_col(df, ["cm2pct","real","dev"]) or
                    find_col(df, ["cm2pct","real"]))
    eur_delta_col = (find_col(df, ["cm2","eur","dev"]) or
                     find_col(df, ["delta","cm2"]) or
                     find_col(df, ["cm2","eur","delta"]))
    if eur_delta_col is None and {"cm2_forecast","cm2_budget"} <= set(df.columns):
        df["_cm2_eur_delta_tmp_"] = df["cm2_forecast"] - df["cm2_budget"]
        eur_delta_col = "_cm2_eur_delta_tmp_"

    if real_dev_col is None:
        st.warning("Could not find a precomputed 'Real CM2% Deviation' column in the file.")
    else:
        bridge = df[[c for c in ["project_id","customer", real_dev_col, eur_delta_col] if c in df.columns]].copy()
        bridge = bridge.dropna(subset=[real_dev_col])
        if not bridge.empty:
            fig4 = px.bar(
                bridge, x="project_id", y=real_dev_col,
                color=np.where(bridge[real_dev_col] > 0, "Positive", "Negative"),
                hover_data=[c for c in ["customer", eur_delta_col] if c in bridge.columns],
                color_discrete_sequence=["#7fc8a9","#e07a5f"],
                title="Real CM2% Deviation (hover shows € delta)"
            )
            fig4.update_layout(xaxis_title="Project", yaxis_title="Real CM2% Δ")
            st.plotly_chart(fig4, use_container_width=True, config=plotly_config("real_cm2_dev_bridge"))
        else:
            st.info("No data available for Real CM2% Deviation.")

# ------------------------------------------------------------
# 4) Probabilities (Sentence UI + working presets)
# ------------------------------------------------------------
with tabs[3]:
    st.subheader("Probabilities")

    # --- defaults in session state to avoid StreamlitAPIException ---
    def ensure_defaults():
        ss = st.session_state
        ss.setdefault("p_target_col",
            find_col(df, ["real","cm2","pct","dev"]) or
            find_col(df, ["cm2pct","real","dev"]) or
            find_col(df, ["cm2pct","forecast"]) or
            df.columns[0]
        )
        ss.setdefault("p_logic", "AND")
        ss.setdefault("p_use_cond2", False)
        ss.setdefault("p_cond1_col", df.columns[0])
        ss.setdefault("p_cond1_kind", "flag")  # 'flag' or 'num'
        ss.setdefault("p_cond1_flag_val", 1)
        ss.setdefault("p_cond1_op", "≥")
        ss.setdefault("p_cond1_thr", 0.0)
        ss.setdefault("p_cond2_col", df.columns[0])
        ss.setdefault("p_cond2_kind", "flag")
        ss.setdefault("p_cond2_flag_val", 1)
        ss.setdefault("p_cond2_op", "≥")
        ss.setdefault("p_cond2_thr", 0.0)
        # section B defaults
        numeric_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        flag_candidates = [c for c in df.columns if is_binary_series(df[c])]
        if "total_penalties" in df.columns and "penalty_flag" not in df.columns:
            df["penalty_flag"] = (pd.to_numeric(df["total_penalties"], errors="coerce").fillna(0) > 0).astype(int)
            flag_candidates.append("penalty_flag")
        ss.setdefault("b_num", numeric_candidates[0] if numeric_candidates else df.columns[0])
        ss.setdefault("b_comp", "≥")
        ss.setdefault("b_thr", 0.0)
        ss.setdefault("b_flag", flag_candidates[0] if flag_candidates else df.columns[0])
        # section C defaults
        ss.setdefault("c_tgt", flag_candidates[0] if flag_candidates else df.columns[0])
        ss.setdefault("c_num", numeric_candidates[0] if numeric_candidates else df.columns[0])
        ss.setdefault("c_op", "≥")
        ss.setdefault("c_thr", 0.0)

    ensure_defaults()

    # --- PRESET HANDLERS ---
    def apply_preset(name: str):
        # set state then rerun so widgets pick up values
        if name == "eng_delay":
            # Condition: ENG delay = Yes
            col = "eng_delay" if "eng_delay" in df.columns else next((c for c in df.columns if "eng" in c and "delay" in c), df.columns[0])
            st.session_state.update({
                "p_use_cond2": False,
                "p_cond1_col": col,
                "p_cond1_kind": "flag",
                "p_cond1_flag_val": 1
            })
        elif name == "constr_bo_and_cpm_ho":
            c1 = "constr_b_o" if "constr_b_o" in df.columns else next((c for c in df.columns if "constr" in c and "b_o" in c), df.columns[0])
            c2 = "cpm_h_o" if "cpm_h_o" in df.columns else next((c for c in df.columns if "cpm" in c and "h_o" in c), df.columns[0])
            st.session_state.update({
                "p_use_cond2": True, "p_logic": "AND",
                "p_cond1_col": c1, "p_cond1_kind": "num", "p_cond1_op": ">", "p_cond1_thr": 0.0,
                "p_cond2_col": c2, "p_cond2_kind": "num", "p_cond2_op": ">", "p_cond2_thr": 0.0
            })
        elif name == "penalty_given_proc_delay":
            if "penalty_flag" not in df.columns and "total_penalties" in df.columns:
                df["penalty_flag"] = (pd.to_numeric(df["total_penalties"], errors="coerce").fillna(0) > 0).astype(int)
            proc_delay_col = "proc_delay" if "proc_delay" in df.columns else next((c for c in df.columns if "proc" in c and "delay" in c), df.columns[0])
            st.session_state.update({
                "b_num": "total_penalties" if "total_penalties" in df.columns else st.session_state["b_num"],
                "b_comp": ">", "b_thr": 0.0,
                "b_flag": proc_delay_col
            })
        st.rerun()

    # OUTCOME
    target_choices = [c for c in df.columns if "cm2" in c and "pct" in c]
    p_target_col = st.selectbox(
        "Outcome metric that defines 'negative margin' (< 0):",
        target_choices if target_choices else df.columns,
        index=(target_choices.index(st.session_state["p_target_col"]) if target_choices and st.session_state["p_target_col"] in target_choices else 0),
        key="p_target_col"
    )
    target_flag = (pd.to_numeric(df[p_target_col], errors="coerce") < 0).astype(int)

    st.markdown("### A) Probability of **negative margin** given conditions")

    # --- Condition 1 ---
    colA, colB = st.columns([1,1])
    with colA:
        c1_col = st.selectbox(
            "Condition 1",
            df.columns,
            index=list(df.columns).index(st.session_state["p_cond1_col"]),
            key="p_cond1_col",
            format_func=humanize_col
        )
        c1_is_flag = is_binary_series(df[c1_col])
        if c1_is_flag:
            st.session_state["p_cond1_kind"] = "flag"
            c1_val = st.selectbox("is", ["Yes (1)", "No (0)"], index=0 if st.session_state["p_cond1_flag_val"]==1 else 1, key="p_cond1_flag_val_str")
            st.session_state["p_cond1_flag_val"] = 1 if "Yes" in c1_val else 0
        else:
            st.session_state["p_cond1_kind"] = "num"
            st.session_state["p_cond1_op"] = st.selectbox("operator", ["≥", ">", "≤", "<", "="], index=["≥",">","≤","<","="].index(st.session_state["p_cond1_op"]), key="p_cond1_op")
            st.session_state["p_cond1_thr"] = st.number_input("threshold", value=float(st.session_state["p_cond1_thr"]), step=1.0, key="p_cond1_thr")

    # --- Condition 2 ---
    with colB:
        st.checkbox("Add Condition 2", value=st.session_state["p_use_cond2"], key="p_use_cond2")
        st.radio("Logic", ["AND", "OR"], horizontal=True, key="p_logic")
        if st.session_state["p_use_cond2"]:
            c2_col = st.selectbox(
                "Condition 2",
                df.columns,
                index=list(df.columns).index(st.session_state["p_cond2_col"]),
                key="p_cond2_col",
                format_func=humanize_col
            )
            c2_is_flag = is_binary_series(df[c2_col])
            if c2_is_flag:
                st.session_state["p_cond2_kind"] = "flag"
                c2_val = st.selectbox("is", ["Yes (1)", "No (0)"], index=0 if st.session_state["p_cond2_flag_val"]==1 else 1, key="p_cond2_flag_val_str")
                st.session_state["p_cond2_flag_val"] = 1 if "Yes" in c2_val else 0
            else:
                st.session_state["p_cond2_kind"] = "num"
                st.session_state["p_cond2_op"] = st.selectbox("operator", ["≥", ">", "≤", "<", "="], index=["≥",">","≤","<","="].index(st.session_state["p_cond2_op"]), key="p_cond2_op")
                st.session_state["p_cond2_thr"] = st.number_input("threshold", value=float(st.session_state["p_cond2_thr"]), step=1.0, key="p_cond2_thr")

    # Build masks
    def build_mask(col, kind, flag_val, op, thr):
        if kind == "flag":
            return (pd.to_numeric(df[col], errors="coerce") == int(flag_val))
        return (to_flag(df[col], op, float(thr)) == 1)

    m1 = build_mask(
        st.session_state["p_cond1_col"], st.session_state["p_cond1_kind"],
        st.session_state.get("p_cond1_flag_val",1),
        st.session_state.get("p_cond1_op","≥"),
        st.session_state.get("p_cond1_thr",0.0)
    )
    if st.session_state["p_use_cond2"]:
        m2 = build_mask(
            st.session_state["p_cond2_col"], st.session_state["p_cond2_kind"],
            st.session_state.get("p_cond2_flag_val",1),
            st.session_state.get("p_cond2_op","≥"),
            st.session_state.get("p_cond2_thr",0.0)
        )
        mask = m1 | m2 if st.session_state["p_logic"] == "OR" else m1 & m2
        cond_text = f"{humanize_col(st.session_state['p_cond1_col'])} {'= Yes' if st.session_state['p_cond1_kind']=='flag' and st.session_state.get('p_cond1_flag_val',1)==1 else (st.session_state.get('p_cond1_op','≥')+' '+str(st.session_state.get('p_cond1_thr',0.0)) if st.session_state['p_cond1_kind']=='num' else '= No')}"
        cond_text2 = f"{humanize_col(st.session_state['p_cond2_col'])} {'= Yes' if st.session_state['p_cond2_kind']=='flag' and st.session_state.get('p_cond2_flag_val',1)==1 else (st.session_state.get('p_cond2_op','≥')+' '+str(st.session_state.get('p_cond2_thr',0.0)) if st.session_state['p_cond2_kind']=='num' else '= No')}"
        cond_text = f"{cond_text} **{st.session_state['p_logic']}** {cond_text2}"
    else:
        mask = m1
        cond_text = f"{humanize_col(st.session_state['p_cond1_col'])} {'= Yes' if st.session_state['p_cond1_kind']=='flag' and st.session_state.get('p_cond1_flag_val',1)==1 else (st.session_state.get('p_cond1_op','≥')+' '+str(st.session_state.get('p_cond1_thr',0.0)) if st.session_state['p_cond1_kind']=='num' else '= No')}"

    n = int(mask.sum())
    if n > 0:
        prob = target_flag[mask].mean() * 100
        st.markdown(f"""
        <div style="background:#eaf7ef;border:1px solid #bce3c7;padding:14px;border-radius:8px;font-size:1.05rem;">
        <b>Probability of a project ending with negative margin if {cond_text} = {prob:.1f}%</b> <span style="opacity:.65">(n={n})</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No rows satisfy the chosen conditions.")

    st.markdown("#### Quick presets")
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.button("P(neg. margin | ENG delay=Yes)", use_container_width=True, on_click=apply_preset, args=("eng_delay",))
    with pc2:
        st.button("P(neg. margin | CONSTR b_o>0 AND CPM h_o>0)", use_container_width=True, on_click=apply_preset, args=("constr_bo_and_cpm_ho",))
    with pc3:
        st.button("P(penalty=1 | PROC delay=Yes)", use_container_width=True, on_click=apply_preset, args=("penalty_given_proc_delay",))

    st.divider()
    st.markdown("### B) Numeric ↔ Flag (and reverse)")

    # candidates (recompute after penalty_flag possibly added)
    numeric_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    flag_candidates = [c for c in df.columns if is_binary_series(df[c])]
    if "penalty_flag" in df.columns and "penalty_flag" not in flag_candidates:
        flag_candidates.append("penalty_flag")

    colN, colF = st.columns([1,1])
    with colN:
        st.selectbox("Numeric variable", numeric_candidates, key="b_num", index=(numeric_candidates.index(st.session_state["b_num"]) if st.session_state["b_num"] in numeric_candidates else 0), format_func=humanize_col)
        st.selectbox("Compare", ["≥", ">", "≤", "<", "="], key="b_comp", index=["≥",">","≤","<","="].index(st.session_state["b_comp"]))
        st.number_input("Threshold", value=float(st.session_state["b_thr"]), step=1.0, key="b_thr")
    with colF:
        st.selectbox("Flag variable (0/1)", flag_candidates if flag_candidates else df.columns, key="b_flag", index=(flag_candidates.index(st.session_state["b_flag"]) if flag_candidates and st.session_state["b_flag"] in flag_candidates else 0), format_func=humanize_col)

    num_flag = to_flag(df[st.session_state["b_num"]], st.session_state["b_comp"], float(st.session_state["b_thr"]))
    bflag = pd.to_numeric(df[st.session_state["b_flag"]], errors="coerce").fillna(0).astype(int)

    m1b = bflag == 1
    if int(m1b.sum()) > 0:
        p1 = num_flag[m1b].mean() * 100
        st.success(f"P({humanize_col(st.session_state['b_num'])} {st.session_state['b_comp']} {st.session_state['b_thr']} | {humanize_col(st.session_state['b_flag'])}=1) = {p1:.1f}%  (n={int(m1b.sum())})")
    else:
        st.info(f"No rows with {humanize_col(st.session_state['b_flag'])}=1.")

    m2b = num_flag == 1
    if int(m2b.sum()) > 0:
        p2 = bflag[m2b].mean() * 100
        st.success(f"P({humanize_col(st.session_state['b_flag'])}=1 | {humanize_col(st.session_state['b_num'])} {st.session_state['b_comp']} {st.session_state['b_thr']}) = {p2:.1f}%  (n={int(m2b.sum())})")
    else:
        st.info(f"No rows where {humanize_col(st.session_state['b_num'])} {st.session_state['b_comp']} {st.session_state['b_thr']}.")

    st.divider()
    st.markdown("### C) Flag | Numeric→Flag")
    colC1, colC2 = st.columns([1,1])
    with colC1:
        st.selectbox("Target flag", flag_candidates if flag_candidates else df.columns, key="c_tgt", index=(flag_candidates.index(st.session_state["c_tgt"]) if flag_candidates and st.session_state["c_tgt"] in flag_candidates else 0), format_func=humanize_col)
    with colC2:
        st.selectbox("Numeric to convert → flag", numeric_candidates if numeric_candidates else df.columns, key="c_num", index=(numeric_candidates.index(st.session_state["c_num"]) if numeric_candidates and st.session_state["c_num"] in numeric_candidates else 0), format_func=humanize_col)
        st.selectbox("Operator", ["≥", ">", "≤", "<", "="], key="c_op", index=["≥",">","≤","<","="].index(st.session_state["c_op"]))
        st.number_input("Threshold", value=float(st.session_state["c_thr"]), step=1.0, key="c_thr")

    conv_flag = to_flag(df[st.session_state["c_num"]], st.session_state["c_op"], float(st.session_state["c_thr"]))
    tgt_ser = pd.to_numeric(df[st.session_state["c_tgt"]], errors="coerce").fillna(0).astype(int)
    m3 = conv_flag == 1
    if int(m3.sum()) > 0:
        p3 = tgt_ser[m3].mean() * 100
        st.success(f"P({humanize_col(st.session_state['c_tgt'])}=1 | {humanize_col(st.session_state['c_num'])} {st.session_state['c_op']} {st.session_state['c_thr']}) = {p3:.1f}%  (n={int(m3.sum())})")
    else:
        st.info("No rows match the numeric→flag condition.")

    st.divider()
    st.markdown("#### Averages & medians")
    stats_cols = [c for c in ["total_penalties", "total_delays", "total_o"] if c in df.columns]
    if stats_cols:
        agg = pd.DataFrame({
            "mean": df[stats_cols].mean(numeric_only=True),
            "median": df[stats_cols].median(numeric_only=True),
        })
        st.dataframe(agg.style.format({"mean": "{:.2f}", "median": "{:.2f}"}), use_container_width=True)
    else:
        st.info("No standard penalty/delay/overrun totals found for summary.")

# ------------------------------------------------------------
# 5) Overrun Heatmap
# ------------------------------------------------------------
with tabs[4]:
    st.subheader("Overrun & Delay density heatmap")
    svc_filtered = svc[~svc["service"].isin(["MAN", "PROC"])]
    heat = svc_filtered.groupby("service")[["h_o", "b_o", "delay"]].mean().reset_index()
    heat["Service"] = heat["service"].map(pretty)
    melt = heat.melt(id_vars="Service", var_name="Type", value_name="Rate")
    fig6 = px.density_heatmap(
        melt, x="Type", y="Service", z="Rate",
        color_continuous_scale="tealrose",
        title="Average overrun rate by service"
    )
    st.plotly_chart(fig6, use_container_width=True, config=plotly_config("overrun_heatmap"))

# ------------------------------------------------------------
# 6) Drivers (service-level, no totals, no check_v)
# ------------------------------------------------------------
with tabs[5]:
    st.subheader("Drivers of CM2% deviation (service-level)")
    real_dev_col = (find_col(df, ["real","cm2","pct","dev"]) or
                    find_col(df, ["cm2pct","real","dev"]) or
                    find_col(df, ["cm2pct","forecast"]))
    if real_dev_col is None:
        st.warning("No suitable CM2% deviation/real column found.")
    else:
        y = (pd.to_numeric(df[real_dev_col], errors="coerce") < 0).astype(int)
        feat_cols = []
        feat_df = pd.DataFrame(index=df.index)
        for s in service_blocks:
            for suffix, label in [("b_o","budget_overrun"), ("h_o","hours_overrun"), ("delay","delay")]:
                col = f"{s}_{suffix}"
                if col in df.columns:
                    fcol = f"{s}_{suffix}_flag"
                    feat_df[fcol] = (pd.to_numeric(df[col], errors="coerce").fillna(0) > 0).astype(int)
                    feat_cols.append((fcol, s, label))

        if feat_df.shape[1] == 0 or y.nunique() < 2:
            st.info("Not enough variation to fit driver model.")
        else:
            X = feat_df.values
            pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", LogisticRegression(max_iter=400))])
            pipe.fit(X, y)
            coefs = pipe.named_steps["clf"].coef_[0]
            odds = np.exp(coefs)

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
                service_impact, x="Service", y="Impact",
                color="Service", title="Service impact score (sum |log-odds|)",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig8.update_layout(showlegend=False)
            st.plotly_chart(fig8, use_container_width=True, config=plotly_config("drivers_services"))
