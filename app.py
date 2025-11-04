# app.py
import math
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
def normalize(c):
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

# ------------------------------------------------------------
# Load + Filters
# ------------------------------------------------------------
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
if not uploaded:
    st.info("Upload your latest 'Project List Main.xlsx'")
    st.stop()

df = pd.read_excel(uploaded, sheet_name=0, header=0)
df.columns = [normalize(c) for c in df.columns]

# Country selector
countries = sorted(df["country"].dropna().unique()) if "country" in df.columns else []
if countries:
    selected_countries = st.sidebar.multiselect("Select countries:", countries, default=countries)
    df = df[df["country"].isin(selected_countries)]

# Customer selector
customers = sorted(df["customer"].dropna().unique()) if "customer" in df.columns else []
if customers:
    selected_customers = st.sidebar.multiselect("Select customers:", customers, default=customers)
    df = df[df["customer"].isin(selected_customers)]

st.sidebar.success(f"✅ {len(df)} projects loaded after filters")

# ------------------------------------------------------------
# Derived fields
# ------------------------------------------------------------
df["contract_value"] = pd.to_numeric(df.get("contract_value"), errors="coerce")
df["cash_received"]  = pd.to_numeric(df.get("cash_received"), errors="coerce")
df["cm2_forecast"]   = pd.to_numeric(df.get("cm2_forecast"), errors="coerce")
df["cm2_actual"]     = pd.to_numeric(df.get("cm2_actual"), errors="coerce")
df["cm2pct_forecast"]= pd.to_numeric(df.get("cm2pct_forecast"), errors="coerce")
df["cm2pct_actual"]  = pd.to_numeric(df.get("cm2pct_actual"), errors="coerce")
df["total_penalties"]= pd.to_numeric(df.get("total_penalties"), errors="coerce")

# Weighted margins
total_contract = df["contract_value"].sum()
total_cash     = df["cash_received"].sum()
valid = df["contract_value"].notna() & df["cm2_forecast"].notna()
weighted_fore_pct = (df.loc[valid, "cm2_forecast"].sum()/df.loc[valid,"contract_value"].sum())*100 if valid.any() else np.nan
total_fore_eur = df.loc[valid, "cm2_forecast"].sum() if valid.any() else np.nan

# Real margin adjustment
cm2p_fore = df["cm2pct_forecast"].fillna(0)
cm2p_act  = df["cm2pct_actual"].fillna(0)
df["cm2pct_real"] = [f + (a-f)*2 if (a-f)<0 else f+(a-f) for f,a in zip(cm2p_fore,cm2p_act)]
valid2 = df["contract_value"].notna()
weighted_real_pct = (df.loc[valid2,"cm2pct_real"]*df.loc[valid2,"contract_value"]).sum()/df.loc[valid2,"contract_value"].sum()
total_real_eur = df["cm2_actual"].sum()

# ------------------------------------------------------------
# KPIs
# ------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Projects", df["project_id"].nunique())
with col2: st.metric("Contract Value Σ (EUR)", f"{total_contract:,.0f}")
with col3: st.metric("Cash Received Σ (EUR)", f"{total_cash:,.0f}")
with col4: st.metric("Compounded CM2% (Forecast)", f"{weighted_fore_pct:,.1f}%", delta=f"{total_fore_eur:,.0f} €")
with col5: st.metric("Real Compounded CM2% (Actual)", f"{weighted_real_pct:,.1f}%", delta=f"{total_real_eur:,.0f} €")

# ------------------------------------------------------------
# Service table build
# ------------------------------------------------------------
service_blocks = ["tpm","cpm","eng","qa_qc_exp","hse","constr","com","man","proc"]
svc_rows = []
for s in service_blocks:
    for field in ["budget","forecast","actual","h_o","b_o","delay"]:
        if f"{s}_{field}" not in df.columns:
            df[f"{s}_{field}"] = np.nan
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
svc["inflation"] = np.where(svc["budget"]>0, svc["actual"]/svc["budget"], np.nan)

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tabs = st.tabs(["Overview", "Internal Services Metrics", "Margin Bridge", "Forecast Accuracy", "Overrun Heatmap", "Drivers"])

# ------------------------------------------------------------
# 1️⃣ Overview
# ------------------------------------------------------------
with tabs[0]:
    st.subheader("Portfolio Overview")

    # Correlation heatmap (restored)
    base_cols = ["contract_value","cash_received","cm2_forecast","cm2_actual","total_penalties"]
    df_num = df[[c for c in base_cols if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    if not df_num.empty:
        corr = df_num.corr("spearman")
        fig = px.imshow(corr, color_continuous_scale="tealrose", aspect="auto", title="Correlation heatmap")
        st.plotly_chart(fig, use_container_width=True)

    # Bubble chart
    st.subheader("Contract Value vs Penalties (bubble = CM2% Forecast)")
    fig = px.scatter(
        df,
        x="contract_value",
        y="total_penalties",
        size="cm2pct_forecast",
        color="country" if "country" in df.columns else None,
        hover_data=["project_id","customer"] if "customer" in df.columns else ["project_id"],
        color_discrete_sequence=px.colors.qualitative.Vivid,
        title="Penalty distribution across projects"
    )
    fig.update_traces(marker=dict(line=dict(width=0.5, color="rgba(0,0,0,0.4)")))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# 2️⃣ Internal Services Metrics
# ------------------------------------------------------------
with tabs[1]:
    st.subheader("Internal Services Metrics")

    # HSE, TPM, CPM, QA/QC/Exp have N/A for delays
    mask_na = svc["service"].isin(["HSE","TPM","CPM","QA_QC_EXP"])
    svc.loc[mask_na, "delay"] = np.nan

    svc_agg = svc.groupby("service").agg(
        projects=("project_id","nunique"),
        budget=("budget","sum"),
        actual=("actual","sum"),
        forecast=("forecast","sum"),
        h_overruns=("h_o","sum"),
        b_overruns=("b_o","sum"),
        delays=("delay","sum"),
        median_inflation=("inflation","median")
    ).reset_index()
    svc_agg["inflation_factor"] = svc_agg["actual"]/svc_agg["budget"]

    pretty = {
        "TPM":"TPM","CPM":"CPM","ENG":"Engineering","QA_QC_EXP":"QA/QC/Exp",
        "HSE":"HSE","CONSTR":"Construction","COM":"Commissioning",
        "MAN":"Manufacturing","PROC":"Procurement"
    }
    svc_agg["Service"] = svc_agg["service"].map(pretty)

    # Fill N/A text
    svc_agg[["h_overruns","b_overruns","delays"]] = svc_agg[["h_overruns","b_overruns","delays"]].fillna("n/a")

    st.dataframe(
        svc_agg[["Service","projects","budget","actual","forecast","h_overruns","b_overruns","delays","median_inflation","inflation_factor"]],
        use_container_width=True
    )

    color_seq = px.colors.sequential.Tealgrn
    fig = px.bar(
        svc_agg, x="Service", y=["budget","actual","forecast"], barmode="group",
        title="Budget vs Actual vs Forecast (hours)", color_discrete_sequence=color_seq
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(
        svc_agg, x="Service", y="inflation_factor",
        color="Service", color_discrete_sequence=color_seq,
        title="Inflation factor (Actual/Budget)"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------
# 3️⃣ Margin Bridge
# ------------------------------------------------------------
with tabs[2]:
    st.subheader("Margin Δ (Forecast → Actual)")
    df["margin_delta"] = df["cm2_actual"] - df["cm2_forecast"]
    bridge = df[["project_id","customer","contract_value","cm2_forecast","cm2_actual","margin_delta"]].copy()
    fig = px.bar(
        bridge, x="project_id", y="margin_delta", color=np.where(bridge["margin_delta"]>0,"Gain","Loss"),
        color_discrete_sequence=["#7fc8a9","#e07a5f"], hover_data=["customer"],
        title="Margin difference per project (EUR)"
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# 4️⃣ Forecast Accuracy
# ------------------------------------------------------------
with tabs[3]:
    st.subheader("Forecast Accuracy by Service")
    svc["forecast_accuracy"] = 1 - abs((svc["forecast"] - svc["actual"]) / svc["budget"].replace(0,np.nan))
    acc = svc.groupby("service")["forecast_accuracy"].mean().reset_index()
    acc["Service"] = acc["service"].map(pretty)
    fig = px.bar(
        acc, x="Service", y="forecast_accuracy",
        color="Service", color_discrete_sequence=px.colors.sequential.Tealgrn,
        title="Average forecast accuracy (1 - |Δ|/Budget)"
    )
    fig.update_yaxes(range=[0,1])
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# 5️⃣ Overrun Heatmap
# ------------------------------------------------------------
with tabs[4]:
    st.subheader("Overrun & Delay density heatmap")
    heat = svc.groupby("service")[["h_o","b_o","delay"]].mean().reset_index()
    heat["Service"] = heat["service"].map(pretty)
    melt = heat.melt(id_vars="Service", var_name="Type", value_name="Rate")
    fig = px.density_heatmap(
        melt, x="Type", y="Service", z="Rate",
        color_continuous_scale="tealrose", title="Average overrun rate by service"
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# 6️⃣ Drivers
# ------------------------------------------------------------
with tabs[5]:
    st.subheader("Drivers of CM2% drop (logistic model)")
    df["cm2_drop"] = (df["cm2pct_actual"] < df["cm2pct_forecast"]).astype(int)
    X = df[["total_o","total_delays","check_v"]].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["cm2_drop"]

    if y.sum() > 1:
        pipe = Pipeline([("scaler",StandardScaler()),("clf",LogisticRegression(max_iter=200))])
        pipe.fit(X,y)
        coef = pipe.named_steps["clf"].coef_[0]
        ors = np.exp(coef)
        coef_df = pd.DataFrame({"Feature":X.columns,"Odds_Ratio":ors}).sort_values("Odds_Ratio",ascending=False)
        fig = px.bar(
            coef_df, x="Feature", y="Odds_Ratio",
            color="Feature", color_discrete_sequence=px.colors.qualitative.Vivid,
            title="Odds ratios for CM2% drop (↑ = higher risk)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough variance to fit the logistic model.")
