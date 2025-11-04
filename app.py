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

# ------------------------------------------------------------
# Load + Filters
# ------------------------------------------------------------
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
if not uploaded:
    st.info("Upload your 'Project List Main.xlsx' file.")
    st.stop()

df = pd.read_excel(uploaded)
df.columns = [normalize(c) for c in df.columns]

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
# Derived fields
# ------------------------------------------------------------
for col in [
    "contract_value", "cash_received", "cm2_forecast", "cm2_actual",
    "cm2pct_forecast", "cm2pct_actual", "total_penalties", "total_o",
    "total_delays", "check_v"
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

total_contract = df["contract_value"].sum()
total_cash = df["cash_received"].sum()
weighted_fore_pct = (df["cm2_forecast"].sum() / total_contract * 100) if total_contract else 0
weighted_real_pct = (df["cm2_actual"].sum() / total_contract * 100) if total_contract else 0

# ------------------------------------------------------------
# KPIs
# ------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Projects", len(df))
with col2:
    st.metric("Contract Value Σ (EUR)", f"{total_contract:,.0f}")
with col3:
    st.metric("Cash Received Σ (EUR)", f"{total_cash:,.0f}")
with col4:
    st.metric("Compounded CM2% (Forecast)", f"{weighted_fore_pct:,.1f}%")
with col5:
    st.metric("Real Compounded CM2% (Actual)", f"{weighted_real_pct:,.1f}%")

# ------------------------------------------------------------
# Service data setup
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

pretty = {
    "TPM": "TPM",
    "CPM": "CPM",
    "ENG": "Engineering",
    "QA_QC_EXP": "QA/QC/Exp",
    "HSE": "HSE",
    "CONSTR": "Construction",
    "COM": "Commissioning",
    "MAN": "Manufacturing",
    "PROC": "Procurement",
}

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tabs = st.tabs([
    "Overview", "Internal Services Metrics",
    "Margin Bridge", "Forecast Accuracy", "Overrun Heatmap", "Drivers"
])

# ------------------------------------------------------------
# 1️⃣ Overview
# ------------------------------------------------------------
with tabs[0]:
    st.subheader("Portfolio Overview")

    heat_cols = ["contract_value", "cash_received", "cm2_forecast", "cm2_actual", "total_penalties", "total_o", "total_delays"]
    df_num = df[heat_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if not df_num.empty and df_num.shape[1] > 1:
        corr = df_num.corr("spearman")
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="tealrose", title="Correlation heatmap")
        st.plotly_chart(fig, use_container_width=True, config=plotly_config("correlation_heatmap"))

    # Margin Bridge (now real CM2% Δ vs EUR hover)
    st.subheader("CM2% Real Deviation vs EUR Delta")
    df["real_cm2_dev"] = df["cm2pct_actual"] - df["cm2pct_forecast"]
    df["cm2_delta_eur"] = df["cm2_actual"] - df["cm2_forecast"]
    bridge = df[["project_id", "customer", "real_cm2_dev", "cm2_delta_eur"]].dropna(subset=["real_cm2_dev"])
    if not bridge.empty:
        fig = px.bar(
            bridge, x="project_id", y="real_cm2_dev",
            color=np.where(bridge["real_cm2_dev"] > 0, "Positive", "Negative"),
            hover_data=["customer", "cm2_delta_eur"],
            color_discrete_sequence=["#7fc8a9", "#e07a5f"],
            title="CM2% Actual vs Forecast (hover for € delta)"
        )
        fig.update_layout(xaxis_title="Project", yaxis_title="CM2% Δ (Actual - Forecast)")
        st.plotly_chart(fig, use_container_width=True, config=plotly_config("margin_bridge"))

# ------------------------------------------------------------
# 2️⃣ Internal Services Metrics
# ------------------------------------------------------------
with tabs[1]:
    st.subheader("Internal Services Metrics")

    # Full table (includes Man & Proc)
    svc_agg = svc.groupby("service").agg(
        projects=("project_id", "nunique"),
        budget=("budget", "sum"),
        actual=("actual", "sum"),
        forecast=("forecast", "sum"),
        h_overruns=("h_o", "sum"),
        b_overruns=("b_o", "sum"),
        delays=("delay", "sum"),
        median_inflation=("inflation", "median")
    ).reset_index()
    svc_agg["Service"] = svc_agg["service"].map(pretty)
    svc_agg["inflation_factor"] = svc_agg["actual"] / svc_agg["budget"]
    st.dataframe(svc_agg, use_container_width=True)

    # Chart (exclude Man & Proc)
    svc_chart = svc_agg[~svc_agg["service"].isin(["MAN", "PROC"])]
    fig3 = px.bar(
        svc_chart, x="Service", y=["budget", "actual", "forecast"],
        barmode="group", color_discrete_sequence=px.colors.sequential.Tealgrn,
        title="Budget vs Actual vs Forecast (hours)"
    )
    st.plotly_chart(fig3, use_container_width=True, config=plotly_config("services_baf"))

# ------------------------------------------------------------
# 3️⃣ Forecast Accuracy
# ------------------------------------------------------------
with tabs[2]:
    st.subheader("Forecast Accuracy by Service")
    svc["forecast_accuracy"] = 1 - abs((svc["forecast"] - svc["actual"]) / svc["budget"].replace(0, np.nan))
    acc = svc.groupby("service")["forecast_accuracy"].mean().reset_index()
    acc["Service"] = acc["service"].map(pretty)
    acc_chart = acc[~acc["service"].isin(["MAN", "PROC"])]
    fig5 = px.bar(acc_chart, x="Service", y="forecast_accuracy", color_discrete_sequence=px.colors.sequential.Tealgrn)
    fig5.update_yaxes(range=[0, 1])
    st.plotly_chart(fig5, use_container_width=True, config=plotly_config("forecast_accuracy"))

# ------------------------------------------------------------
# 4️⃣ Overrun Heatmap
# ------------------------------------------------------------
with tabs[3]:
    st.subheader("Overrun & Delay density heatmap")
    svc_filtered = svc[~svc["service"].isin(["MAN", "PROC"])]
    heat = svc_filtered.groupby("service")[["h_o", "b_o", "delay"]].mean().reset_index()
    heat["Service"] = heat["service"].map(pretty)
    melt = heat.melt(id_vars="Service", var_name="Type", value_name="Rate")
    fig6 = px.density_heatmap(melt, x="Type", y="Service", z="Rate", color_continuous_scale="tealrose")
    st.plotly_chart(fig6, use_container_width=True, config=plotly_config("overrun_heatmap"))

# ------------------------------------------------------------
# 5️⃣ Drivers (Fixed)
# ------------------------------------------------------------
with tabs[4]:
    st.subheader("Drivers of CM2% Drop (logistic model)")

    df["cm2_drop"] = (df["cm2pct_actual"] < df["cm2pct_forecast"]).astype(int)
    predictors = [c for c in ["total_o", "total_delays", "check_v"] if c in df.columns]
    if not predictors:
        st.info("No numeric predictors available.")
    else:
        X = df[predictors].apply(pd.to_numeric, errors="coerce").fillna(0)
        y = df["cm2_drop"]
        if y.nunique() < 2:
            st.warning("Insufficient variation in target variable (all projects up or down).")
        else:
            model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=300))])
            model.fit(X, y)
            odds = np.exp(model.named_steps["clf"].coef_[0])
            coef_df = pd.DataFrame({"Feature": X.columns, "Odds_Ratio": odds}).sort_values("Odds_Ratio", ascending=False)
            fig7 = px.bar(coef_df, x="Feature", y="Odds_Ratio", color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig7, use_container_width=True, config=plotly_config("drivers_logit"))
