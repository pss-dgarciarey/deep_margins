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
# Derived fields
# ------------------------------------------------------------
for col in [
    "contract_value", "cash_received", "cm2_forecast", "cm2_actual",
    "cm2pct_forecast", "cm2pct_actual", "cm2pct_budget",
    "cm2_budget", "total_penalties", "total_o", "total_delays", "check_v"
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Compounded metrics
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
# Services normalization
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

# Pretty mapping
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

    # Correlation Heatmap
    heat_cols = [
        "contract_value", "cash_received",
        "cm2_forecast", "cm2_actual",
        "total_penalties", "total_o", "total_delays"
    ]
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
            hover_data=["project_id", "customer", "total_penalties", "cm2pct_forecast"],
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
    mode = st.radio("Y-axis:", ["CM2% Forecast", "CM2 Forecast (EUR)"], horizontal=True)
    yaxis = "cm2pct_forecast" if mode == "CM2% Forecast" else "cm2_forecast"
    df[yaxis] = pd.to_numeric(df[yaxis], errors="coerce")
    fig2 = px.scatter(
        df, x="contract_value", y=yaxis,
        color="country" if "country" in df.columns else None,
        hover_data=["project_id", "customer"],
        color_discrete_sequence=px.colors.qualitative.Set2,
        title=f"Contract Value vs {yaxis.replace('_', ' ').title()}"
    )
    st.plotly_chart(fig2, use_container_width=True, config=plotly_config("margin_scatter"))

# ------------------------------------------------------------
# 2️⃣ Internal Services Metrics
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

    # Table (all services)
    st.dataframe(
        svc_agg[[ 
            "Service", "projects", "budget", "actual", "forecast",
            "h_overruns", "b_overruns", "delays", "median_inflation", "inflation_factor"
        ]],
        use_container_width=True,
    )

    # Chart (exclude Man & Proc)
    svc_chart = svc_agg[~svc_agg["service"].isin(["MAN", "PROC"])]
    fig3 = px.bar(
        svc_chart, x="Service", y=["budget", "actual", "forecast"],
        barmode="group", color_discrete_sequence=px.colors.sequential.Tealgrn,
        title="Budget vs Actual vs Forecast (hours)"
    )
    st.plotly_chart(fig3, use_container_width=True, config=plotly_config("services_baf"))

# ------------------------------------------------------------
# 3️⃣ Margin Bridge
# ------------------------------------------------------------
with tabs[2]:
    st.subheader("Budget vs Forecast CM2% Deviation (hover = EUR)")
    df["delta_pct"] = df["cm2pct_forecast"] - df.get("cm2pct_budget", 0)
    df["delta_eur"] = df["cm2_forecast"] - df.get("cm2_budget", 0)

    bridge = df[["project_id", "customer", "delta_eur", "delta_pct"]].dropna(subset=["delta_pct"])
    fig4 = px.bar(
        bridge, x="project_id", y="delta_pct",
        color=np.where(bridge["delta_pct"] > 0, "Gain", "Loss"),
        hover_data=["customer", "delta_eur"],
        color_discrete_sequence=["#7fc8a9", "#e07a5f"],
        title="Δ CM2% (Budget vs Forecast)"
    )
    st.plotly_chart(fig4, use_container_width=True, config=plotly_config("margin_bridge"))

# ------------------------------------------------------------
# 4️⃣ Forecast Accuracy
# ------------------------------------------------------------
with tabs[3]:
    st.subheader("Forecast Accuracy by Service")
    svc_filtered = svc[~svc["service"].isin(["MAN", "PROC"])]
    svc_filtered["forecast_accuracy"] = 1 - abs((svc_filtered["forecast"] - svc_filtered["actual"]) / svc_filtered["budget"].replace(0, np.nan))
    acc = svc_filtered.groupby("service")["forecast_accuracy"].mean().reset_index()
    acc["Service"] = acc["service"].map(pretty)

    fig5 = px.bar(
        acc, x="Service", y="forecast_accuracy",
        color_discrete_sequence=px.colors.sequential.Tealgrn,
        title="Average forecast accuracy (1 - |Δ|/Budget)"
    )
    fig5.update_yaxes(range=[0, 1])
    st.plotly_chart(fig5, use_container_width=True, config=plotly_config("forecast_accuracy"))

# ------------------------------------------------------------
# 5️⃣ Overrun Heatmap
# ------------------------------------------------------------
with tabs[4]:
    st.subheader("Overrun & Delay density heatmap")
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
# 6️⃣ Drivers
# ------------------------------------------------------------
with tabs[5]:
    st.subheader("Drivers of CM2% deviation (logistic model)")
    if {"cm2pct_forecast", "cm2pct_budget"}.issubset(df.columns):
        df["cm2_drop"] = (df["cm2pct_forecast"] < df["cm2pct_budget"]).astype(int)
        X = df[[c for c in ["total_o", "total_delays", "check_v"] if c in df.columns]].fillna(0)
        y = df["cm2_drop"]
        if y.sum() > 1 and X.shape[1] >= 1:
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
            pipe.fit(X, y)
            ors = np.exp(pipe.named_steps["clf"].coef_[0])
            coef_df = pd.DataFrame({"Feature": X.columns, "Odds_Ratio": ors}).sort_values("Odds_Ratio", ascending=False)
            fig7 = px.bar(
                coef_df, x="Feature", y="Odds_Ratio",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Odds ratios for CM2% deviation risk (↑ = higher risk)"
            )
            st.plotly_chart(fig7, use_container_width=True, config=plotly_config("drivers_logit"))
        else:
            st.warning("Not enough valid data to fit the model.")
    else:
        st.warning("Missing CM2% budget/forecast columns for driver analysis.")
