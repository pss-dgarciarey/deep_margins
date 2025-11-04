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
    """Add top-right PNG export button."""
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
    st.info("Upload your latest 'Project List Main.xlsx'")
    st.stop()

df = pd.read_excel(uploaded, sheet_name=0, header=0)
df.columns = [normalize(c) for c in df.columns]

# Filters
countries = sorted(df["country"].dropna().unique()) if "country" in df.columns else []
if countries:
    selected_countries = st.sidebar.multiselect("Countries:", countries, default=countries)
    df = df[df["country"].isin(selected_countries)]

customers = sorted(df["customer"].dropna().unique()) if "customer" in df.columns else []
if customers:
    selected_customers = st.sidebar.multiselect("Customers:", customers, default=customers)
    df = df[df["customer"].isin(selected_customers)]

st.sidebar.success(f"✅ {len(df)} projects loaded after filters")

# ------------------------------------------------------------
# Derived fields
# ------------------------------------------------------------
num_cols = [
    "contract_value", "cash_received",
    "cm2_forecast", "cm2_actual",
    "cm2pct_forecast", "cm2pct_actual",
    "total_penalties", "total_o", "total_delays", "check_v",
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

total_contract = df.get("contract_value", pd.Series(dtype=float)).sum()
total_cash = df.get("cash_received", pd.Series(dtype=float)).sum()

valid = df["contract_value"].notna() & df["cm2_forecast"].notna()
weighted_fore_pct = (
    (df.loc[valid, "cm2_forecast"].sum() / df.loc[valid, "contract_value"].sum()) * 100
    if valid.any()
    else np.nan
)
total_fore_eur = df.loc[valid, "cm2_forecast"].sum() if valid.any() else np.nan

# --- Fixed safe CM2% real calculation ---
if {"cm2pct_forecast", "cm2pct_actual"} <= set(df.columns):
    df["cm2pct_forecast"] = pd.to_numeric(df["cm2pct_forecast"], errors="coerce").fillna(0)
    df["cm2pct_actual"] = pd.to_numeric(df["cm2pct_actual"], errors="coerce").fillna(0)

    def _real_cm2(row):
        f, a = row["cm2pct_forecast"], row["cm2pct_actual"]
        delta = a - f
        return f + (delta * 2 if delta < 0 else delta)

    df["cm2pct_real"] = df.apply(_real_cm2, axis=1)
else:
    df["cm2pct_real"] = np.nan

valid2 = df.get("contract_value").notna() if "contract_value" in df.columns else pd.Series([], dtype=bool)
weighted_real_pct = (
    (df.loc[valid2, "cm2pct_real"] * df.loc[valid2, "contract_value"]).sum()
    / df.loc[valid2, "contract_value"].sum()
    if valid2.any()
    else np.nan
)
total_real_eur = df.get("cm2_actual", pd.Series(dtype=float)).sum()

# ------------------------------------------------------------
# KPIs
# ------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Projects", df["project_id"].nunique() if "project_id" in df.columns else len(df))
with col2:
    st.metric("Contract Value Σ (EUR)", f"{total_contract:,.0f}")
with col3:
    st.metric("Cash Received Σ (EUR)", f"{total_cash:,.0f}")
with col4:
    st.metric("Compounded CM2% (Forecast)", f"{weighted_fore_pct:,.1f}%", delta=f"{(total_fore_eur or 0):,.0f} €")
with col5:
    st.metric("Real Compounded CM2% (Actual)", f"{weighted_real_pct:,.1f}%", delta=f"{(total_real_eur or 0):,.0f} €")

# ------------------------------------------------------------
# Build service data
# ------------------------------------------------------------
service_blocks = ["tpm", "cpm", "eng", "qa_qc_exp", "hse", "constr", "com", "man", "proc"]
svc_rows = []
for s in service_blocks:
    for field in ["budget", "forecast", "actual", "h_o", "b_o", "delay"]:
        col = f"{s}_{field}"
        if col not in df.columns:
            df[col] = np.nan
    for _, r in df.iterrows():
        svc_rows.append(
            {
                "project_id": r.get("project_id"),
                "service": s.upper(),
                "budget": safe_num(r[f"{s}_budget"]),
                "forecast": safe_num(r[f"{s}_forecast"]),
                "actual": safe_num(r[f"{s}_actual"]),
                "h_o": safe_int(r.get(f"{s}_h_o")),
                "b_o": safe_int(r.get(f"{s}_b_o")),
                "delay": safe_int(r.get(f"{s}_delay")),
            }
        )

svc = pd.DataFrame(svc_rows)
svc["inflation"] = np.where(svc["budget"] > 0, svc["actual"] / svc["budget"], np.nan)

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tabs = st.tabs(
    ["Overview", "Internal Services Metrics", "Margin Bridge", "Forecast Accuracy", "Overrun Heatmap", "Drivers"]
)

# ------------------------------------------------------------
# 1️⃣ Overview (fixed visuals)
# ------------------------------------------------------------
with tabs[0]:
    st.subheader("Portfolio Overview")

    # Correlation heatmap
    base_cols = [c for c in ["contract_value", "cash_received", "cm2_forecast", "cm2_actual", "total_penalties"] if c in df.columns]
    df_num = df[base_cols].apply(pd.to_numeric, errors="coerce") if base_cols else pd.DataFrame()
    if not df_num.empty:
        corr = df_num.corr("spearman")
        fig = px.imshow(corr, color_continuous_scale="tealrose", aspect="auto", title="Correlation heatmap")
        st.plotly_chart(fig, use_container_width=True, config=plotly_config("correlation_heatmap"))
    else:
        st.info("No numeric fields available for correlation.")

    # =========================
    # Bubble chart (fixed)
    # =========================
    st.subheader("Contract Value vs CM2% Forecast (bubble = penalties)")
    df_bubble = df.copy()
    for col in ["contract_value", "cm2pct_forecast", "total_penalties"]:
        if col in df_bubble.columns:
            df_bubble[col] = pd.to_numeric(df_bubble[col], errors="coerce").fillna(0)

    df_bubble = df_bubble.dropna(subset=["contract_value", "cm2pct_forecast"])
    if not df_bubble.empty:
        df_bubble["total_penalties"] = df_bubble["total_penalties"].clip(lower=0)
        df_bubble["has_penalty"] = df_bubble["total_penalties"] > 0

        fig = px.scatter(
            df_bubble,
            x="contract_value",
            y="cm2pct_forecast",
            size="total_penalties",
            color="country" if "country" in df_bubble.columns else None,
            hover_data=[c for c in ["project_id", "customer", "total_penalties", "cm2pct_forecast"] if c in df_bubble.columns],
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Penalty-weighted forecast margin distribution",
            size_max=50
        )

        # Different shapes for zero-penalty projects
        fig.update_traces(
            marker=dict(
                symbol=[
                    "square" if not p else "circle" for p in df_bubble["has_penalty"]
                ],
                line=dict(width=0.4, color="rgba(0,0,0,0.3)")
            )
        )

        fig.update_layout(
            xaxis_title="Contract Value (EUR)",
            yaxis_title="CM2% Forecast",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True, config=plotly_config("penalty_bubble"))
    else:
        st.warning("No valid numeric rows for the bubble chart after cleaning.")

    # =========================
    # Margin scatter (fixed)
    # =========================
    st.subheader("Margin Scatter")
    mode = st.radio(
        "Y-axis:",
        ["CM2% Forecast", "CM2 Forecast (EUR)"],
        horizontal=True,
        key="margin_mode",
    )
    df_scatter = df.copy()
    if mode == "CM2% Forecast":
        yaxis = "cm2pct_forecast"
        title = "Contract Value vs CM2% Forecast"
    else:
        yaxis = "cm2_forecast"
        title = "Contract Value vs CM2 Forecast (EUR)"

    if yaxis in df_scatter.columns:
        df_scatter[yaxis] = pd.to_numeric(df_scatter[yaxis], errors="coerce")
        fig2 = px.scatter(
            df_scatter,
            x="contract_value",
            y=yaxis,
            color="country" if "country" in df_scatter.columns else None,
            hover_data=[c for c in ["project_id", "customer"] if c in df_scatter.columns],
            color_discrete_sequence=px.colors.qualitative.Set2,
            title=title,
        )
        fig2.update_layout(
            xaxis_title="Contract Value (EUR)",
            yaxis_title=yaxis.replace("_", " ").title(),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig2, use_container_width=True, config=plotly_config("margin_scatter"))
    else:
        st.info("Required fields not present for the margin scatter.")

# ------------------------------------------------------------
# 2️⃣ Internal Services Metrics
# ------------------------------------------------------------
with tabs[1]:
    st.subheader("Internal Services Metrics")

    na_delay_services = {"TPM", "CPM", "HSE", "QA_QC_EXP"}
    svc.loc[svc["service"].isin(na_delay_services), "delay"] = np.nan

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
    svc_agg["inflation_factor"] = svc_agg["actual"] / svc_agg["budget"]

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
    svc_agg["Service"] = svc_agg["service"].map(pretty)

    svc_agg.loc[svc_agg["service"].isin(na_delay_services), "delays"] = np.nan
    svc_view = svc_agg.copy()
    svc_view[["h_overruns", "b_overruns", "delays"]] = svc_view[["h_overruns", "b_overruns", "delays"]].where(
        svc_view[["h_overruns", "b_overruns", "delays"]].notna(), "n/a"
    )

    st.dataframe(
        svc_view[
            [
                "Service",
                "projects",
                "budget",
                "actual",
                "forecast",
                "h_overruns",
                "b_overruns",
                "delays",
                "median_inflation",
                "inflation_factor",
            ]
        ],
        use_container_width=True,
    )

    color_seq = px.colors.sequential.Tealgrn
    fig3 = px.bar(
        svc_agg,
        x="Service",
        y=["budget", "actual", "forecast"],
        barmode="group",
        title="Budget vs Actual vs Forecast (hours)",
        color_discrete_sequence=color_seq,
    )
    st.plotly_chart(fig3, use_container_width=True, config=plotly_config("services_baf"))

    fig4 = px.bar(
        svc_agg,
        x="Service",
        y="inflation_factor",
        color="Service",
        color_discrete_sequence=color_seq,
        title="Inflation factor (Actual/Budget)",
    )
    st.plotly_chart(fig4, use_container_width=True, config=plotly_config("services_inflation"))

# ------------------------------------------------------------
# 3️⃣ Margin Bridge
# ------------------------------------------------------------
with tabs[2]:
    st.subheader("Margin Δ (Forecast → Actual)")
    if "cm2_actual" in df.columns and "cm2_forecast" in df.columns:
        df["margin_delta"] = df["cm2_actual"] - df["cm2_forecast"]
        bridge = df[
            ["project_id", "customer", "contract_value", "cm2_forecast", "cm2_actual", "margin_delta"]
        ].copy()
        fig5 = px.bar(
            bridge,
            x="project_id",
            y="margin_delta",
            color=np.where(bridge["margin_delta"] > 0, "Gain", "Loss"),
            color_discrete_sequence=["#7fc8a9", "#e07a5f"],
            hover_data=["customer"],
            title="Margin difference per project (EUR)",
        )
        st.plotly_chart(fig5, use_container_width=True, config=plotly_config("margin_bridge"))
    else:
        st.info("Margin fields not found.")

# ------------------------------------------------------------
# 4️⃣ Forecast Accuracy
# ------------------------------------------------------------
with tabs[3]:
    st.subheader("Forecast Accuracy by Service")
    svc["forecast_accuracy"] = 1 - abs((svc["forecast"] - svc["actual"]) / svc["budget"].replace(0, np.nan))
    acc = svc.groupby("service")["forecast_accuracy"].mean().reset_index()
    acc["Service"] = acc["service"].map(pretty)
    fig6 = px.bar(
        acc,
        x="Service",
        y="forecast_accuracy",
        color="Service",
        color_discrete_sequence=px.colors.sequential.Tealgrn,
        title="Average forecast accuracy (1 - |Δ|/Budget)",
    )
    fig6.update_yaxes(range=[0, 1])
    st.plotly_chart(fig6, use_container_width=True, config=plotly_config("forecast_accuracy"))

# ------------------------------------------------------------
# 5️⃣ Overrun Heatmap
# ------------------------------------------------------------
with tabs[4]:
    st.subheader("Overrun & Delay density heatmap")
    heat = svc.groupby("service")[["h_o", "b_o", "delay"]].mean().reset_index()
    heat["Service"] = heat["service"].map(pretty)
    melt = heat.melt(id_vars="Service", var_name="Type", value_name="Rate")
    fig7 = px.density_heatmap(
        melt,
        x="Type",
        y="Service",
        z="Rate",
        color_continuous_scale="tealrose",
        title="Average overrun rate by service",
    )
    st.plotly_chart(fig7, use_container_width=True, config=plotly_config("overrun_heatmap"))

# ------------------------------------------------------------
# 6️⃣ Drivers
# ------------------------------------------------------------
with tabs[5]:
    st.subheader("Drivers of CM2% drop (logistic model)")
    if {"cm2pct_actual", "cm2pct_forecast"}.issubset(df.columns):
        df["cm2_drop"] = (df["cm2pct_actual"] < df["cm2pct_forecast"]).astype(int)
        X = df[
            [c for c in ["total_o", "total_delays", "check_v"] if c in df.columns]
        ].apply(pd.to_numeric, errors="coerce").fillna(0)
        y = df["cm2_drop"]
        if y.sum() > 1 and X.shape[1] >= 1:
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
            pipe.fit(X, y)
            coefs = pipe.named_steps["clf"].coef_[0]
            ors = np.exp(coefs)
            coef_df = pd.DataFrame({"Feature": X.columns, "Odds_Ratio": ors}).sort_values(
                "Odds_Ratio", ascending=False
            )
            fig8 = px.bar(
                coef_df,
                x="Feature",
                y="Odds_Ratio",
                color="Feature",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Odds ratios for CM2% drop (↑ = higher risk)",
            )
            st.plotly_chart(fig8, use_container_width=True, config=plotly_config("drivers_logit"))
        else:
            st.info("Not enough variance to fit the logistic model.")
    else:
        st.info("CM2% fields not found for driver model.")
