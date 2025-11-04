# deep_margins/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config("PSS Project Analytics", layout="wide")

# ============================ HELPERS ============================
def safe_col(df, name_like):
    for c in df.columns:
        if name_like.lower() in c.lower():
            return c
    return None

def load_excel(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()

def format_eur(x):
    try:
        return f"{x:,.0f}".replace(",", " ")
    except:
        return "-"

# ============================ SIDEBAR ============================
st.sidebar.header("Upload Excel (.xlsx)")
uploaded_file = st.sidebar.file_uploader(" ", type=["xlsx"])
if uploaded_file is None:
    st.stop()

df = load_excel(uploaded_file)
if df.empty:
    st.stop()

# ============================ COLUMN DETECTION ============================
col_country = safe_col(df, "country")
col_customer = safe_col(df, "customer")
col_cv = safe_col(df, "contract")
col_cash = safe_col(df, "cash")
col_cm2_bud_pct = safe_col(df, "cm2pct_budget")
col_cm2_fore_pct = safe_col(df, "cm2pct_forecast")
col_cm2_bud_eur = safe_col(df, "cm2_budget")
col_cm2_fore_eur = safe_col(df, "cm2_forecast")
col_penalties = safe_col(df, "penalt")
col_service = safe_col(df, "service")
col_h_overrun = safe_col(df, "hours_over")
col_b_overrun = safe_col(df, "budget_over")
col_delay = safe_col(df, "delay")

# ============================ FILTERS ============================
countries = sorted(df[col_country].dropna().unique())
customers = sorted(df[col_customer].dropna().unique())

sel_countries = st.sidebar.multiselect("Countries", countries, default=countries)
sel_customers = st.sidebar.multiselect("Customers", customers, default=customers)

df = df[df[col_country].isin(sel_countries)]
df = df[df[col_customer].isin(sel_customers)]

st.sidebar.success(f"{len(df)} projects loaded after filters")

# ============================ METRICS ============================
n_projects = df.shape[0]
sum_contract = df[col_cv].sum()
sum_cash = df[col_cash].sum()
cm2_forecast_eur = df[col_cm2_fore_eur].sum()
cm2_budget_eur = df[col_cm2_bud_eur].sum()

cm2_dev_eur = cm2_forecast_eur - cm2_budget_eur
cm2_dev_pct = (cm2_forecast_eur / cm2_budget_eur - 1) * 100 if cm2_budget_eur else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Projects", n_projects)
col2.metric("Σ Contract Value (EUR)", format_eur(sum_contract))
col3.metric("Σ Cash Received (EUR)", format_eur(sum_cash))
col4.metric("Forecast vs Budget CM2%", f"{cm2_dev_pct:.1f}%", format_eur(cm2_dev_eur))

st.markdown("---")

# ============================ OVERVIEW SCATTER ============================
st.header("Margin Scatter")

y_choice = st.radio("Y-axis:", ["CM2% Forecast", "CM2 Δ (EUR)"], horizontal=True)

y_col = col_cm2_fore_pct if y_choice == "CM2% Forecast" else col_cm2_fore_eur
fig_scatter = px.scatter(
    df,
    x=col_cv,
    y=y_col,
    color=col_country,
    hover_data=[col_customer, col_country, col_cm2_bud_pct, col_cm2_fore_pct],
    title=f"Contract Value vs {y_choice}",
)
fig_scatter.update_traces(marker=dict(size=10, opacity=0.8))
st.plotly_chart(fig_scatter, use_container_width=True)

# ============================ BUBBLE CHART ============================
st.header("Penalty Distribution Bubble Chart")

# size = number of penalties (nonzero)
df["bubble_size"] = df[col_penalties].fillna(0).astype(float)
df["bubble_shape"] = np.where(df["bubble_size"] == 0, "square", "circle")

fig_bubble = px.scatter(
    df,
    x=col_cv,
    y=col_cm2_fore_pct,
    color=col_country,
    size="bubble_size",
    hover_data=[col_customer, col_country, col_penalties],
    title="Contract Value vs CM2% Forecast (bubble = penalties)",
)
fig_bubble.update_traces(
    marker=dict(opacity=0.75, sizemode="area"),
)
st.plotly_chart(fig_bubble, use_container_width=True)

# ============================ INTERNAL SERVICE METRICS ============================
st.header("Internal Services Metrics")

svc_df = df.groupby(col_service, dropna=False).agg({
    "contract_value": "count",
    "budget": "sum" if "budget" in df.columns else "sum",
}).reset_index()

table_cols = [
    col_service,
    "projects",
    "budget",
    "forecast",
    "hours_overruns",
    "budget_overruns",
    "delays",
    "median_inflation",
    "inflation_factor"
]
if all(c in df.columns for c in table_cols):
    st.dataframe(df[table_cols])

# Exclude man & proc for plots
excluded = ["Manufacturing", "Procurement"]
df_plot = df[~df[col_service].isin(excluded)]

fig_bar = px.bar(
    df_plot,
    x=col_service,
    y=[col_cm2_bud_eur, col_cm2_fore_eur],
    barmode="group",
    title="Budget vs Forecast CM2 by Service",
)
st.plotly_chart(fig_bar, use_container_width=True)

# ============================ FORECAST ACCURACY ============================
st.header("Forecast Accuracy by Service")

if all(c in df.columns for c in [col_service, col_cm2_bud_eur, col_cm2_fore_eur]):
    acc_df = (
        df[~df[col_service].isin(excluded)]
        .groupby(col_service)
        .apply(lambda x: 1 - abs(x[col_cm2_fore_eur] - x[col_cm2_bud_eur]).sum() / x[col_cm2_bud_eur].sum())
        .reset_index(name="forecast_accuracy")
    )
    fig_acc = px.bar(
        acc_df,
        x=col_service,
        y="forecast_accuracy",
        color=col_service,
        title="Average Forecast Accuracy (1 - |Δ| / Budget)",
    )
    fig_acc.update_layout(showlegend=False)
    st.plotly_chart(fig_acc, use_container_width=True)

# ============================ MARGIN BRIDGE ============================
st.header("CM2% Real Deviation vs EUR Delta")

if all(c in df.columns for c in [col_cm2_bud_pct, col_cm2_fore_pct, col_cm2_bud_eur, col_cm2_fore_eur]):
    df["real_cm2_dev"] = df[col_cm2_fore_pct] - df[col_cm2_bud_pct]
    df["cm2_delta_eur"] = df[col_cm2_fore_eur] - df[col_cm2_bud_eur]

    fig_bridge = px.bar(
        df,
        x=col_customer,
        y="real_cm2_dev",
        color=col_country,
        hover_data=["cm2_delta_eur"],
        title="Budget vs Forecast CM2% Deviation (hover for EUR)",
    )
    st.plotly_chart(fig_bridge, use_container_width=True)
else:
    st.warning("Missing columns for CM2 bridge.")

# ============================ HEATMAP ============================
st.header("Overrun & Delay density heatmap")

heat_vars = [v for v in [col_h_overrun, col_b_overrun, col_delay] if v in df.columns]
if col_service and heat_vars:
    melt_df = df.melt(
        id_vars=[col_service],
        value_vars=heat_vars,
        var_name="metric",
        value_name="value"
    ).dropna()
    melt_df["rate"] = (melt_df["value"] > 0).astype(int)
    heat_data = melt_df.groupby([col_service, "metric"])["rate"].mean().reset_index()
    fig_heat = px.density_heatmap(
        heat_data,
        x="metric",
        y=col_service,
        z="rate",
        color_continuous_scale="RdYlGn_r",
        title="Average Overrun Rate by Service",
    )
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("No overrun/delay data available.")
