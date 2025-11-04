# app.py
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="PSS Project Analytics", layout="wide")

# -----------------------------
# Config / constants
# -----------------------------
RAW_SERVICE_LABELS = [
    "TPM", "CPM", "Eng", "QA/QC/EXP", "Proc", "Man", "HSE", "Constr", "Com", "Contrs"
]
# Map messy labels to clean service keys
SERVICE_KEY_MAP = {
    "tpm": "TPM",
    "cpm": "CPM",
    "eng": "Eng",
    "qa/qc/exp": "QA_QC_EXP",
    "qa_qc_exp": "QA_QC_EXP",
    "proc": "Proc",
    "man": "Man",
    "hse": "HSE",
    "constr": "Constr",
    "contrs": "Constr",  # typo in old files -> treat as Constr
    "com": "Com"
}

SERVICE_FIELDS = ["Budget", "Forecast", "Actual", "H_O", "B_O", "Delay"]

PROJECT_ID_COL = "Project ID"
CUSTOMER_COL = "Customer"
COUNTRY_COL  = "Country"
CONTRACT_VALUE_COL = "Contract Value"
CASH_RECEIVED_COL  = "Cash Received"
PT_COL = "PT (days)"
CM2_BUDGET_COL = "CM2_Budget"
CM2P_BUDGET_COL = "CM2%_Budget"
CM2P_FORECAST_COL = "CM2%_Forecast"
CHECKV_COL = "Check_V"

# -----------------------------
# Utilities
# -----------------------------
def normalize_colname(c: str) -> str:
    """Lowercase, replace separators, collapse spaces, keep % sign explicit."""
    if not isinstance(c, str):
        c = str(c)
    c = c.strip()
    c = c.replace(" ", "_").replace("/", "_").replace("-", "_")
    c = c.replace("%", "pct")
    while "__" in c:
        c = c.replace("__", "_")
    return c.lower()

def denorm_map(cols):
    """Return dict normalized->original for reference."""
    return {normalize_colname(c): c for c in cols}

def coerce_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    phat = successes / n
    denom = 1 + z**2/n
    center = (phat + z**2/(2*n)) / denom
    margin = z * math.sqrt((phat*(1-phat) + z**2/(4*n))/n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

# -----------------------------
# Load / prep
# -----------------------------
@st.cache_data(show_spinner=False)
def load_excel(file) -> pd.DataFrame:
    if isinstance(file, str):
        df = pd.read_excel(file, sheet_name=0, header=0)
    else:
        df = pd.read_excel(file, sheet_name=0, header=0)
    return df

def normalize_dataframe(df_raw: pd.DataFrame):
    colmap = denorm_map(df_raw.columns)
    df = df_raw.copy()
    df.columns = [normalize_colname(c) for c in df.columns]
    return df, colmap

def find_service_columns(df_norm: pd.DataFrame):
    """Return dict: {service_key: {field: normalized_col or None}}"""
    cols = df_norm.columns.tolist()
    result = {}
    for raw in RAW_SERVICE_LABELS:
        key = SERVICE_KEY_MAP[normalize_colname(raw)]
        result[key] = {}
        # try both 'hse_budget' and 'hse_budget' (cover 'hse_budget' & 'hse_budget' identical) + space variants already normalized
        base = normalize_colname(raw)
        for field in SERVICE_FIELDS:
            # primary pattern
            patterns = [
                f"{base}_{normalize_colname(field)}",    # eng_budget
                f"{base}{normalize_colname(field)}",     # engbudget (rare)
            ]
            # specific weirdness: 'HSE Budget' in some files => 'hse_budget' already
            # add explicit alternates for common typos
            if base == "contrs":
                altbase = "constr"
                patterns.append(f"{altbase}_{normalize_colname(field)}")
            if base == "qa_qc_exp":
                patterns.append(f"qa_qc_exp_{normalize_colname(field)}")

            hit = next((p for p in patterns if p in cols), None)
            result[key][field] = hit
    return result

def build_service_long(df_norm: pd.DataFrame, svc_cols):
    rows = []
    for idx, row in df_norm.iterrows():
        proj_id = row.get(normalize_colname(PROJECT_ID_COL))
        customer = row.get(normalize_colname(CUSTOMER_COL))
        country  = row.get(normalize_colname(COUNTRY_COL))

        for svc_key, field_map in svc_cols.items():
            entry = {
                "project_id": proj_id,
                "customer": customer,
                "country": country,
                "service": svc_key
            }
            # Pull values
            for field in SERVICE_FIELDS:
                col = field_map.get(field)
                val = row.get(col) if col else np.nan
                entry[field.lower()] = val
            # Coerce numeric where needed
            for q in ["budget", "forecast", "actual"]:
                val = entry.get(q, np.nan)
                try:
                    entry[q] = float(val)
                except Exception:
                    entry[q] = np.nan

            for q in ["h_o", "b_o", "delay"]:
                val = entry.get(q, 0)
                try:
                    entry[q] = int(pd.to_numeric(val, errors="coerce") or 0)
                except Exception:
                    entry[q] = 0

            # Derived
            entry["inflation_factor"] = (entry["actual"] / entry["budget"]) if entry["budget"] and entry["budget"] != 0 else np.nan
            entry["forecast_overrun"] = int((entry["forecast"] > entry["budget"])) if not pd.isna(entry["forecast"]) and not pd.isna(entry["budget"]) else np.nan
            rows.append(entry)
    svc_long = pd.DataFrame(rows)
    return svc_long

def build_project_flags(svc_long: pd.DataFrame, df_norm: pd.DataFrame):
    # Any flags per project
    agg = svc_long.groupby("project_id").agg(
        any_hours_overrun=("h_o", lambda s: int((s.fillna(0) > 0).any())),
        any_budget_overrun=("b_o", lambda s: int((s.fillna(0) > 0).any())),
        any_delay=("delay", lambda s: int((s.fillna(0) > 0).any()))
    ).reset_index()

    proj = df_norm.copy()
    proj_id_norm = normalize_colname(PROJECT_ID_COL)
    proj = proj.merge(agg, left_on=proj_id_norm, right_on="project_id", how="left")
    proj["any_hours_overrun"]  = proj["any_hours_overrun"].fillna(0).astype(int)
    proj["any_budget_overrun"] = proj["any_budget_overrun"].fillna(0).astype(int)
    proj["any_delay"]          = proj["any_delay"].fillna(0).astype(int)

    # Totals may exist as columns already; keep them if present
    for c in ["total_h_o", "total_b_o", "total_o", "total_delays", "total_penalties"]:
        if c in proj.columns:
            proj[c] = pd.to_numeric(proj[c], errors="coerce")

    return proj

def probability_internal_overrun_given_hours(svc_long: pd.DataFrame):
    df = svc_long.copy()
    df["h_o_bin"] = (df["h_o"].fillna(0) > 0).astype(int)
    df["b_o_bin"] = (df["b_o"].fillna(0) > 0).astype(int)

    # Overall across services/projects
    mask = df["h_o_bin"] == 1
    n = mask.sum()
    if n == 0:
        return dict(p=np.nan, n=0, ci=(np.nan, np.nan))
    k = int((df.loc[mask, "b_o_bin"] == 1).sum())
    p = k / n
    ci = wilson_ci(k, n)
    return dict(p=p, n=n, ci=ci)

def correlation_frame(df_norm: pd.DataFrame):
    # choose numeric columns that matter and won’t leak obvious targets
    keep_like = [
        normalize_colname(CONTRACT_VALUE_COL),
        normalize_colname(CASH_RECEIVED_COL),
        normalize_colname(PT_COL),
        normalize_colname(CM2P_BUDGET_COL),
        normalize_colname(CM2P_FORECAST_COL),
        normalize_colname(CM2_BUDGET_COL),
        normalize_colname(CHECKV_COL),
        "total_h_o","total_b_o","total_o","total_delays","total_penalties",
        "any_hours_overrun","any_budget_overrun","any_delay"
    ]
    cols = [c for c in keep_like if c in df_norm.columns]
    if not cols:
        return pd.DataFrame()
    df_num = df_norm[cols].apply(pd.to_numeric, errors="coerce")
    return df_num

# -----------------------------
# Sidebar / data input
# -----------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
path_hint = st.sidebar.text_input("...or local path", value="")
if uploaded:
    df_raw = load_excel(uploaded)
elif path_hint.strip():
    df_raw = load_excel(path_hint.strip())
else:
    st.info("Upload your `Project List Main.xlsx` to begin.")
    st.stop()

df_norm, colmap = normalize_dataframe(df_raw)
svc_cols = find_service_columns(df_norm)
svc_long = build_service_long(df_norm, svc_cols)
proj_df = build_project_flags(svc_long, df_norm)

# Basic filters
country_list = sorted(proj_df.get(normalize_colname(COUNTRY_COL), pd.Series(dtype=str)).dropna().unique().tolist())
customer_list = sorted(proj_df.get(normalize_colname(CUSTOMER_COL), pd.Series(dtype=str)).dropna().unique().tolist())
with st.sidebar:
    st.header("Filters")
    ctry_sel = st.multiselect("Country", options=country_list, default=country_list)
    cust_sel = st.multiselect("Customer", options=customer_list, default=customer_list)

def apply_filters(df):
    out = df.copy()
    if country_list:
        out = out[out.get(normalize_colname(COUNTRY_COL)).isin(ctry_sel)]
    if customer_list:
        out = out[out.get(normalize_colname(CUSTOMER_COL)).isin(cust_sel)]
    return out

proj_f = apply_filters(proj_df)
svc_f = svc_long[svc_long["project_id"].isin(proj_f["project_id"])]

# -----------------------------
# TOP KPIs
# -----------------------------
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Projects", proj_f["project_id"].nunique())
with col2:
    if normalize_colname(CONTRACT_VALUE_COL) in proj_f.columns:
        st.metric("Σ Contract Value (EUR)", f"{pd.to_numeric(proj_f[normalize_colname(CONTRACT_VALUE_COL)], errors='coerce').sum():,.0f}")
with col3:
    if normalize_colname(CASH_RECEIVED_COL) in proj_f.columns:
        st.metric("Σ Cash Received (EUR)", f"{pd.to_numeric(proj_f[normalize_colname(CASH_RECEIVED_COL)], errors='coerce').sum():,.0f}")
with col4:
    if normalize_colname(PT_COL) in proj_f.columns:
        st.metric("Avg PT (days)", f"{pd.to_numeric(proj_f[normalize_colname(PT_COL)], errors='coerce').mean():.1f}")
with col5:
    if normalize_colname(CM2P_FORECAST_COL) in proj_f.columns:
        st.metric("Avg CM2% (Forecast)", f"{pd.to_numeric(proj_f[normalize_colname(CM2P_FORECAST_COL)], errors='coerce').mean():.1f}%")

# -----------------------------
# Tabs
# -----------------------------
t_overview, t_services, t_overruns, t_fxact, t_drivers, t_penalties, t_quality, t_export = st.tabs([
    "Overview", "Services", "Overruns & Delays", "Forecast vs Actual", "Drivers", "Penalties & Cash", "Data Quality", "Export"
])

# === OVERVIEW ===
with t_overview:
    st.subheader("Portfolio overview")
    if normalize_colname(COUNTRY_COL) in proj_f.columns:
        fig = px.histogram(proj_f, x=normalize_colname(COUNTRY_COL))
        st.plotly_chart(fig, use_container_width=True)

    num_df = correlation_frame(proj_f)
    if not num_df.empty:
        method = st.selectbox("Correlation method", ["pearson","spearman"], index=1)
        corr = num_df.corr(method=method)
        fig = px.imshow(corr, text_auto=False, aspect="auto", title=f"Correlation heatmap ({method})")
        st.plotly_chart(fig, use_container_width=True)

    # Probability: P(B_O | H_O) overall & per-service
    st.subheader("Probability: Internal-service **Budget Overrun** given **Hours Overrun**")
    prob = probability_internal_overrun_given_hours(svc_f)
    if not math.isnan(prob["p"]):
        st.markdown(f"**P(B_O | H_O)** = **{prob['p']*100:.1f}%**  (n={prob['n']}, 95% CI {prob['ci'][0]*100:.1f}–{prob['ci'][1]*100:.1f}%)")
    # Per-service
    svc_stats = []
    for s in sorted(svc_cols.keys()):
        sub = svc_f[svc_f["service"] == s]
        mask = (sub["h_o"].fillna(0) > 0)
        n = int(mask.sum())
        k = int((sub.loc[mask, "b_o"].fillna(0) > 0).sum())
        p = (k/n) if n else np.nan
        lo, hi = wilson_ci(k, n) if n else (np.nan, np.nan)
        svc_stats.append({"service": s, "n_with_hours_overrun": n, "p_bo_given_ho": p, "ci_low": lo, "ci_high": hi})
    svc_stats_df = pd.DataFrame(svc_stats)
    st.dataframe(svc_stats_df, use_container_width=True)

# === SERVICES ===
with t_services:
    st.subheader("Service-level metrics")
    # KPIs per service
    svc_agg = svc_f.groupby("service").agg(
        projects=("project_id","nunique"),
        budget=("budget","sum"),
        actual=("actual","sum"),
        forecast=("forecast","sum"),
        hours_overruns=("h_o", lambda s: int((s.fillna(0) > 0).sum())),
        budget_overruns=("b_o", lambda s: int((s.fillna(0) > 0).sum())),
        delays=("delay", lambda s: int((s.fillna(0) > 0).sum())),
        median_inflation=("inflation_factor","median")
    ).reset_index()
    svc_agg["inflation_factor"] = svc_agg["actual"] / svc_agg["budget"]
    st.dataframe(svc_agg, use_container_width=True)

    # Bars: Budget vs Actual by service
    dfm = svc_agg.melt(id_vars=["service"], value_vars=["budget","actual","forecast"], var_name="metric", value_name="hours")
    fig = px.bar(dfm, x="service", y="hours", color="metric", barmode="group", title="Budget vs Actual vs Forecast (hours)")
    st.plotly_chart(fig, use_container_width=True)

    # Inflation factor
    fig = px.bar(svc_agg, x="service", y="inflation_factor", title="Inflation factor (Actual/Budget)")
    st.plotly_chart(fig, use_container_width=True)

# === OVERRUNS & DELAYS ===
with t_overruns:
    st.subheader("Overruns & delays")
    # Totals per project
    cols = [c for c in ["total_h_o","total_b_o","total_o","total_delays","total_penalties"] if c in proj_f.columns]
    if cols:
        melt = proj_f.melt(id_vars=["project_id"], value_vars=cols, var_name="metric", value_name="value")
        fig = px.box(melt, x="metric", y="value", points="all", title="Distributions of totals per project")
        st.plotly_chart(fig, use_container_width=True)

    # On-time completion rate by service
    svc_f["on_time"] = (svc_f["delay"].fillna(0) == 0).astype(int)
    ontime = svc_f.groupby("service")["on_time"].mean().reset_index()
    fig = px.bar(ontime, x="service", y="on_time", title="On-time rate by service")
    st.plotly_chart(fig, use_container_width=True)

# === FORECAST vs ACTUAL ===
with t_fxact:
    st.subheader("Forecast vs Actual (Overrun detection)")
    # Define "forecast overrun" as Forecast > Budget (hours).
    tmp = svc_f.dropna(subset=["forecast_overrun", "h_o"]).copy()
    tmp["forecast_overrun"] = tmp["forecast_overrun"].astype(int)
    tmp["actual_overrun"] = (tmp["h_o"].fillna(0) > 0).astype(int)
    if not tmp.empty:
        cm = confusion_matrix(tmp["actual_overrun"], tmp["forecast_overrun"], labels=[0,1])
        cm_df = pd.DataFrame(cm, index=["Actual No Overrun","Actual Overrun"], columns=["Forecast No Overrun","Forecast Overrun"])
        fig = px.imshow(cm_df, text_auto=True, title="Confusion Matrix (all services combined)")
        st.plotly_chart(fig, use_container_width=True)
        # Precision/recall
        report = classification_report(tmp["actual_overrun"], tmp["forecast_overrun"], labels=[0,1], output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)
    else:
        st.info("Not enough Forecast/Budget pairs to compute confusion matrix.")

    # Per-service F1
    per = []
    for s in sorted(svc_cols.keys()):
        ssub = tmp[tmp["service"] == s]
        if ssub.empty:
            continue
        cm = confusion_matrix(ssub["actual_overrun"], ssub["forecast_overrun"], labels=[0,1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        per.append({"service": s, "precision": prec, "recall": rec, "f1": f1})
    if per:
        st.dataframe(pd.DataFrame(per).sort_values("f1", ascending=False), use_container_width=True)

# === DRIVERS (logistic) ===
with t_drivers:
    st.subheader("Drivers of overruns/delays (logistic)")
    target_choice = st.selectbox("Target", ["any_hours_overrun","any_budget_overrun","any_delay"], index=0)
    # Feature set: do not include totals; use contract value, PT, CM2, Check_V and inflation by service
    feat_cols = []
    for base in [normalize_colname(CONTRACT_VALUE_COL), normalize_colname(PT_COL),
                 normalize_colname(CM2P_FORECAST_COL), normalize_colname(CHECKV_COL)]:
        if base in proj_f.columns:
            feat_cols.append(base)

    # Join in per-project average inflation by service
    infl = svc_f.groupby(["project_id","service"])["inflation_factor"].median().unstack()
    infl.columns = [f"infl_{c.lower()}" for c in infl.columns]
    X = proj_f.set_index("project_id").join(infl, how="left")

    y = proj_f.set_index("project_id")[target_choice].astype(int)
    # Add selected numeric features from project
    for c in feat_cols:
        X[c] = pd.to_numeric(X.get(c), errors="coerce")

    X = X.select_dtypes(include=[np.number]).fillna(0)

    if X.shape[0] >= 10 and X.shape[1] >= 1:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200, n_jobs=None))])
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)[:,1]
        try:
            auc = roc_auc_score(y, proba)
        except Exception:
            auc = np.nan
        st.write(f"ROC AUC: **{auc:.3f}**  |  Samples: {X.shape[0]}")
        # Coeffs as odds ratios
        clf = pipe.named_steps["clf"]
        coefs = clf.coef_[0]
        ors = np.exp(coefs)
        coef_df = pd.DataFrame({"feature": X.columns, "odds_ratio": ors, "coef": coefs}).sort_values("odds_ratio", ascending=False)
        st.dataframe(coef_df, use_container_width=True)
        fig = px.bar(coef_df, x="feature", y="odds_ratio", title="Odds ratios (scaled features)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data/columns to fit the model.")

# === PENALTIES & CASH ===
with t_penalties:
    st.subheader("Penalties and payment terms")
    # Scatter: PT (days) vs Total Penalties
    xcol = normalize_colname(PT_COL)
    ycol = "total_penalties" if "total_penalties" in proj_f.columns else None
    if xcol in proj_f.columns and ycol:
        dfp = proj_f[[xcol, ycol, "project_id"]].dropna()
        fig = px.scatter(dfp, x=xcol, y=ycol, hover_data=["project_id"], trendline="ols",
                         title="Payment term (days) vs Total Penalties")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Payment terms or penalty totals not found.")

    # Bubble: Contract Value vs Delays, bubble size = penalties
    if normalize_colname(CONTRACT_VALUE_COL) in proj_f.columns and "total_delays" in proj_f.columns:
        bub = proj_f.copy()
        fig = px.scatter(bub, x=normalize_colname(CONTRACT_VALUE_COL), y="total_delays",
                         size=("total_penalties" if "total_penalties" in bub.columns else None),
                         hover_name="project_id",
                         title="Contract Value vs Delays (bubble = penalties)")
        st.plotly_chart(fig, use_container_width=True)

# === DATA QUALITY ===
with t_quality:
    st.subheader("Data quality & margin check")
    if normalize_colname(CHECKV_COL) in proj_f.columns:
        cv = pd.to_numeric(proj_f[normalize_colname(CHECKV_COL)], errors="coerce")
        fig = px.histogram(cv, nbins=30, title="Distribution of Check_V (claimed vs actual margin delta)")
        st.plotly_chart(fig, use_container_width=True)
        thr = st.slider("Flag abs(Check_V) >", min_value=0.0, max_value=float(np.nanmax(abs(cv.fillna(0)))) if len(cv)>0 else 10.0, value=5.0)
        flags = proj_f.loc[abs(cv) > thr, ["project_id", normalize_colname(CHECKV_COL)]]
        st.dataframe(flags.rename(columns={normalize_colname(CHECKV_COL): "Check_V"}), use_container_width=True)
    else:
        st.info("Check_V column not present.")

# === EXPORT ===
with t_export:
    st.subheader("Download tidy datasets")
    # Service long table
    svc_out = svc_f.copy()
    csv1 = svc_out.to_csv(index=False).encode("utf-8")
    st.download_button("Download service-level CSV", data=csv1, file_name="services_long.csv", mime="text/csv")
    # Project table
    proj_out = proj_f.copy()
    csv2 = proj_out.to_csv(index=False).encode("utf-8")
    st.download_button("Download project-level CSV", data=csv2, file_name="projects_tidy.csv", mime="text/csv")

    st.caption("Note: hours & EUR are used as-is; flags are 0/1. Forecast overrun = Forecast > Budget.")
