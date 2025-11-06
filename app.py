#This code is poperty of Power Service Solutions GmbH. 
#The use, download, distribution os benefit in any way via the use of it without the explicit consent of Power Service Solutions GmbH constitutes a breach in the intelectual property law.
#This code was developed, tested and deployed by Daniel Garcia Rey

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# ------------------------------------------------------------
# Page + Theme Config
# ------------------------------------------------------------
st.set_page_config(page_title="Deep Margins", page_icon="📊", layout="wide")

THEMES = {
    "Corporate": {
        "bg": "#e6e9ee", "card": "#ffffff", "text": "#0f172a", "muted": "#6b7280",
        "accent": "#2f6feb", "template": "plotly_white",
        "font_import": "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap",
        "font_family": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial",
    },
    "Dark": {
        "bg": "#0f172a", "card": "#111827", "text": "#e5e7eb", "muted": "#94a3b8",
        "accent": "#84ccff", "template": "plotly_dark",
        "font_import": "https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&display=swap",
        "font_family": "'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial",
    },
    "Ocean": {
        "bg": "#e6f7fb", "card": "#ffffff", "text": "#0b2530", "muted": "#517a88",
        "accent": "#0ea5b7", "template": "plotly_white",
        "font_import": "https://fonts.googleapis.com/css2?family=Rubik:wght@400;600;700&display=swap",
        "font_family": "'Rubik', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial",
    },
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
# (Optional) Simple login — remove if not used in your app
# ------------------------------------------------------------
if "auth" not in st.session_state: st.session_state["auth"] = True  # flip to False to enable login
if "theme" not in st.session_state: st.session_state["theme"] = "Corporate"

if not st.session_state["auth"]:
    st.title("🔐 Deep Margins Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login", use_container_width=True):
        st.session_state["auth"] = True   # plug your auth here
        st.success("Welcome!")
    st.stop()

# Sidebar theme selector
st.sidebar.title("🎨 Theme")
sel_theme = st.sidebar.selectbox(
    "Choose Theme", list(THEMES.keys()),
    index=list(THEMES.keys()).index(st.session_state["theme"])
)
apply_theme(sel_theme)
st.session_state["theme"] = sel_theme

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
SERVICE_PRETTY = {
    "tpm": "TPM", "cpm": "CPM", "eng": "Engineering", "qa_qc_exp": "QA/QC/Exp",
    "hse": "HSE", "constr": "Construction", "com": "Commissioning",
    "man": "Manufacturing", "proc": "Procurement",
}
SERVICE_BLOCKS = ["tpm", "cpm", "eng", "qa_qc_exp", "hse", "constr", "com", "man", "proc"]

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
    try: return float(x)
    except Exception: return np.nan

def safe_int(v):
    try: return int(float(v))
    except Exception: return 0

def plotly_config(name: str):
    return {"displaylogo": False,
            "toImageButtonOptions": {"format":"png","filename":f"{name}","height":1350,"width":2400,"scale":2}}

def is_binary_series(s: pd.Series) -> bool:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty: return False
    uniq = pd.unique(s.astype(float))
    return np.all(np.isin(uniq, [0.0, 1.0]))

def to_flag(s: pd.Series, op: str, thr: float):
    s = pd.to_numeric(s, errors="coerce")
    if op in ("≥", ">="): return (s >= thr).astype(int)
    if op == ">": return (s > thr).astype(int)
    if op in ("≤", "<="): return (s <= thr).astype(int)
    if op == "<": return (s < thr).astype(int)
    if op in ("=", "==", "is"): return (s == thr).astype(int)
    return (s > thr).astype(int)

def humanize_col(c: str) -> str:
    s = c.lower()
    for k,v in SERVICE_PRETTY.items():
        s = s.replace(f"{k}_", f"{v} ")
    s = (s.replace("b_o", "budget overrun")
           .replace("h_o", "hours overrun")
           .replace("cm2pct", "CM2%")
           .replace("cm2_", "CM2 "))
    s = s.replace("_", " ").strip()
    return s[:1].upper() + s[1:]

# ---------- Column resolver ----------

def find_col(df: pd.DataFrame, must_contain):
    tokens = [t.lower() for t in must_contain if t]
    svc_tokens = set(["tpm","cpm","eng","qa_qc_exp","hse","constr","com","man","proc"])
    suffix_tokens = set(["b_o","h_o","delay"])
    services = [t for t in tokens if t in svc_tokens]
    suffixes = [t for t in tokens if t in suffix_tokens]
    if services and suffixes:
        candidate = f"{services[0]}_{suffixes[0]}"
        if candidate in df.columns:
            return candidate
    for c in df.columns:
        lc = c.lower()
        if all(t in lc for t in tokens):
            return c
    joined = "_".join(tokens)
    for c in df.columns:
        if joined in c.lower():
            return c
    return None

# ------------------------------------------------------------
# Data load + filters
# ------------------------------------------------------------
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], key="upl_main")
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

# Numeric coercions
for col in [
    "contract_value", "cash_received",
    "cm2_forecast", "cm2_actual", "cm2_budget",
    "cm2pct_forecast", "cm2pct_actual", "cm2pct_budget",
    "total_penalties", "total_o", "total_delays", "check_v"
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Identify Real CM2% Deviation column once (for targets used elsewhere)
REAL_DEV_COL = (
    find_col(df, ["real","cm2","pct","dev"]) or
    find_col(df, ["cm2pct","real","dev"]) or
    find_col(df, ["cm2pct","dev"]) or
    None
)

# Create profit label (Profitable vs Non-Profitable) based on Real CM2% Deviation, if present
if REAL_DEV_COL:
    df["profit_label"] = np.where(pd.to_numeric(df[REAL_DEV_COL], errors="coerce") < 0,
                                  "Non-Profitable", "Profitable")

# ------------------------------------------------------------
# KPIs
# ------------------------------------------------------------
st.title("📊 Deep Margins — PSS Analytics Dashboard")

total_contract = df.get("contract_value", pd.Series(dtype=float)).sum()
total_cash = df.get("cash_received", pd.Series(dtype=float)).sum()
weighted_fore_pct = (df.get("cm2_forecast", 0).sum() / total_contract * 100) if total_contract else 0
weighted_real_pct = (df.get("cm2_actual", 0).sum() / total_contract * 100) if total_contract else 0

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
    "Margin Bridge", "Probabilities", "Overrun Heatmap", "Drivers",
    "Patterns", "Project Analyzer"
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
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="tealrose",
            title="Correlation heatmap (incl. hours & budget overruns)"
        )
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
        fig.update_layout(xaxis_title="Contract Value (EUR)", yaxis_title="CM2% Forecast")
        st.plotly_chart(fig, use_container_width=True, config=plotly_config("penalty_bubble"))

# ------------------------------------------------------------
# 2) Internal Services Metrics
# ------------------------------------------------------------
with tabs[1]:
    st.subheader("Internal Services Metrics")

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
        find_col(df, ["cm2","eur","delta"]) or None
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
# 4) Probabilities (Simple language + Advanced A/B/C)
# ------------------------------------------------------------
with tabs[3]:
    st.subheader("Probabilities")
    st.caption("Estimate P(negative margin) under conditions. See previous version for full UI.")
    st.info("(Kept minimal here to focus on new 'Patterns' requirements.)")

# ------------------------------------------------------------
# 5) Overrun Heatmap
# ------------------------------------------------------------
with tabs[4]:
    st.subheader("Overrun & Delay density heatmap")
    svc_filtered = svc[~svc["service"].isin(["MAN", "PROC"])]
    heat = svc_filtered.groupby("service")[["h_o", "b_o", "delay"]].mean().reset_index() if not svc_filtered.empty else pd.DataFrame()
    if not heat.empty:
        heat["Service"] = heat["service"].map(PRETTY_UP)
        melt = heat.melt(id_vars="Service", var_name="Type", value_name="Rate")
        fig6 = px.density_heatmap(
            melt, x="Type", y="Service", z="Rate",
            color_continuous_scale="tealrose",
            title="Average overrun rate by service"
        )
        st.plotly_chart(fig6, use_container_width=True, config=plotly_config("overrun_heatmap"))
    else:
        st.info("No service-level data to plot.")

# ------------------------------------------------------------
# 6) Drivers (odds-like ranking over flags)
# ------------------------------------------------------------
with tabs[5]:
    st.subheader("Drivers of CM2% deviation (service-level flags)")

    real_dev_col = REAL_DEV_COL or find_col(df, ["cm2pct","forecast"])  # fallback
    if real_dev_col is None or df[real_dev_col].dropna().nunique() == 0:
        st.info("Not enough variation to fit driver model.")
    else:
        y = (pd.to_numeric(df[real_dev_col], errors="coerce") < 0).astype(int)
        feat_df = pd.DataFrame(index=df.index)
        feat_cols = []
        for s in SERVICE_BLOCKS:
            for suffix, label in [("b_o","budget_overrun"), ("h_o","hours_overrun"), ("delay","delay")]:
                col = f"{s}_{suffix}"
                if col in df.columns:
                    fcol = f"{s}_{suffix}_flag"
                    if fcol in EXCLUDED_DRIVER_FLAGS:  # user preference
                        continue
                    feat_df[fcol] = (pd.to_numeric(df[col], errors="coerce").fillna(0) > 0).astype(int)
                    feat_cols.append((fcol, s, label))
        if feat_df.shape[1] == 0 or y.nunique() < 2:
            st.info("Not enough data to compute drivers.")
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

# ------------------------------------------------------------
# 7) PATTERNS — shared variables & variable distributions (based on CM2% Forecast)
# ------------------------------------------------------------
with tabs[6]:
    st.subheader("Patterns — what profitable vs non‑profitable projects share (by CM2% Forecast)")

    # 1) Define profitable using CM2% Forecast threshold
    fore_col = (
        find_col(df, ["cm2pct", "forecast"]) or
        find_col(df, ["cm2", "pct", "forecast"]) or
        "cm2pct_forecast"
    )
    if fore_col not in df.columns:
        st.warning("Couldn't find a 'CM2% Forecast' column.")
        st.stop()

    fore = pd.to_numeric(df[fore_col], errors="coerce")
    mask_valid = fore.notna()
    dfx = df.loc[mask_valid].copy()

    col_thr, col_top = st.columns([1,1])
    with col_thr:
        thr = st.number_input(
            "Profit threshold for CM2% Forecast ( > threshold = profitable )",
            value=0.0, step=0.5,
            help="Default 0.0 → positive forecast margin counts as profitable."
        )
    with col_top:
        topn = st.number_input("Top N patterns", min_value=3, max_value=20, value=8, step=1)

    y = (pd.to_numeric(dfx[fore_col], errors="coerce") > float(thr)).astype(int)
    cls = y.map({1: "Profitable", 0: "Non‑Profitable"})
    dfx["_class"] = cls

    # 2) A quick auto-ranked view (binary + numeric) — still useful, keep it
    def _auto_patterns(dfx, y, exclude):
        # Binary candidates
        flag_cols = [c for c in dfx.columns if is_binary_series(dfx[c]) and c not in exclude]
        rows_bin = []
        if flag_cols:
            pm, nm = (y == 1), (y == 0)
            for c in flag_cols:
                s = pd.to_numeric(dfx[c], errors="coerce").fillna(0).astype(int)
                r1, r0 = (s[pm].mean() if pm.any() else 0.0), (s[nm].mean() if nm.any() else 0.0)
                rows_bin.append({
                    "Variable": humanize_col(c), "Column": c,
                    "Rate_Profitable": r1, "Rate_NonProfitable": r0,
                    "Delta_pp": r1 - r0,
                })
        bin_df = pd.DataFrame(rows_bin)

        # Numeric candidates
        num_cols = [c for c in dfx.columns if pd.api.types.is_numeric_dtype(dfx[c]) and c not in exclude]
        if fore_col in num_cols:
            num_cols.remove(fore_col)
        rows_num = []
        pm, nm = (y == 1), (y == 0)
        n1, n0 = int(pm.sum()), int(nm.sum())
        for c in num_cols:
            x = pd.to_numeric(dfx[c], errors="coerce")
            xp, xn = x[pm].dropna(), x[nm].dropna()
            if xp.empty or xn.empty: continue
            m1, m0 = xp.median(), xn.median()
            v1, v0 = xp.var(ddof=1), xn.var(ddof=1)
            try:
                sp = np.sqrt(max(((n1-1)*v1 + (n0-1)*v0, 0)) / max((n1+n0-2, 1)))
            except Exception:
                sp = np.nan
            effect = (m1 - m0) / sp if (sp and sp > 0) else np.nan
            rows_num.append({
                "Variable": humanize_col(c), "Column": c,
                "Median_Profitable": m1, "Median_NonProfitable": m0,
                "Diff": m1 - m0, "EffectSize_d": effect,
            })
        num_df = pd.DataFrame(rows_num)
        return bin_df, num_df

    exclude = {fore_col, "project_id", "check_v"}
    bin_df, num_df = _auto_patterns(dfx.copy(), y, exclude)

    with st.expander("Top patterns (auto‑ranked)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Binary flags more common in PROFITABLE**")
            if not bin_df.empty:
                top_bin_prof = bin_df.sort_values("Delta_pp", ascending=False).head(topn)
                figp = px.bar(top_bin_prof, x="Delta_pp", y="Variable", orientation="h",
                              title="Δ percentage points (Profitable − Non‑Profitable)",
                              color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(figp, use_container_width=True, config=plotly_config("patterns_bin_prof"))
                st.dataframe(top_bin_prof[["Variable","Rate_Profitable","Rate_NonProfitable","Delta_pp"]]
                             .style.format({"Rate_Profitable":"{:.2%}","Rate_NonProfitable":"{:.2%}","Delta_pp":"{:+.2%}"}),
                             use_container_width=True)
            else:
                st.info("No binary/flag variables detected.")
        with col2:
            st.markdown("**Binary flags more common in NON‑PROFITABLE**")
            if not bin_df.empty:
                top_bin_non = bin_df.sort_values("Delta_pp").head(topn)
                fign = px.bar(top_bin_non.assign(Delta_to_Non=-top_bin_non["Delta_pp"]),
                              x="Delta_to_Non", y="Variable", orientation="h",
                              title="Δ toward Non‑Profitable", color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fign, use_container_width=True, config=plotly_config("patterns_bin_non"))
                st.dataframe(top_bin_non[["Variable","Rate_Profitable","Rate_NonProfitable","Delta_pp"]]
                             .style.format({"Rate_Profitable":"{:.2%}","Rate_NonProfitable":"{:.2%}","Delta_pp":"{:+.2%}"}),
                             use_container_width=True)
            else:
                st.info("No binary/flag variables detected.")

    st.divider()

    # 3) YOUR ASK: Shared variables (categorical) + service-specific breakdown
    st.markdown("### Shared categorical variables (at‑a‑glance)")

    # Build portfolio-level categorical flags
    cats = {}
    if "total_penalties" in dfx.columns:
        cats["Penalties > 0"] = (pd.to_numeric(dfx["total_penalties"], errors="coerce").fillna(0) > 0).astype(int)
    if "total_o" in dfx.columns:
        cats["Hours Overrun Total > 0"] = (pd.to_numeric(dfx["total_o"], errors="coerce").fillna(0) > 0).astype(int)
    if "total_delays" in dfx.columns:
        cats["Delays Total > 0"] = (pd.to_numeric(dfx["total_delays"], errors="coerce").fillna(0) > 0).astype(int)

    # Any budget/hours/delay flag across services
    def any_suffix_flag(df_in: pd.DataFrame, suffix: str) -> pd.Series:
        acc = pd.Series(0, index=df_in.index)
        for s in SERVICE_BLOCKS:
            c = f"{s}_{suffix}"
            if c in df_in.columns:
                acc = acc | (pd.to_numeric(df_in[c], errors="coerce").fillna(0) > 0).astype(int)
        return acc.astype(int)

    cats["Any Budget Overrun (any service)"] = any_suffix_flag(dfx, "b_o")
    cats["Any Hours Overrun (any service)"] = any_suffix_flag(dfx, "h_o")
    cats["Any Delay (any service)"] = any_suffix_flag(dfx, "delay")

    cat_rows = []
    pm, nm = (y == 1), (y == 0)
    for label, ser in cats.items():
        r1 = ser[pm].mean() if pm.any() else 0.0
        r0 = ser[nm].mean() if nm.any() else 0.0
        cat_rows.append({
            "Variable": label,
            "Rate_Profitable": r1,
            "Rate_NonProfitable": r0,
            "Delta_pp": r1 - r0,
        })
    cat_df = pd.DataFrame(cat_rows)

    if not cat_df.empty:
        # Show top 5 by absolute separation; this guarantees >=3 categorical as requested
        cat_top = cat_df.reindex(cat_df["Delta_pp"].abs().sort_values(ascending=False).index).head(5)
        figc = px.bar(cat_top, x="Delta_pp", y="Variable", orientation="h",
                      title="Shared categorical patterns (Δ Profitable − Non‑Profitable)")
        st.plotly_chart(figc, use_container_width=True, config=plotly_config("patterns_shared_cats"))
        st.dataframe(cat_top[["Variable","Rate_Profitable","Rate_NonProfitable","Delta_pp"]]
                     .style.format({"Rate_Profitable":"{:.2%}","Rate_NonProfitable":"{:.2%}","Delta_pp":"{:+.2%}"}),
                     use_container_width=True)
    else:
        st.info("No categorical totals/flags available.")

    st.markdown("### Service‑specific patterns (b_o / h_o / delay)")
    suffix = st.radio("Choose signal", ["b_o", "h_o", "delay"], horizontal=True, index=2)

    svc_rows = []
    for s in SERVICE_BLOCKS:
        c = f"{s}_{suffix}"
        if c in dfx.columns:
            ser = (pd.to_numeric(dfx[c], errors="coerce").fillna(0) > 0).astype(int)
            r1 = ser[pm].mean() if pm.any() else 0.0
            r0 = ser[nm].mean() if nm.any() else 0.0
            svc_rows.append({
                "Service": SERVICE_PRETTY.get(s, s.upper()),
                "Rate_Profitable": r1,
                "Rate_NonProfitable": r0,
                "Delta_pp": r1 - r0,
            })
    svc_df = pd.DataFrame(svc_rows)

    if not svc_df.empty:
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Most characteristic of PROFITABLE**")
            top_p = svc_df.sort_values("Delta_pp", ascending=False).head(6)
            fig_sp = px.bar(top_p, x="Delta_pp", y="Service", orientation="h",
                            title=f"{suffix.upper()} — Δ Profitable − Non‑Profitable")
            st.plotly_chart(fig_sp, use_container_width=True, config=plotly_config("svc_prof"))
            st.dataframe(top_p.style.format({"Rate_Profitable":"{:.2%}","Rate_NonProfitable":"{:.2%}","Delta_pp":"{:+.2%}"}), use_container_width=True)
        with colB:
            st.markdown("**Most characteristic of NON‑PROFITABLE**")
            top_n = svc_df.sort_values("Delta_pp").head(6)
            fig_sn = px.bar(top_n.assign(Delta_to_Non=-top_n["Delta_pp"]), x="Delta_to_Non", y="Service", orientation="h",
                            title=f"{suffix.upper()} — Δ toward Non‑Profitable")
            st.plotly_chart(fig_sn, use_container_width=True, config=plotly_config("svc_non"))
            st.dataframe(top_n.style.format({"Rate_Profitable":"{:.2%}","Rate_NonProfitable":"{:.2%}","Delta_pp":"{:+.2%}"}), use_container_width=True)
    else:
        st.info("No service-level flags present for the selected signal.")

    st.divider()

    # 4) Variable-focused bell-style (overlay histogram) split by profit class
    st.markdown("### Variable distribution by class (choose any column)")
    # Candidate variables: numeric + counts; show a clean pick list
    cand_vars = [c for c in dfx.columns if pd.api.types.is_numeric_dtype(dfx[c]) and c not in {fore_col}]
    # fast aliases
    aliases = [
        c for c in ["total_penalties", "total_delays", "total_o",
                    "eng_delay", "com_delay", "proc_b_o", "eng_b_o", "man_delay", "tpm_h_o"] if c in dfx.columns
    ]
    pick = st.selectbox("Select variable", aliases + [c for c in cand_vars if c not in aliases], format_func=humanize_col)

    if pick:
        prof_vals = pd.to_numeric(dfx.loc[dfx["_class"]=="Profitable", pick], errors="coerce")
        nonp_vals = pd.to_numeric(dfx.loc[dfx["_class"]=="Non‑Profitable", pick], errors="coerce")
        med_prof = prof_vals.median() if not prof_vals.dropna().empty else float("nan")
        med_nonp = nonp_vals.median() if not nonp_vals.dropna().empty else float("nan")
        st.write(f"Median {humanize_col(pick)} — Profitable: **{med_prof:.2f}**, Non‑Profitable: **{med_nonp:.2f}**")
        figd = px.histogram(dfx, x=pick, color="_class", barmode="overlay", nbins=20,
                            category_orders={"_class":["Profitable","Non‑Profitable"]},
                            labels={pick: humanize_col(pick), "_class":"Class"})
        figd.update_traces(opacity=0.70)
        st.plotly_chart(figd, use_container_width=True, config=plotly_config("var_bell"))

# ------------------------------------------------------------
# 8) Project Analyzer — heuristic (no ML training requested)
# ------------------------------------------------------------
with tabs[7]:
    st.subheader("Project Analyzer — heuristic risk from shared patterns")

    # Use simple heuristics based on the categorical patterns above
    # (penalties>0, any b_o/h_o/delay, totals > 0) and service-specific deltas.
    name_col_guess = next((c for c in ["name", "project_name", "project", "project_id"] if c in df.columns), None)
    if name_col_guess is None:
        st.info("No project name/id column found.")
    else:
        projects = list(df[name_col_guess].astype(str).values)
        sel = st.selectbox("Select a project", projects)
        row = df.loc[df[name_col_guess].astype(str) == str(sel)].head(1)
        if row.empty:
            st.warning("Project not found.")
        else:
            r = row.iloc[0]
            # Heuristic scoring
            score_box = {"v": 0}
            notes = []

            def add(flag, pts, label):
                if flag:
                    score_box["v"] += pts
                    notes.append(f"+{pts}: {label}")

            # Portfolio-level flags
            add((r.get("total_penalties", 0) or 0) > 0, 2, "Penalties present")
            add((r.get("total_o", 0) or 0) > 0, 1, "Hours overrun total > 0")
            add((r.get("total_delays", 0) or 0) > 0, 1, "Delays total > 0")

            # Any service flags
            def any_flag(row, suffix):
                for s in SERVICE_BLOCKS:
                    v = row.get(f"{s}_{suffix}", 0)
                    if pd.to_numeric(pd.Series([v]), errors="coerce").fillna(0).iloc[0] > 0:
                        return True
                return False

            add(any_flag(r, "b_o"), 2, "Budget overrun in at least one service")
            add(any_flag(r, "h_o"), 2, "Hours overrun in at least one service")
            add(any_flag(r, "delay"), 2, "Delay in at least one service")

            risk_band = "Low" if score_box["v"] <= 1 else ("Medium" if score_box["v"] <= 4 else "High")
            st.metric("Heuristic risk (negative margin)", risk_band)
            st.write("**Signals triggering risk:**")
            if notes:
                for n in notes: st.write("- ", n)
            else:
                st.write("- None (clean profile)")

            # Show quick service snapshot for ENG/COM/PROC/MAN/TPM as you asked
            focus = ["eng", "com", "proc", "man", "tpm"]
            cols = []
            for s in focus:
                cols += [c for c in [f"{s}_b_o", f"{s}_h_o", f"{s}_delay"] if c in df.columns]
            snap = row[cols].T
            if not snap.empty:
                snap = snap.rename(index=lambda c: humanize_col(c))
                st.dataframe(snap, use_container_width=True)
                st.caption("Heuristic shown above; below: model probability, risk levers, and CM2% forecast estimate.")

            # ----- Model-based probability of negative margin (logistic, signals only)
            if REAL_DEV_COL and df[REAL_DEV_COL].notna().sum() >= 5:
                y_all = (pd.to_numeric(df[REAL_DEV_COL], errors="coerce") < 0).astype(int)
                # Build feature matrix from signals
                F = pd.DataFrame(index=df.index)
                for s in SERVICE_BLOCKS:
                    for suf in ["b_o","h_o","delay"]:
                        coln = f"{s}_{suf}"
                        if coln in df.columns:
                            F[f"{coln}_flag"] = (pd.to_numeric(df[coln], errors="coerce").fillna(0) > 0).astype(int)
                for cfeat in ["total_penalties","total_delays","total_o","contract_value"]:
                    if cfeat in df.columns:
                        F[cfeat] = pd.to_numeric(df[cfeat], errors="coerce").fillna(0)
                F = F.fillna(0)
                if y_all.nunique() == 2 and F.shape[1] > 0:
                    try:
                        clf = LogisticRegression(max_iter=600, class_weight="balanced")
                        clf.fit(F, y_all)
                        prob_neg = clf.predict_proba(F.loc[row.index])[0][1] * 100
                        st.metric("Model P(negative margin)", f"{prob_neg:.1f}%")
                    except Exception as e:
                        st.info(f"Couldn't fit probability model: {e}")
                else:
                    st.info("Not enough variation to fit probability model.")
            else:
                st.info("Outcome column not available to fit probability model.")

            # ----- Heuristic risk levers (what to change first)
            st.markdown("#### Heuristic risk levers (what to change first)")
            levers_rows = []
            try:
                y_all = (pd.to_numeric(df[REAL_DEV_COL], errors="coerce") < 0).astype(int)
                # Rebuild features for lever calc
                F = pd.DataFrame(index=df.index)
                for s in SERVICE_BLOCKS:
                    for suf in ["b_o","h_o","delay"]:
                        coln = f"{s}_{suf}"
                        if coln in df.columns:
                            F[f"{coln}_flag"] = (pd.to_numeric(df[coln], errors="coerce").fillna(0) > 0).astype(int)
                for cfeat in ["total_penalties","total_delays","total_o","contract_value"]:
                    if cfeat in df.columns:
                        F[cfeat] = pd.to_numeric(df[cfeat], errors="coerce").fillna(0)
                F = F.fillna(0)

                projF = F.loc[row.index].iloc[0]
                # Flag levers
                flag_cols = [c for c in F.columns if c.endswith("_flag")]
                for c in flag_cols:
                    r_neg = F.loc[y_all==1, c].mean() if (y_all==1).any() else 0.0
                    r_pos = F.loc[y_all==0, c].mean() if (y_all==0).any() else 0.0
                    delta_to_neg = r_neg - r_pos
                    if projF[c] == 1 and delta_to_neg > 0:
                        levers_rows.append({
                            "Variable": humanize_col(c.replace("_flag","")),
                            "Project": "Yes",
                            "Δ toward negative": delta_to_neg,
                            "Suggestion": "Avoid / resolve",
                        })
                # Numeric totals levers
                for c in ["total_penalties","total_delays","total_o"]:
                    if c in F.columns:
                        val = float(projF[c])
                        p_med = float(pd.to_numeric(df.loc[y_all==0, c], errors="coerce").median())
                        n_med = float(pd.to_numeric(df.loc[y_all==1, c], errors="coerce").median())
                        if np.isfinite(val) and np.isfinite(p_med) and np.isfinite(n_med):
                            if n_med > p_med and val > p_med:
                                levers_rows.append({
                                    "Variable": humanize_col(c),
                                    "Project": f"{val:.2f}",
                                    "Δ toward negative": (n_med - p_med),
                                    "Suggestion": f"Reduce ≤ {p_med:.2f}",
                                })
                            elif n_med < p_med and val < p_med:
                                levers_rows.append({
                                    "Variable": humanize_col(c),
                                    "Project": f"{val:.2f}",
                                    "Δ toward negative": (p_med - n_med),
                                    "Suggestion": f"Increase ≥ {p_med:.2f}",
                                })
                if levers_rows:
                    levers_df = pd.DataFrame(levers_rows).sort_values("Δ toward negative", ascending=False).head(8)
                    st.dataframe(levers_df, use_container_width=True)
                else:
                    st.info("No obvious single-variable risk levers detected for this project.")
            except Exception as e:
                st.info(f"Couldn't compute heuristic levers: {e}")            # ----- Estimated CM2% Forecast from signals (Ridge)
            st.markdown("#### Estimated CM2% Forecast (signals-based)")
            fore_col = (
                find_col(df, ["cm2pct", "forecast"]) or
                find_col(df, ["cm2", "pct", "forecast"]) or
                "cm2pct_forecast"
            )
            if fore_col in df.columns:
                y_fore = pd.to_numeric(df[fore_col], errors="coerce")
                mask = y_fore.notna()

                # Build features fresh for regression (don't rely on earlier blocks)
                F_fore = pd.DataFrame(index=df.index)
                for s in SERVICE_BLOCKS:
                    for suf in ["b_o","h_o","delay"]:
                        coln = f"{s}_{suf}"
                        if coln in df.columns:
                            F_fore[f"{coln}_flag"] = (pd.to_numeric(df[coln], errors="coerce").fillna(0) > 0).astype(int)
                for cfeat in ["total_penalties","total_delays","total_o","contract_value"]:
                    if cfeat in df.columns:
                        F_fore[cfeat] = pd.to_numeric(df[cfeat], errors="coerce").fillna(0)
                F_fore = F_fore.fillna(0)

                F_reg = F_fore.loc[mask]
                y_reg = y_fore.loc[mask]
                if F_reg.shape[0] >= 8 and F_reg.shape[1] > 0:
                    try:
                        reg = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
                        reg.fit(F_reg, y_reg)
                        yhat = float(reg.predict(F_fore.loc[row.index])[0])
                        
