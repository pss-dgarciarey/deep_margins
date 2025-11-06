#This code is poperty of Power Service Solutions GmbH. 
#The use, download, distribution os benefit in any way via the use of it without the explicit consent of Power Service Solutions GmbH constitutes a breach in the intelectual property law.
#This code was developed, tested and deployed by Daniel Garcia Rey

# ================================================================
# app.py — Deep Margins Dashboard (full: Overview scatter, Drivers general,
# Probabilities Simple + Advanced, Patterns tab, Project Analyzer with heuristics)
# ================================================================

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

# (Optional) Simple login — keep disabled for now
if "auth" not in st.session_state: st.session_state["auth"] = True
if "theme" not in st.session_state: st.session_state["theme"] = "Corporate"

if not st.session_state["auth"]:
    st.title("🔐 Deep Margins Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login", use_container_width=True):
        st.session_state["auth"] = True
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

# Exclude a few noisy flags from the Drivers ranking
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
    while "__" in c: c = c.replace("__", "_")
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
           .replace("delay", "delay")
           .replace("cm2pct", "CM2%")
           .replace("cm2_", "CM2 "))
    s = s.replace("_", " ").strip()
    return s[:1].upper() + s[1:]

# ---------- Column resolver ----------
def find_col(df: pd.DataFrame, must_contain):
    """
    Prefer '<service>_<suffix>' when tokens include a service + {b_o,h_o,delay};
    else fallback to 'all tokens present' search, then underscore-joined.
    """
    tokens = [t.lower() for t in must_contain if t]
    svc_tokens = set(["tpm","cpm","eng","qa_qc_exp","hse","constr","com","man","proc"])
    suffix_tokens = set(["b_o","h_o","delay"])
    services = [t for t in tokens if t in svc_tokens]
    suffixes = [t for t in tokens if t in suffix_tokens]
    if services and suffixes:
        candidate = f"{services[0]}_{suffixes[0]}"
        if candidate in df.columns: return candidate
    for c in df.columns:
        lc = c.lower()
        if all(t in lc for t in tokens): return c
    joined = "_".join(tokens)
    for c in df.columns:
        if joined in c.lower(): return c
    return None

# ---------- Natural-language helpers for Probabilities (Simple mode) ----------
def _normalize_query_text(q: str) -> str:
    q = q.lower().strip()
    m = re.search(r"\b(when|if|given|where)\b(.*)$", q)
    if m: q = m.group(2).strip()
    q = q.replace("%", "pct")

    for pat, rep in [
        (r"\bno more than\b", "<="), (r"\bat most\b", "<="),
        (r"\bnot more than\b", "<="), (r"\bless than or equal to\b", "<="),
        (r"\bat least\b", ">="), (r"\bnot less than\b", ">="),
        (r"\bgreater than or equal to\b", ">="),
        (r"\bmore than\b", ">"), (r"\bgreater than\b", ">"), (r"\bover\b", ">"),
        (r"\bunder\b", "<"), (r"\bbelow\b", "<"), (r"\bless than\b", "<"),
        (r"\bequals?\b", "="), (r"\bis\b", " is "), (r"\bare\b", " is "),
    ]:
        q = re.sub(pat, f" {rep} ", q)

    for pat, rep in {
        r"\bconstruction\b": "constr",
        r"\bengineering\b": "eng",
        r"\bcommissioning\b|\bcommisioning\b|\bcommission\b": "com",
        r"\bmanufacturing\b": "man",
        r"\bprocurement\b": "proc",
        r"\bqa\s*/?\s*qc\s*/?\s*exp\b|\bquality\b": "qa_qc_exp",
        r"\bhealth\s*and\s*safety\b|\bhse\b": "hse",
        r"\bcontract\s*value\b": "contract value",
        r"\bpenalties?\b": "total penalties",
        r"\btotal\s*delays?\b": "total delays",
        r"\bcm2\s*forecast\b": "cm2 forecast",
        r"\bcm2\s*pct\s*forecast\b|\bcm2pct\s*forecast\b|\bcm2\s*%?\s*forecast\b": "cm2pct forecast",
    }.items():
        q = re.sub(pat, rep, q)

    q = re.sub(r"\b([a-z_]+)\s+(?:is\s+)?not\s+delayed\b", r"\1 delay = no", q)
    q = re.sub(r"\b([a-z_]+)\s+(?:is\s+)?delayed\b", r"\1 delay = yes", q)

    q = re.sub(r"\b([a-z_]+)\s+(?:is\s+)?budget\s+(?:is\s+)?over[-\s]?run(s)?\b", r"\1 b_o > 0", q)
    q = re.sub(r"\b([a-z_]+)\s+(?:is\s+)?hours?\s+(?:is\s+)?over[-\s]?run(s)?\b", r"\1 h_o > 0", q)
    q = re.sub(r"\b([a-z_]+)\s+budget\s+over[-\s]?run(s)?\b", r"\1 b_o > 0", q)
    q = re.sub(r"\b([a-z_]+)\s+hours?\s+over[-\s]?run(s)?\b", r"\1 h_o > 0", q)
    q = re.sub(r"\b([a-z_]+)\s+is\s+b_o\s*>\s*0\b", r"\1 b_o > 0", q)
    q = re.sub(r"\b([a-z_]+)\s+is\s+h_o\s*>\s*0\b", r"\1 h_o > 0", q)

    q = re.sub(r"\bnot\s+delayed\b", "delay = no", q)
    q = re.sub(r"\bno\s+delay\b", "delay = no", q)
    q = re.sub(r"\bdelays?\b", "delay", q)
    q = re.sub(r"\bbudget\s+(?:is\s+)?over[-\s]?run(s)?\b", "b_o > 0", q)
    q = re.sub(r"\bhours?\s+(?:is\s+)?over[-\s]?run(s)?\b", "h_o > 0", q)

    q = re.sub(r"\s+", " ", q).strip()
    return q

def _parse_number(text: str) -> float:
    s = text.strip().lower().replace("€","").replace("$","").replace("£","")
    s = s.replace(" ", "").replace("_","").replace(",", "")
    if s.count(".") > 1: s = s.replace(".", "")
    mult = 1.0
    if s.endswith("bn"): mult, s = 1e9, s[:-2]
    elif s.endswith("b"): mult, s = 1e9, s[:-1]
    elif s.endswith("m"): mult, s = 1e6, s[:-1]
    elif s.endswith("k"): mult, s = 1e3, s[:-1]
    s = s.replace("pct", "")
    try: return float(s) * mult
    except Exception: return np.nan

def parse_query_to_mask(q: str, df: pd.DataFrame, return_details: bool = False):
    normalized = _normalize_query_text(q)
    parts = re.split(r"\b(and|or)\b", normalized)

    clauses, ops, details = [], [], []
    for part in parts:
        part = part.strip()
        if part in ("and","or"):
            ops.append(part.upper()); continue
        if not part: continue

        m = re.search(r"(.+?)\s*(=|is|==|>=|<=|>|<)\s*(.+)$", part)
        if m:
            lhs, op, rhs = [x.strip() for x in m.groups()]
            tokens = [t for t in re.split(r"[^\w]+", lhs) if t]
            if "cm2pct" in tokens and "forecast" in tokens:
                tokens = [t for t in tokens if t not in ("cm2pct","forecast")] + ["cm2pct","forecast"]
            col = find_col(df, tokens) or find_col(df, ["_".join(tokens)])
            if col is None:
                details.append({"raw": part, "parsed": None, "reason": "no column"}); continue

            if re.fullmatch(r"(yes|true|1|no|false|0)", rhs):
                val = 1 if rhs in ("yes","true","1") else 0
                ser = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                mask = (ser == val)
                desc = f"{humanize_col(col)} = {'Yes' if val==1 else 'No'}"
            else:
                thr = _parse_number(rhs)
                ser = pd.to_numeric(df[col], errors="coerce")
                mask = (to_flag(ser, op, float(thr)) == 1)

                pretty = humanize_col(col)
                desc = None
                if col.endswith("_h_o") or col.endswith("_b_o"):
                    if (op in {">", ">="} and float(thr) <= 0) or (op in ("is","=","==") and str(rhs) in ("1","yes","true")):
                        desc = f"{pretty} = Yes"
                    elif (op in {"<","<=","=","=="} and float(thr) <= 0) or (op in ("is","=","==") and str(rhs) in ("0","no","false")):
                        desc = f"{pretty} = No"
                if desc is None:
                    sym = op if op in ("<=",">=","<",">","=","is","==") else op
                    desc = f"{pretty} {sym} {thr:g}"

            clauses.append((mask, desc))
            details.append({"raw": part, "parsed": {"col": col, "op": op, "rhs": rhs}, "n": int(mask.sum())})
            continue

        tokens = [t for t in re.split(r"[^\w]+", part) if t]
        if tokens:
            col = find_col(df, tokens) or find_col(df, ["_".join(tokens)])
            if col:
                ser = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                mask = (ser == 1)
                desc = f"{humanize_col(col)} = Yes"
                clauses.append((mask, desc))
                details.append({"raw": part, "parsed": {"col": col, "op": "=", "rhs": 1}, "n": int(mask.sum())})
            else:
                details.append({"raw": part, "parsed": None, "reason": "no column"})

    if not clauses:
        out = pd.Series(False, index=df.index)
        return (out, "No conditions parsed", details, normalized) if return_details else (out, "No conditions parsed")

    mask, desc_text = clauses[0]
    for (m, d), op in zip(clauses[1:], ops):
        if op == "AND":
            mask = mask & m; desc_text = f"{desc_text} AND {d}"
        else:
            mask = mask | m; desc_text = f"{desc_text} OR {d}"

    return (mask, desc_text, details, normalized) if return_details else (mask, desc_text)

# ------------------------------------------------------------
# Data load + filters
# ------------------------------------------------------------
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], key="upl_main")
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

# Profit label from Real CM2% Deviation if present
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

    # Bubble chart (present before)
    st.subheader("Contract Value vs CM2% Forecast (bubble = penalties)")
    df_bubble = df.copy()
    for c in ["contract_value", "cm2pct_forecast", "total_penalties"]:
        if c in df_bubble.columns:
            df_bubble[c] = pd.to_numeric(df_bubble[c], errors="coerce").fillna(0)
    df_bubble = df_bubble.dropna(subset=["contract_value", "cm2pct_forecast"])
    if not df_bubble.empty:
        df_bubble["has_penalty"] = df_bubble["total_penalties"] > 0
        figB = px.scatter(
            df_bubble,
            x="contract_value", y="cm2pct_forecast",
            size="total_penalties",
            color="country" if "country" in df_bubble.columns else None,
            hover_data=[c for c in ["project_id", "customer", "total_penalties", "cm2pct_forecast"] if c in df_bubble.columns],
            color_discrete_sequence=px.colors.qualitative.Set2,
            size_max=50
        )
        figB.update_traces(marker=dict(
            symbol=["square" if not p else "circle" for p in df_bubble["has_penalty"]],
            line=dict(width=0.4, color="rgba(0,0,0,0.3)")
        ))
        figB.update_layout(xaxis_title="Contract Value (EUR)", yaxis_title="CM2% Forecast")
        st.plotly_chart(figB, use_container_width=True, config=plotly_config("penalty_bubble"))

    # Margin Scatter (this was missing — restored)
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

    with st.expander("ℹ️ How to use + examples"):
        st.markdown("""
**What this answers**  
Probability a project ends **Non-Profitable** (outcome metric < 0) under your conditions.

### Write queries (plain language)
- `engineering hours is overrun and commissioning budget is overrun`
- `eng delay = yes` • `no delay in construction`
- `total penalties > 0` • `contract value > 1.5m`
- `cm2% forecast <= 12`
Use **AND / OR**, numbers with **k/m/bn**.
""")

    mode = st.radio("Mode", ["Simple (plain language)", "Advanced (A/B/C)"], horizontal=True, key="prob_mode")

    # ----------------- SIMPLE MODE -----------------
    if mode.startswith("Simple"):
        q = st.text_input("Ask in plain language",
                          placeholder="e.g., engineering hours is overrun and commissioning budget is overrun")
        # choose outcome (prefer Real CM2% Deviation)
        target_choices = [c for c in df.columns if "cm2" in c and "pct" in c] or list(df.columns)
        default_target = REAL_DEV_COL if (REAL_DEV_COL in target_choices) else target_choices[0]
        p_target_col = st.selectbox("Outcome metric defining Non-Profitable (< 0):",
                                    target_choices, index=target_choices.index(default_target),
                                    format_func=humanize_col, key="p_target_simple")
        target_flag = (pd.to_numeric(df[p_target_col], errors="coerce") < 0).astype(int)

        show_diag = st.checkbox("Show parsing details", value=True)
        if q:
            mask, desc, details, normalized = parse_query_to_mask(q, df, return_details=True)
            if show_diag:
                st.caption(f"Normalized: `{normalized}`")
                if details:
                    with st.expander("Recognized conditions & matches"):
                        for d in details:
                            if d.get("parsed"):
                                st.write(f"- **{d['raw']}** → `{d['parsed']['col']}` ({d['parsed']['op']} {d['parsed']['rhs']}) • matches: **{d.get('n',0)}**")
                            else:
                                st.write(f"- **{d['raw']}** → *(not recognized: {d.get('reason','?')})*")

            n = int(mask.sum())
            if n > 0:
                prob = target_flag[mask].mean() * 100
                base = target_flag.mean() * 100
                st.success(f"Probability project ends **Non-Profitable** if {desc}: **{prob:.1f}%**  (n={n})")
                if base > 0:
                    st.caption(f"Base rate: {base:.1f}% • Lift: {prob/base:.2f}×")
            else:
                st.warning("No rows matched—loosen conditions or check diagnostics.")
        st.divider()

    # ----------------- ADVANCED (A/B/C) -----------------
    if mode.endswith("(A/B/C)"):
        st.markdown("### Advanced")
        def ensure_defaults():
            ss = st.session_state
            ss.setdefault("p_target_col",
                REAL_DEV_COL or find_col(df, ["cm2pct","forecast"]) or (df.columns[0] if len(df.columns) else "x")
            )
            ss.setdefault("p_logic", "AND")
            ss.setdefault("p_use_c2", False)
            for k, v in {
                "p_c1_col": df.columns[0], "p_c1_op": "≥", "p_c1_thr": 0.0, "p_c1_flag_val": 1,
                "p_c2_col": df.columns[0], "p_c2_op": "≥", "p_c2_thr": 0.0, "p_c2_flag_val": 1
            }.items(): ss.setdefault(k, v)
            if "penalty_flag" not in df.columns and "total_penalties" in df.columns:
                df["penalty_flag"] = (pd.to_numeric(df["total_penalties"], errors="coerce").fillna(0) > 0).astype(int)
            numeric_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            flag_candidates = [c for c in df.columns if is_binary_series(df[c])]
            if numeric_candidates:
                ss.setdefault("b_num", numeric_candidates[0]); ss.setdefault("c_num", numeric_candidates[0])
            if flag_candidates:
                ss.setdefault("b_flag", flag_candidates[0]); ss.setdefault("c_flag", flag_candidates[0])
            ss.setdefault("b_op", "≥"); ss.setdefault("b_thr", 0.0)
            ss.setdefault("c_op", "≥"); ss.setdefault("c_thr", 0.0)
        ensure_defaults()

        def apply_preset(name: str):
            if name == "eng_delay_yes":
                col = "eng_delay" if "eng_delay" in df.columns else next((c for c in df.columns if "eng" in c and "delay" in c), df.columns[0])
                st.session_state.update({"p_use_c2": False,"p_c1_col": col,"p_c1_op": "=", "p_c1_thr": 1.0, "p_c1_flag_val": 1})
            elif name == "constr_bo_and_cpm_ho":
                c1 = "constr_b_o" if "constr_b_o" in df.columns else next((c for c in df.columns if "constr" in c and "b_o" in c), df.columns[0])
                c2 = "cpm_h_o" if "cpm_h_o" in df.columns else next((c for c in df.columns if "cpm" in c and "h_o" in c), df.columns[0])
                st.session_state.update({"p_use_c2": True, "p_logic": "AND",
                                         "p_c1_col": c1, "p_c1_op": ">", "p_c1_thr": 0.0,
                                         "p_c2_col": c2, "p_c2_op": ">", "p_c2_thr": 0.0})
            elif name == "penalty_given_proc_delay":
                proc_delay_col = "proc_delay" if "proc_delay" in df.columns else next((c for c in df.columns if "proc" in c and "delay" in c), df.columns[0])
                st.session_state.update({"b_num": "total_penalties" if "total_penalties" in df.columns else st.session_state["b_num"],
                                         "b_op": ">", "b_thr": 0.0, "b_flag": proc_delay_col})
            st.rerun()

        target_choices = [c for c in df.columns if "cm2" in c and "pct" in c] or list(df.columns)
        default_target = st.session_state.get("p_target_col", (REAL_DEV_COL if REAL_DEV_COL in target_choices else target_choices[0]))
        p_target_col = st.selectbox("Outcome metric defining Non-Profitable (< 0):",
                                    target_choices, index=target_choices.index(default_target), key="p_target_sel")
        target_flag = (pd.to_numeric(df[p_target_col], errors="coerce") < 0).astype(int)

        st.markdown("### A) Probability of **Non-Profitable** given conditions")
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
                ops = ["≥", ">", "≤", "<", "=", "=="]
                op = st.selectbox("operator", ops, index=ops.index(st.session_state.get("p_c1_op","≥")), key="p_c1_op_sel")
                thr = st.number_input("threshold", value=float(st.session_state.get("p_c1_thr",0.0)), step=1.0, key="p_c1_thr_num")
                c1_desc = f"{humanize_col(c1_col)} {op} {thr}"
                m1 = (to_flag(df[c1_col], op, float(thr)) == 1)

        with colB:
            use_c2 = st.checkbox("Add Condition 2", value=st.session_state.get("p_use_c2", False), key="p_use_c2_chk")
            logic = st.radio("Logic", ["AND", "OR"], horizontal=True,
                             index=(0 if st.session_state.get("p_logic","AND")=="AND" else 1), key="p_logic_radio")
            if use_c2:
                c2_idx = list(df.columns).index(st.session_state.get("p_c2_col", df.columns[0]))
                c2_col = st.selectbox("Condition 2", df.columns, index=c2_idx, format_func=humanize_col, key="p_c2_col_sel")
                if is_binary_series(df[c2_col]):
                    c2_val = st.selectbox("is", ["Yes (1)", "No (0)"], key="p_c2_flag_sel")
                    c2_flag_val = 1 if "Yes" in c2_val else 0
                    c2_desc = f"{humanize_col(c2_col)} = {'Yes' if c2_flag_val==1 else 'No'}"
                    m2 = (pd.to_numeric(df[c2_col], errors="coerce") == c2_flag_val)
                else:
                    ops = ["≥", ">", "≤", "<", "=", "=="]
                    op2 = st.selectbox("operator", ops, index=ops.index(st.session_state.get("p_c2_op","≥")), key="p_c2_op_sel")
                    thr2 = st.number_input("threshold", value=float(st.session_state.get("p_c2_thr",0.0)), step=1.0, key="p_c2_thr_num")
                    c2_desc = f"{humanize_col(c2_col)} {op2} {thr2}"
                    m2 = (to_flag(df[c2_col], op2, float(thr2)) == 1)
            else:
                m2 = pd.Series(True, index=df.index); c2_desc = "None"

        mask = (m1 | m2) if (use_c2 and logic == "OR") else (m1 & m2)
        cond_text = c1_desc if not use_c2 else f"{c1_desc} **{logic}** {c2_desc}"
        n = int(mask.sum())
        if n > 0:
            prob = target_flag[mask].mean() * 100
            st.markdown(
                f"""<div style="background:#eaf7ef;border:1px solid #bce3c7;padding:14px;border-radius:8px;font-size:1.05rem;">
                <b>Probability project ends Non-Profitable if {cond_text}: {prob:.1f}%</b>
                <span style="opacity:.65">(n={n})</span></div>""",
                unsafe_allow_html=True
            )
        else:
            st.info("No rows satisfy the chosen conditions.")

        st.markdown("#### Quick presets")
        p1,p2,p3 = st.columns(3)
        with p1:
            st.button("P(Non-Profitable | ENG delay=Yes)", use_container_width=True,
                      on_click=lambda: apply_preset("eng_delay_yes"))
        with p2:
            st.button("P(Non-Profitable | CONSTR b_o>0 AND CPM h_o>0)", use_container_width=True,
                      on_click=lambda: apply_preset("constr_bo_and_cpm_ho"))
        with p3:
            st.button("P(penalty>0 | PROC delay=Yes)", use_container_width=True,
                      on_click=lambda: apply_preset("penalty_given_proc_delay"))

        st.divider()

        # B) Numeric ↔ Flag (and reverse)
        st.markdown("### B) Numeric ↔ Flag (and reverse)")
        numeric_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        flag_candidates = [c for c in df.columns if is_binary_series(df[c])]
        bn_idx = numeric_candidates.index(st.session_state.get("b_num", numeric_candidates[0])) if numeric_candidates else 0
        bf_idx = flag_candidates.index(st.session_state.get("b_flag", flag_candidates[0])) if flag_candidates else 0
        colN, colF = st.columns(2)
        with colN:
            b_num = st.selectbox("Numeric variable", numeric_candidates if numeric_candidates else df.columns,
                                 index=bn_idx, format_func=humanize_col, key="p_b_num_sel")
            b_comp = st.selectbox("Compare", ["≥", ">", "≤", "<", "=", "=="],
                                  index=["≥",">","≤","<","=","=="].index(st.session_state.get("b_op","≥")),
                                  key="p_b_op_sel")
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
            c_op = st.selectbox("Operator", ["≥", ">", "≤", "<", "=", "=="],
                                index=["≥",">","≤","<","=","=="].index(st.session_state.get("c_op","≥")), key="p_c_op_sel")
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
            agg = pd.DataFrame({"mean": df[stats_cols].mean(numeric_only=True),
                                "median": df[stats_cols].median(numeric_only=True)})
            st.dataframe(agg.style.format({"mean": "{:.2f}", "median": "{:.2f}"}), use_container_width=True)
        else:
            st.info("No standard totals found (penalties / delays / overruns).")

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
# 6) Drivers (odds-like ranking over flags + general service impact)
# ------------------------------------------------------------
with tabs[5]:
    st.subheader("Drivers of CM2% deviation (service-level flags)")
    real_dev_col = REAL_DEV_COL or find_col(df, ["cm2pct","forecast"])  # fallback
    if real_dev_col is None or df[real_dev_col].dropna().nunique() == 0:
        st.info("Not enough variation to fit driver model.")
    else:
        y = (pd.to_numeric(df[real_dev_col], errors="coerce") < 0).astype(int)
        feat_df = pd.DataFrame(index=df.index); feat_cols = []
        for s in SERVICE_BLOCKS:
            for suffix, label in [("b_o","budget_overrun"), ("h_o","hours_overrun"), ("delay","delay")]:
                col = f"{s}_{suffix}"
                if col in df.columns:
                    fcol = f"{s}_{suffix}_flag"
                    if fcol in EXCLUDED_DRIVER_FLAGS:  # skip noisy ones
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
                title="Odds ratios by overrun/delay flag (↑ = higher risk of Non-Profitable)"
            )
            st.plotly_chart(fig7, use_container_width=True, config=plotly_config("drivers_features"))

            # ---- General (service-level) impact chart — restored
            feature_table["Impact"] = np.abs(np.log(feature_table["Odds_Ratio"].replace(0, np.nan))).fillna(0)
            service_impact = feature_table.groupby("Service")["Impact"].sum().reset_index().sort_values("Impact", ascending=False)
            fig8 = px.bar(
                service_impact, x="Service", y="Impact", color="Service",
                title="Service impact score (Σ |log-odds| across flags)",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig8.update_layout(showlegend=False)
            st.plotly_chart(fig8, use_container_width=True, config=plotly_config("drivers_services"))

# ------------------------------------------------------------
# 7) PATTERNS — shared variables & distributions (by CM2% Forecast)
# ------------------------------------------------------------
with tabs[6]:
    st.subheader("Patterns — what profitable vs non-profitable projects share (by CM2% Forecast)")
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
        thr = st.number_input("Profit threshold for CM2% Forecast ( > threshold = profitable )",
                              value=0.0, step=0.5)
    with col_top:
        topn = st.number_input("Top N patterns", min_value=3, max_value=20, value=8, step=1)

    ycls = (pd.to_numeric(dfx[fore_col], errors="coerce") > float(thr)).astype(int)
    dfx["_class"] = ycls.map({1: "Profitable", 0: "Non-Profitable"})
    pm, nm = (ycls == 1), (ycls == 0)

    # ---- Shared categorical (totals + any service flags)
    st.markdown("### Shared categorical variables (at-a-glance)")
    def any_suffix_flag(df_in: pd.DataFrame, suffix: str) -> pd.Series:
        acc = pd.Series(0, index=df_in.index)
        for s in SERVICE_BLOCKS:
            c = f"{s}_{suffix}"
            if c in df_in.columns:
                acc = acc | (pd.to_numeric(df_in[c], errors="coerce").fillna(0) > 0).astype(int)
        return acc.astype(int)

    cats = {}
    if "total_penalties" in dfx.columns:
        cats["Penalties > 0"] = (pd.to_numeric(dfx["total_penalties"], errors="coerce").fillna(0) > 0).astype(int)
    if "total_o" in dfx.columns:
        cats["Hours Overrun Total > 0"] = (pd.to_numeric(dfx["total_o"], errors="coerce").fillna(0) > 0).astype(int)
    if "total_delays" in dfx.columns:
        cats["Delays Total > 0"] = (pd.to_numeric(dfx["total_delays"], errors="coerce").fillna(0) > 0).astype(int)
    cats["Any Budget Overrun (any service)"] = any_suffix_flag(dfx, "b_o")
    cats["Any Hours Overrun (any service)"] = any_suffix_flag(dfx, "h_o")
    cats["Any Delay (any service)"] = any_suffix_flag(dfx, "delay")

    rows = []
    for label, ser in cats.items():
        r1 = ser[pm].mean() if pm.any() else 0.0
        r0 = ser[nm].mean() if nm.any() else 0.0
        rows.append({"Variable": label, "Rate_Profitable": r1, "Rate_NonProfitable": r0, "Delta_pp": r1 - r0})
    cat_df = pd.DataFrame(rows)
    if not cat_df.empty:
        cat_top = cat_df.reindex(cat_df["Delta_pp"].abs().sort_values(ascending=False).index).head(5)
        figc = px.bar(cat_top, x="Delta_pp", y="Variable", orientation="h",
                      title="Shared categorical patterns (Δ Profitable − Non-Profitable)")
        st.plotly_chart(figc, use_container_width=True, config=plotly_config("patterns_shared_cats"))
        st.dataframe(cat_top[["Variable","Rate_Profitable","Rate_NonProfitable","Delta_pp"]]
                     .style.format({"Rate_Profitable":"{:.2%}","Rate_NonProfitable":"{:.2%}","Delta_pp":"{:+.2%}"}),
                     use_container_width=True)
    else:
        st.info("No categorical totals/flags available.")

    # ---- Service-specific breakdown (b_o / h_o / delay)
    st.markdown("### Service-specific patterns (b_o / h_o / delay)")
    suffix = st.radio("Choose signal", ["b_o", "h_o", "delay"], horizontal=True, index=2)
    svc_rows = []
    for s in SERVICE_BLOCKS:
        c = f"{s}_{suffix}"
        if c in dfx.columns:
            ser = (pd.to_numeric(dfx[c], errors="coerce").fillna(0) > 0).astype(int)
            r1 = ser[pm].mean() if pm.any() else 0.0
            r0 = ser[nm].mean() if nm.any() else 0.0
            svc_rows.append({"Service": SERVICE_PRETTY.get(s, s.upper()),
                             "Rate_Profitable": r1, "Rate_NonProfitable": r0, "Delta_pp": r1 - r0})
    svc_df = pd.DataFrame(svc_rows)
    if not svc_df.empty:
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Most characteristic of PROFITABLE**")
            top_p = svc_df.sort_values("Delta_pp", ascending=False).head(6)
            fig_sp = px.bar(top_p, x="Delta_pp", y="Service", orientation="h",
                            title=f"{suffix.upper()} — Δ Profitable − Non-Profitable")
            st.plotly_chart(fig_sp, use_container_width=True, config=plotly_config("svc_prof"))
            st.dataframe(top_p.style.format({"Rate_Profitable":"{:.2%}","Rate_NonProfitable":"{:.2%}","Delta_pp":"{:+.2%}"}),
                         use_container_width=True)
        with colB:
            st.markdown("**Most characteristic of NON-PROFITABLE**")
            top_n = svc_df.sort_values("Delta_pp").head(6)
            fig_sn = px.bar(top_n.assign(Delta_to_Non=-top_n["Delta_pp"]), x="Delta_to_Non", y="Service", orientation="h",
                            title=f"{suffix.upper()} — Δ toward Non-Profitable")
            st.plotly_chart(fig_sn, use_container_width=True, config=plotly_config("svc_non"))
            st.dataframe(top_n.style.format({"Rate_Profitable":"{:.2%}","Rate_NonProfitable":"{:.2%}","Delta_pp":"{:+.2%}"}),
                         use_container_width=True)

    st.divider()

    # ---- Variable distribution (bell-style overlay)
    st.markdown("### Variable distribution by class")
    cand_vars = [c for c in dfx.columns if pd.api.types.is_numeric_dtype(dfx[c]) and c not in {fore_col}]
    aliases = [c for c in ["total_penalties", "total_delays", "total_o",
                           "eng_delay", "com_delay", "proc_b_o", "eng_b_o", "man_delay", "tpm_h_o"] if c in dfx.columns]
    pick = st.selectbox("Select variable", aliases + [c for c in cand_vars if c not in aliases], format_func=humanize_col)
    if pick:
        prof_vals = pd.to_numeric(dfx.loc[dfx["_class"]=="Profitable", pick], errors="coerce")
        nonp_vals = pd.to_numeric(dfx.loc[dfx["_class"]=="Non-Profitable", pick], errors="coerce")
        med_prof = prof_vals.median() if not prof_vals.dropna().empty else float("nan")
        med_nonp = nonp_vals.median() if not nonp_vals.dropna().empty else float("nan")
        st.write(f"Median {humanize_col(pick)} — Profitable: **{med_prof:.2f}**, Non-Profitable: **{med_nonp:.2f}**")
        figd = px.histogram(dfx, x=pick, color="_class", barmode="overlay", nbins=20,
                            category_orders={"_class":["Profitable","Non-Profitable"]},
                            labels={pick: humanize_col(pick), "_class":"Class"})
        figd.update_traces(opacity=0.70)
        st.plotly_chart(figd, use_container_width=True, config=plotly_config("var_bell"))

# ------------------------------------------------------------
# 8) Project Analyzer — heuristic + model probability + CM2% estimate
# ------------------------------------------------------------
with tabs[7]:
    st.subheader("Project Analyzer — heuristic + model probability + CM2% estimate")

    # Identify the project column
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
            sel_idx = row.index[0]
            r = row.iloc[0]

            # ---------------- Heuristic risk (5-level) ----------------
            score_box = {"v": 0}
            notes = []

            def add(flag, pts, label):
                if flag:
                    score_box["v"] += pts
                    notes.append(f"+{pts}: {label}")

            # Portfolio-level totals
            add((r.get("total_penalties", 0) or 0) > 0, 2, "Penalties present")
            add((r.get("total_o", 0) or 0) > 0, 1, "Hours overrun total > 0")
            add((r.get("total_delays", 0) or 0) > 0, 1, "Delays total > 0")

            # Any service flags
            def any_flag(row_s, suffix):
                for s_ in SERVICE_BLOCKS:
                    v_ = row_s.get(f"{s_}_{suffix}", 0)
                    if pd.to_numeric(pd.Series([v_]), errors="coerce").fillna(0).iloc[0] > 0:
                        return True
                return False

            add(any_flag(r, "b_o"), 2, "Budget overrun in at least one service")
            add(any_flag(r, "h_o"), 2, "Hours overrun in at least one service")
            add(any_flag(r, "delay"), 2, "Delay in at least one service")

            score = score_box["v"]
            if score <= 0:
                risk_band = "Very low"
            elif score <= 2:
                risk_band = "Low"
            elif score <= 4:
                risk_band = "Medium"
            elif score <= 7:
                risk_band = "High"
            else:
                risk_band = "Extremely high"

            st.metric("Heuristic risk of Non-Profitable outcome", risk_band)
            st.write("**Signals triggering risk:**")
            if notes:
                for n in notes: st.write("- ", n)
            else:
                st.write("- None (clean profile)")

            # Snapshot of key service flags
            focus = ["eng", "com", "proc", "man", "tpm"]
            cols = []
            for s_ in focus:
                cols += [c for c in [f"{s_}_b_o", f"{s_}_h_o", f"{s_}_delay"] if c in df.columns]
            snap = row[cols].T
            if not snap.empty:
                snap = snap.rename(index=lambda c: humanize_col(c))
                st.dataframe(snap, use_container_width=True)
                st.caption("Heuristic shown above; below: probability, risk levers, and CM2% estimate.")

            # ---------------- Probability P(Non-Profitable) (5-level) ----------------
            def band_from_prob(pct: float) -> str:
                if pct < 10:  return "Very low"
                if pct < 25:  return "Low"
                if pct < 50:  return "Medium"
                if pct < 75:  return "High"
                return "Extremely high"

            prob_neg_val = None
            if REAL_DEV_COL and df[REAL_DEV_COL].notna().sum() >= 5:
                y_all = (pd.to_numeric(df[REAL_DEV_COL], errors="coerce") < 0).astype(int)

                # Features: service flags + totals + size
                F = pd.DataFrame(index=df.index)
                for s_ in SERVICE_BLOCKS:
                    for suf_ in ["b_o", "h_o", "delay"]:
                        coln = f"{s_}_{suf_}"
                        if coln in df.columns:
                            F[f"{coln}_flag"] = (pd.to_numeric(df[coln], errors="coerce").fillna(0) > 0).astype(int)
                for cfeat in ["total_penalties", "total_delays", "total_o", "contract_value"]:
                    if cfeat in df.columns:
                        F[cfeat] = pd.to_numeric(df[cfeat], errors="coerce").fillna(0)
                F = F.fillna(0)

                if y_all.nunique() == 2 and F.shape[1] > 0:
                    try:
                        clf = LogisticRegression(max_iter=600, class_weight="balanced")
                        clf.fit(F, y_all)
                        prob_neg_val = float(clf.predict_proba(F.loc[[sel_idx]])[0, 1] * 100.0)
                        band = band_from_prob(prob_neg_val)
                        st.metric("Probability project ends Non-Profitable", f"{band} ({prob_neg_val:.1f}%)")
                    except Exception as e:
                        st.info(f"Couldn't fit probability model: {e}")
                else:
                    st.info("Not enough variation to fit probability model.")
            else:
                st.info("Outcome column not available to fit probability model.")
            st.session_state["prob_neg_val"] = prob_neg_val

            # ---------------- Heuristic risk levers ----------------
            st.markdown("#### Heuristic risk levers (what to change first)")
            levers_rows = []
            try:
                if REAL_DEV_COL and df[REAL_DEV_COL].notna().any():
                    y_all = (pd.to_numeric(df[REAL_DEV_COL], errors="coerce") < 0).astype(int)

                    F = pd.DataFrame(index=df.index)
                    for s_ in SERVICE_BLOCKS:
                        for suf_ in ["b_o", "h_o", "delay"]:
                            coln = f"{s_}_{suf_}"
                            if coln in df.columns:
                                F[f"{coln}_flag"] = (pd.to_numeric(df[coln], errors="coerce").fillna(0) > 0).astype(int)
                    for cfeat in ["total_penalties", "total_delays", "total_o", "contract_value"]:
                        if cfeat in df.columns:
                            F[cfeat] = pd.to_numeric(df[cfeat], errors="coerce").fillna(0)
                    F = F.fillna(0)

                    projF = F.loc[sel_idx]

                    # Flag levers: more common among negative outcomes
                    flag_cols = [c for c in F.columns if c.endswith("_flag")]
                    for c in flag_cols:
                        r_neg = F.loc[y_all == 1, c].mean() if (y_all == 1).any() else 0.0
                        r_pos = F.loc[y_all == 0, c].mean() if (y_all == 0).any() else 0.0
                        delta_to_neg = r_neg - r_pos
                        if projF[c] == 1 and delta_to_neg > 0:
                            levers_rows.append({
                                "Variable": humanize_col(c.replace("_flag", "")),
                                "Project": "Yes",
                                "Δ toward negative": delta_to_neg,
                                "Suggestion": "Avoid / resolve",
                            })

                    # Numeric totals levers
                    for c in ["total_penalties", "total_delays", "total_o"]:
                        if c in F.columns:
                            val = float(projF[c])
                            p_med = float(pd.to_numeric(df.loc[y_all == 0, c], errors="coerce").median())
                            n_med = float(pd.to_numeric(df.loc[y_all == 1, c], errors="coerce").median())
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
                st.info(f"Couldn't compute heuristic levers: {e}")

            # ---------------- Estimated CM2% (signals-based, calibrated & anchored) ----------------
            st.markdown("#### Estimated CM2% (signals-based)")

            # Prefer REAL outcome if present; fallback to forecast
            target_col = REAL_DEV_COL if (REAL_DEV_COL and df[REAL_DEV_COL].notna().sum() >= 8) else (
                find_col(df, ["cm2pct","forecast"]) or "cm2pct_forecast"
            )

            if target_col not in df.columns:
                st.info("No target column (real or forecast) found to train an estimator.")
            else:
                y_target = pd.to_numeric(df[target_col], errors="coerce")
                mask = y_target.notna()

                # Build features
                F_fore = pd.DataFrame(index=df.index)
                for s_ in SERVICE_BLOCKS:
                    for suf_ in ["b_o", "h_o", "delay"]:
                        coln_ = f"{s_}_{suf_}"
                        if coln_ in df.columns:
                            F_fore[f"{coln_}_flag"] = (pd.to_numeric(df[coln_], errors="coerce").fillna(0) > 0).astype(int)
                for cfeat_ in ["total_penalties", "total_delays", "total_o", "contract_value"]:
                    if cfeat_ in df.columns:
                        F_fore[cfeat_] = pd.to_numeric(df[cfeat_], errors="coerce").fillna(0)
                F_fore = F_fore.fillna(0)

                F_reg = F_fore.loc[mask]
                y_reg = y_target.loc[mask]

                if F_reg.shape[0] >= 8 and F_reg.shape[1] > 0:
                    try:
                        # Base model
                        reg = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
                        reg.fit(F_reg, y_reg)

                        # In-sample predictions
                        y_hat_train = reg.predict(F_reg)

                        # Linear calibration
                        try:
                            slope, intercept = np.polyfit(y_hat_train, y_reg, 1)
                        except Exception:
                            slope, intercept = 1.0, 0.0

                        # Quantile mapping (non-parametric correction)
                        qs = np.linspace(0, 1, 11)
                        edges = np.quantile(y_hat_train, qs)
                        edges = np.unique(edges)
                        if edges.size < 3:
                            edges = np.array([y_hat_train.min(), y_hat_train.mean(), y_hat_train.max()])
                        centers = (edges[:-1] + edges[1:]) / 2

                        med_targets = []
                        for i in range(len(edges) - 1):
                            lo, hi = edges[i], edges[i+1]
                            if i == len(edges) - 2:
                                sel_bin = (y_hat_train >= lo) & (y_hat_train <= hi)
                            else:
                                sel_bin = (y_hat_train >= lo) & (y_hat_train < hi)
                            m = np.median(y_reg[sel_bin]) if sel_bin.any() else np.nan
                            med_targets.append(m)
                        med_targets = np.array(med_targets)
                        if np.isnan(med_targets).any():
                            ok = ~np.isnan(med_targets)
                            med_targets = np.interp(np.arange(len(med_targets)), np.where(ok)[0], med_targets[ok])

                        # Predict for selected project
                        base_pred = float(reg.predict(F_fore.loc[[sel_idx]])[0])

                        # Apply calibration and quantile mapping
                        cal_pred = slope * base_pred + intercept
                        qm_pred = np.interp(base_pred, centers, med_targets, left=med_targets[0], right=med_targets[-1])

                        # Blend (give more weight to non-param mapping)
                        yhat = 0.6 * qm_pred + 0.4 * cal_pred

                        # Anchor by P(Non-Profitable) if available and REAL target exists
                        pneg = st.session_state.get("prob_neg_val", None)
                        if pneg is not None and REAL_DEV_COL:
                            y_all_full = (pd.to_numeric(df[REAL_DEV_COL], errors="coerce") < 0).astype(int)
                            pos_med = float(pd.to_numeric(y_target[y_all_full == 0], errors="coerce").median())
                            neg_med = float(pd.to_numeric(y_target[y_all_full == 1], errors="coerce").median())
                            p = max(0.0, min(1.0, pneg / 100.0))
                            if p >= 0.5:
                                w = min(1.0, (p - 0.5) / 0.35)  # ramps 50%→85% to w≈1
                                yhat = (1 - w) * yhat + w * neg_med
                            else:
                                w = min(1.0, (0.5 - p) / 0.35)
                                yhat = (1 - w) * yhat + w * pos_med)

                        # Clip to sensible range and soften extremes
                        q02, q50, q98 = np.nanpercentile(y_reg, [2, 50, 98])
                        yhat = max(q02, min(q98, yhat))
                        yhat = q50 + 0.85 * (yhat - q50)

                        st.metric(
                            f"Estimated CM2% ({'real' if target_col==REAL_DEV_COL else 'forecast'} target, calibrated)",
                            f"{yhat:.1f}%"
                        )
                        # Optional: quick R² without relying on sklearn.metrics
                        def _r2(y_true, y_pred):
                            y_true = np.asarray(y_true, dtype=float)
                            y_pred = np.asarray(y_pred, dtype=float)
                            if y_true.size < 2 or not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
                                return np.nan
                            ss_res = np.sum((y_true - y_pred) ** 2)
                            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
                            return 1.0 * (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
                        r2 = _r2(y_reg, y_hat_train)
                        st.caption(
                            f"Targets real CM2% when available; otherwise forecast. "
                            f"Linear + quantile calibration, probability-anchored; clipped to 2–98th pct. "
                            f"In-sample R² = {('—' if (r2 is None or (isinstance(r2, float) and np.isnan(r2))) else f'{r2:.2f}')}."
                        )
                    except Exception as e:
                        st.info(f"Couldn't fit CM2% estimator: {e}")
                else:
                    st.info("Not enough rows/columns to fit CM2% estimator.")
