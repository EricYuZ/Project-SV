"""
Surveillance Tool - Streamlit Dashboard
=======================================

Reads `surveillance_dashboard_input.csv` from the data prep package, computes
the 9 indicator scores + composite Surveillance Score in the dashboard layer
(per the user's design choice that scoring lives here, not in the data pull),
and renders 4 pages exactly as described in
`surveillance_tool_project_documentation.md` Section 14:

    Page 1 - Executive Portfolio Overview
    Page 2 - Company Drilldown
    Page 3 - Stress Testing
    Page 4 - Indicator Methodology
    Page 5 - Data Coverage & Assumptions

Run
---
    streamlit run surveillance_dashboard/app.py
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import textwrap
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scoring import (
    ALERT_BINS,
    ALERT_COLORS,
    INDICATOR_FIELDS,
    INDICATOR_LABELS,
    INDICATOR_WEIGHTS,
    SECTOR_BASE,
    TIER_COLORS,
    TIER_LABELS,
    assign_alert,
    assign_tier,
    compute_scores,
    explanation_for_company,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Surveillance Tool",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "ewif_public_data_prep_package" / "output" / "surveillance_dashboard_input.csv"
DEFAULT_DIAGNOSTICS = REPO_ROOT / "ewif_public_data_prep_package" / "output" / "surveillance_data_diagnostics.csv"
DEFAULT_HISTORY = REPO_ROOT / "ewif_public_data_prep_package" / "output" / "surveillance_quarterly_history.csv"
DEFAULT_QUARTERLY_SCORES = REPO_ROOT / "ewif_public_data_prep_package" / "output" / "surveillance_quarterly_scores.csv"

st.markdown(
    """
    <style>
    .calc-shell {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 16px;
        background: #ffffff;
        margin: 8px 0 12px 0;
    }
    .calc-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 8px;
    }
    .calc-title h4 {
        margin: 0;
        font-size: 1rem;
        line-height: 1.25;
    }
    .calc-score {
        min-width: 64px;
        text-align: center;
        border-radius: 999px;
        color: #ffffff;
        font-weight: 700;
        padding: 4px 10px;
    }
    .calc-meta {
        color: #4b5563;
        font-size: 0.86rem;
        margin-bottom: 8px;
    }
    .formula-box {
        border-left: 4px solid #2563eb;
        background: #f8fafc;
        border-radius: 6px;
        padding: 10px 12px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.86rem;
        color: #111827;
        overflow-wrap: anywhere;
        margin: 8px 0 10px 0;
    }
    [data-testid="stSidebar"] {
        background: #f8fafc;
    }
    .sidebar-panel {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 10px 12px;
        background: #ffffff;
        margin: 8px 0 12px 0;
    }
    .sidebar-panel-title {
        color: #111827;
        font-size: 0.88rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .sidebar-panel-caption {
        color: #6b7280;
        font-size: 0.78rem;
        line-height: 1.25;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_data(
    input_path: str,
    diag_path: str,
    history_path: str,
    quarterly_score_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_df = pd.read_csv(input_path)
    try:
        hist = pd.read_csv(history_path)
        hist["period_end"] = pd.to_datetime(hist["period_end"], errors="coerce")
    except FileNotFoundError:
        hist = pd.DataFrame(columns=["ticker", "company_name", "metric", "value_type", "period_end", "value", "source_tag", "form", "filed"])
    # Pass history into compute_scores so Beneish M-Score and velocity-trigger
    # tiers can be derived (EWIF v1.0 Sections V.C and VI.B).
    base_df = compute_scores(raw_df, history_df=hist)
    try:
        diag = pd.read_csv(diag_path)
    except FileNotFoundError:
        diag = pd.DataFrame(columns=["ticker", "company_name", "sector_group", "metric", "flag_type"])
    try:
        quarterly_scores = pd.read_csv(quarterly_score_path)
    except FileNotFoundError:
        quarterly_scores = pd.DataFrame()
    return raw_df, base_df, diag, hist, quarterly_scores


SCENARIO_RAW_FIELD_GROUPS: List[Dict[str, object]] = [
    {
        "indicator": "facility_liquidity_score",
        "fields": [
            {"column": "cash", "label": "Cash"},
            {"column": "current_assets", "label": "Current assets"},
            {"column": "current_liabilities", "label": "Current liabilities"},
        ],
    },
    {
        "indicator": "financial_performance_score",
        "fields": [
            {"column": "revenue", "label": "Revenue"},
            {"column": "net_income", "label": "Net income"},
            {"column": "assets", "label": "Assets"},
            {"column": "liabilities", "label": "Liabilities"},
            {"column": "total_debt", "label": "Total debt"},
            {"column": "cfo", "label": "Cash flow from operations"},
            {"column": "capex", "label": "Capex"},
            {"column": "interest_expense", "label": "Interest expense"},
            {"column": "depreciation", "label": "Depreciation"},
        ],
    },
    {
        "indicator": "behavioral_payment_score",
        "fields": [],
    },
    {
        "indicator": "market_implied_risk_score",
        "fields": [
            {"column": "ret_3m", "label": "3-month return"},
            {"column": "vol_3m", "label": "3-month annualized volatility"},
            {"column": "drawdown_6m", "label": "6-month drawdown"},
        ],
    },
    {
        "indicator": "news_sentiment_score",
        "fields": [],
    },
    {
        "indicator": "accounting_integrity_score",
        "fields": [
            {"column": "accounts_receivable", "label": "Accounts receivable"},
        ],
    },
    {
        "indicator": "connectivity_contagion_score",
        "fields": [],
    },
    {
        "indicator": "governance_discipline_score",
        "fields": [],
    },
    {
        "indicator": "collateral_recovery_score",
        "fields": [],
    },
]


def _safe_div(num, den):
    if pd.isna(num) or pd.isna(den) or den == 0:
        return np.nan
    return float(num) / float(den)


def _recompute_latest_snapshot_ratios(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild latest-snapshot derived fields after scenario edits."""
    out = raw_df.copy()
    numeric = [
        "revenue", "net_income", "assets", "liabilities", "cash", "cfo", "capex",
        "current_assets", "current_liabilities", "total_debt",
    ]
    for col in numeric:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["fcf"] = out["cfo"] - out["capex"]
    out["leverage_proxy"] = [_safe_div(n, d) for n, d in zip(out["liabilities"], out["assets"])]
    out["debt_to_assets"] = [_safe_div(n, d) for n, d in zip(out["total_debt"], out["assets"])]
    out["cash_to_assets"] = [_safe_div(n, d) for n, d in zip(out["cash"], out["assets"])]
    out["net_margin"] = [_safe_div(n, d) for n, d in zip(out["net_income"], out["revenue"])]
    out["fcf_to_assets"] = [_safe_div(n, d) for n, d in zip(out["fcf"], out["assets"])]
    out["accrual_proxy"] = [
        _safe_div(ni - cfo, assets)
        for ni, cfo, assets in zip(out["net_income"], out["cfo"], out["assets"])
    ]
    out["current_ratio"] = [
        _safe_div(n, d)
        for n, d in zip(out["current_assets"], out["current_liabilities"])
    ]
    return out


def apply_scenario_overrides(raw_df: pd.DataFrame, overrides: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    if not overrides:
        return raw_df.copy()

    out = raw_df.copy()
    for ticker, values in overrides.items():
        mask = out["ticker"] == ticker
        if not mask.any():
            continue
        for col, value in values.items():
            if col in out.columns:
                out.loc[mask, col] = value
    return _recompute_latest_snapshot_ratios(out)


def _bounded_score(value) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return max(0.0, min(100.0, float(value)))


def apply_indicator_score_overrides(scored_df: pd.DataFrame, overrides: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    """Apply direct factor-score overlays after formula scoring."""
    if not overrides:
        return scored_df.copy()

    out = scored_df.copy()
    for ticker, values in overrides.items():
        mask = out["ticker"] == ticker
        if not mask.any():
            continue
        for ind in INDICATOR_FIELDS:
            if ind not in values:
                continue
            score = _bounded_score(values.get(ind))
            if pd.notna(score):
                out.loc[mask, ind] = score

    out["raw_surveillance_score"] = sum(
        out[ind] * w for ind, w in INDICATOR_WEIGHTS.items()
    ).clip(0, 100)
    out["surveillance_score"] = out["raw_surveillance_score"]
    out["ewif_score"] = out["surveillance_score"]
    out["alert_level"] = out["surveillance_score"].apply(assign_alert)

    tiers: List[str] = []
    top_drivers: List[str] = []
    top_driver_lists = []
    for _, row in out.iterrows():
        ind_scores = {ind: row.get(ind) for ind in INDICATOR_FIELDS}
        tiers.append(
            assign_tier(
                score=row["surveillance_score"],
                indicator_scores=ind_scores,
                velocity_trigger=bool(row.get("velocity_trigger", False)),
            )
        )
        weakest = sorted(
            ((ind, float(row[ind])) for ind in INDICATOR_FIELDS),
            key=lambda x: x[1],
        )[:3]
        top_driver_lists.append(weakest)
        top_drivers.append(
            "; ".join(f"{INDICATOR_LABELS[ind]}: {val:.0f}" for ind, val in weakest)
        )
    out["alert_tier"] = tiers
    out["top_driver_objects"] = top_driver_lists
    out["top_risk_drivers"] = top_drivers
    return out


def ensure_fraud_columns(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Backfill fraud diagnostic columns for cached/older scored frames."""
    out = scored_df.copy()
    if "score_sloan_accruals" not in out.columns:
        out["score_sloan_accruals"] = np.nan
    if "score_beneish" not in out.columns:
        out["score_beneish"] = np.nan
    if "accounting_integrity_score" not in out.columns:
        out["accounting_integrity_score"] = 50.0

    if "fraud_quality_subscore" not in out.columns:
        out["fraud_quality_subscore"] = out[["score_sloan_accruals", "score_beneish"]].mean(axis=1)
        out["fraud_quality_subscore"] = out["fraud_quality_subscore"].fillna(out["accounting_integrity_score"])
    if "fraud_risk_subscore" not in out.columns:
        out["fraud_risk_subscore"] = (100 - out["fraud_quality_subscore"]).clip(0, 100)
    if "fraud_watch_flag" not in out.columns:
        beneish = pd.to_numeric(out.get("beneish_m", pd.Series(np.nan, index=out.index)), errors="coerce")
        out["fraud_watch_flag"] = (
            (beneish > -1.78)
            | (pd.to_numeric(out["score_sloan_accruals"], errors="coerce") <= 25)
            | (pd.to_numeric(out["accounting_integrity_score"], errors="coerce") <= 35)
        )
    return out


# ---------------------------------------------------------------------------
# Sidebar - data source + filters
# ---------------------------------------------------------------------------

st.sidebar.title("Surveillance Tool")
st.sidebar.caption("Public-data proof of concept")

with st.sidebar.expander("Data source", expanded=False):
    input_path = st.text_input(
        "Dashboard input CSV", value=str(DEFAULT_INPUT)
    )
    diag_path = st.text_input(
        "Diagnostics CSV", value=str(DEFAULT_DIAGNOSTICS)
    )
    history_path = st.text_input(
        "Quarterly history CSV", value=str(DEFAULT_HISTORY)
    )
    quarterly_score_path = st.text_input(
        "Quarterly scores CSV", value=str(DEFAULT_QUARTERLY_SCORES)
    )

if not Path(input_path).exists():
    st.error(
        f"Could not find dashboard input at `{input_path}`. "
        "Run `python build_surveillance_data.py` in the data prep package first."
    )
    st.stop()

raw_df, base_df, diag, hist, quarterly_scores = load_data(
    input_path, diag_path, history_path, quarterly_score_path
)
base_df = ensure_fraud_columns(base_df)
if not quarterly_scores.empty:
    quarterly_scores = ensure_fraud_columns(quarterly_scores)
if "scenario_overrides" not in st.session_state:
    st.session_state["scenario_overrides"] = {}

scenario_overrides: Dict[str, Dict[str, object]] = st.session_state["scenario_overrides"]
scenario_active = bool(scenario_overrides)
scenario_raw_df = apply_scenario_overrides(raw_df, scenario_overrides)
df = compute_scores(scenario_raw_df, history_df=hist) if scenario_active else base_df.copy()
df = apply_indicator_score_overrides(df, scenario_overrides) if scenario_active else df
df = ensure_fraud_columns(df)

st.sidebar.markdown("---")
FILTER_KEYS = [
    "filter_company_query_v2",
    "filter_index_v2",
    "filter_sector_v2",
    "filter_industry_v2",
    "filter_alert_v2",
    "filter_tier_v2",
    "filter_score_v2",
    "filter_data_quality_v2",
    "filter_stale_max_v2",
    "filter_assumption_max_v2",
    "filter_velocity_v2",
    "filter_sec_available_only_v2",
    "filter_market_available_only_v2",
    "filter_scenario_only_v2",
]

if scenario_active:
    st.sidebar.info(f"Scenario mode active for {len(scenario_overrides)} ticker(s). Source CSV is unchanged.")
    if st.sidebar.button("Clear all scenario edits", use_container_width=True):
        st.session_state["scenario_overrides"] = {}
        st.rerun()

if st.sidebar.button("Reset filters", use_container_width=True):
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)
    st.rerun()

st.sidebar.markdown(
    """
    <div class="sidebar-panel">
        <div class="sidebar-panel-title">Portfolio Filters</div>
        <div class="sidebar-panel-caption">Narrow the portfolio by company, peer group, risk level, and data quality.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

company_query = st.sidebar.text_input(
    "Company search",
    value="",
    placeholder="Ticker or company name",
    key="filter_company_query_v2",
)

index_options = ["S&P 500", "Nasdaq 100", "Dow"]
if "index_memberships" in df.columns:
    selected_indices = st.sidebar.multiselect(
        "Index",
        options=index_options,
        default=index_options,
        key="filter_index_v2",
        help="Shows the full major-index universe. Companies can belong to more than one index.",
    )
else:
    selected_indices = index_options

sector_options = sorted(df["sector_group"].dropna().unique().tolist())
selected_sectors = st.sidebar.multiselect(
    "Sector",
    options=sector_options,
    default=sector_options,
    key="filter_sector_v2",
)
industry_source = df[df["sector_group"].isin(selected_sectors)] if selected_sectors else df
industry_options = sorted(industry_source["industry_group"].dropna().unique().tolist())
selected_industries = st.sidebar.multiselect(
    "Industry",
    options=industry_options,
    default=industry_options,
    key="filter_industry_v2",
)

alert_options = ["Green", "Yellow", "Orange", "Red"]
selected_alerts = st.sidebar.multiselect(
    "Alert level",
    options=alert_options,
    default=alert_options,
    key="filter_alert_v2",
)
tier_options = ["Tier 1", "Tier 2", "Tier 3", "None"]
selected_tiers = st.sidebar.multiselect(
    "Action tier",
    options=tier_options,
    default=tier_options,
    key="filter_tier_v2",
)

score_range = st.sidebar.slider(
    "Surveillance score",
    min_value=0,
    max_value=100,
    value=(0, 100),
    key="filter_score_v2",
)
data_quality_range = st.sidebar.slider(
    "Data quality",
    min_value=0,
    max_value=100,
    value=(0, 100),
    key="filter_data_quality_v2",
)

max_stale = int(pd.to_numeric(df.get("stale_fact_count", pd.Series([0])), errors="coerce").fillna(0).max())
max_assumptions = int(pd.to_numeric(df.get("assumption_metric_count", pd.Series([0])), errors="coerce").fillna(0).max())
stale_max = st.sidebar.slider(
    "Maximum stale facts",
    min_value=0,
    max_value=max(max_stale, 0),
    value=max(max_stale, 0),
    key="filter_stale_max_v2",
)
assumption_max = st.sidebar.slider(
    "Maximum assumption fills",
    min_value=0,
    max_value=max(max_assumptions, 0),
    value=max(max_assumptions, 0),
    key="filter_assumption_max_v2",
)

with st.sidebar.expander("Advanced filters", expanded=False):
    velocity_filter = st.radio(
        "Velocity trigger",
        options=["All", "Triggered only", "Not triggered"],
        horizontal=False,
        key="filter_velocity_v2",
    )
    sec_available_only = st.checkbox(
        "SEC available only",
        value=False,
        key="filter_sec_available_only_v2",
    )
    market_available_only = st.checkbox(
        "Market data available only",
        value=False,
        key="filter_market_available_only_v2",
    )
    scenario_only = st.checkbox(
        "Scenario-edited names only",
        value=False,
        key="filter_scenario_only_v2",
        disabled=not scenario_active,
    )

mask = pd.Series(True, index=df.index)

if selected_sectors and set(selected_sectors) != set(sector_options):
    mask &= df["sector_group"].isin(selected_sectors)
if selected_industries and set(selected_industries) != set(industry_options):
    mask &= df["industry_group"].isin(selected_industries)
if selected_alerts and set(selected_alerts) != set(alert_options):
    mask &= df["alert_level"].isin(selected_alerts)
if selected_tiers and set(selected_tiers) != set(tier_options):
    mask &= df["alert_tier"].isin(selected_tiers)
if tuple(score_range) != (0, 100):
    mask &= df["surveillance_score"].between(score_range[0], score_range[1])
if tuple(data_quality_range) != (0, 100):
    mask &= df["data_quality_score"].between(data_quality_range[0], data_quality_range[1])
if stale_max < max(max_stale, 0):
    mask &= pd.to_numeric(df["stale_fact_count"], errors="coerce").fillna(0).le(stale_max)
if assumption_max < max(max_assumptions, 0):
    mask &= pd.to_numeric(df["assumption_metric_count"], errors="coerce").fillna(0).le(assumption_max)

if (
    "index_memberships" in df.columns
    and selected_indices
    and set(selected_indices) != set(index_options)
):
    index_mask = pd.Series(False, index=df.index)
    for index_name in selected_indices:
        index_mask |= df["index_memberships"].fillna("").str.contains(index_name, regex=False)
    mask &= index_mask

if company_query.strip():
    q = company_query.strip().lower()
    mask &= (
        df["ticker"].astype(str).str.lower().str.contains(q, regex=False)
        | df["company_name"].astype(str).str.lower().str.contains(q, regex=False)
    )

if velocity_filter == "Triggered only":
    mask &= df["velocity_trigger"].fillna(False)
elif velocity_filter == "Not triggered":
    mask &= ~df["velocity_trigger"].fillna(False)

if sec_available_only and "sec_available" in df.columns:
    mask &= df["sec_available"].fillna(False)
if market_available_only and "market_data_available" in df.columns:
    mask &= df["market_data_available"].fillna(False)
if scenario_only and scenario_active:
    mask &= df["ticker"].isin(scenario_overrides.keys())

fdf = df[mask].copy()

st.sidebar.markdown(
    f"""
    <div class="sidebar-panel">
        <div class="sidebar-panel-title">Current View</div>
        <div class="sidebar-panel-caption">{len(fdf):,} of {len(df):,} companies selected</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def alert_chip(alert: str) -> str:
    color = ALERT_COLORS.get(alert, "#6b7280")
    return (
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:10px;font-weight:600;font-size:0.85em;">{alert}</span>'
    )


def tier_chip(tier: str) -> str:
    color = TIER_COLORS.get(tier, "#6b7280")
    label = tier if tier != "None" else "OK"
    return (
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:10px;font-weight:600;font-size:0.85em;">{label}</span>'
    )


def gauge_chart(score: float, alert: str) -> go.Figure:
    color = ALERT_COLORS.get(alert, "#6b7280")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 44}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.4},
                "steps": [
                    {"range": [0, 35], "color": "#fee2e2"},
                    {"range": [35, 45], "color": "#ffedd5"},
                    {"range": [45, 55], "color": "#fef9c3"},
                    {"range": [55, 100], "color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.85,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=270)
    return fig


def indicator_bar(row: pd.Series) -> go.Figure:
    labels = [INDICATOR_LABELS[ind] for ind in INDICATOR_FIELDS]
    values = [float(row[ind]) for ind in INDICATOR_FIELDS]
    colors = []
    for v in values:
        if v >= 55:
            colors.append(ALERT_COLORS["Green"])
        elif v >= 45:
            colors.append(ALERT_COLORS["Yellow"])
        elif v >= 35:
            colors.append(ALERT_COLORS["Orange"])
        else:
            colors.append(ALERT_COLORS["Red"])
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.0f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 110], title="Indicator score (0 = weak/high risk, 100 = strong/low risk)"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
    )
    return fig


def _score_color(score: float) -> str:
    if pd.isna(score):
        return "#6b7280"
    if score >= 55:
        return ALERT_COLORS["Green"]
    if score >= 45:
        return ALERT_COLORS["Yellow"]
    if score >= 35:
        return ALERT_COLORS["Orange"]
    return ALERT_COLORS["Red"]


def _fmt_calc(value, kind: str = "number") -> str:
    if value is None or pd.isna(value):
        return "-"
    if kind == "bool":
        return "Yes" if bool(value) else "No"
    v = float(value)
    if kind == "money":
        sign = "-" if v < 0 else ""
        av = abs(v)
        if av >= 1e9:
            return f"{sign}${av/1e9:,.2f}B"
        if av >= 1e6:
            return f"{sign}${av/1e6:,.2f}M"
        return f"{sign}${av:,.0f}"
    if kind == "pct":
        return f"{v * 100:+.2f}%"
    if kind == "ratio":
        return f"{v:.3f}"
    if kind == "score":
        return f"{v:.1f}"
    if kind == "int":
        return f"{int(round(v))}"
    return f"{v:,.3f}" if abs(v) < 100 else f"{v:,.0f}"


def _editor_value(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.6g}"


def _parse_editor_number(text: str):
    cleaned = str(text).strip().replace(",", "").replace("$", "")
    if cleaned == "":
        return np.nan
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1].strip()
    try:
        value = float(cleaned)
    except ValueError:
        return np.nan
    return value / 100 if is_percent else value


def _parse_score_input(text: str):
    cleaned = str(text).strip().replace(",", "")
    if cleaned == "":
        return np.nan
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1].strip()
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def _calc_row(
    name: str,
    origin: str,
    value,
    kind: str,
    subscore=None,
    role: str = "",
    conversion: str = "",
) -> Dict[str, str]:
    if not conversion and subscore is not None and not pd.isna(subscore):
        role_lower = role.lower()
        if any(token in role_lower for token in ["lower", "fewer", "smaller"]):
            conversion = "Absolute threshold curve: lower raw value maps to a higher score."
        elif "higher" in role_lower or "stronger" in role_lower:
            conversion = "Absolute threshold curve: higher raw value maps to a higher score."
        elif "blend" in role_lower or "average" in role_lower:
            conversion = "Blended from the listed component sub-scores."
        else:
            conversion = "Converted with the scoring rule shown in the formula above."
    return {
        "Input": name,
        "Origin data": origin,
        "Value": _fmt_calc(value, kind),
        "Sub-score": "-" if subscore is None or pd.isna(subscore) else _fmt_calc(subscore, "score"),
        "Value -> sub-score": conversion or "-",
        "Role in formula": role,
    }


def _render_formula_card(
    title: str,
    score: float,
    formula: str,
    source: str,
    rows: List[Dict[str, str]],
    note: str = "",
    table_key: str | None = None,
) -> None:
    color = _score_color(score)
    st.markdown(
        f"""
        <div class="calc-shell">
            <div class="calc-title">
                <h4>{title}</h4>
                <span class="calc-score" style="background:{color};">{_fmt_calc(score, "score")}</span>
            </div>
            <div class="calc-meta">{source}</div>
            <div class="formula-box">{formula}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if note:
        st.caption(note)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, key=table_key)


def _render_scenario_editor(selected: str, scenario_row: pd.Series, base_row: pd.Series) -> None:
    active_for_name = selected in st.session_state["scenario_overrides"]

    with st.expander("Scenario editor - change company inputs and recalculate", expanded=active_for_name):
        st.caption(
            "Edit latest-snapshot inputs for this company, then apply. The dashboard reruns scoring against "
            "the same loaded universe. Changes are session-only and do not modify the CSV."
        )

        if active_for_name:
            d1, d2, d3, d4 = st.columns(4)
            d1.metric(
                "Score change",
                f"{scenario_row['surveillance_score']:.1f}",
                f"{scenario_row['surveillance_score'] - base_row['surveillance_score']:+.1f}",
            )
            d2.metric(
                "Financial Performance",
                f"{scenario_row['financial_performance_score']:.1f}",
                f"{scenario_row['financial_performance_score'] - base_row['financial_performance_score']:+.1f}",
            )
            d3.metric("Scenario tier", scenario_row["alert_tier"])
            d4.metric("Base tier", base_row["alert_tier"])

        sector_options_for_edit = sorted(SECTOR_BASE.keys())
        if str(scenario_row.get("sector_group")) not in sector_options_for_edit:
            sector_options_for_edit.append(str(scenario_row.get("sector_group")))
        if str(base_row.get("sector_group")) not in sector_options_for_edit:
            sector_options_for_edit.append(str(base_row.get("sector_group")))

        with st.form(f"scenario_form_{selected}"):
            sector_value = st.selectbox(
                "Sector group",
                options=sector_options_for_edit,
                index=sector_options_for_edit.index(str(base_row.get("sector_group"))),
                help="Feeds sector base risk and sector-density calculations.",
            )

            st.markdown("###### Raw input overrides by indicator")
            edited_values: Dict[str, str] = {}
            proxy_score_values: Dict[str, str] = {}
            override_values = st.session_state["scenario_overrides"].get(selected, {})
            for group in SCENARIO_RAW_FIELD_GROUPS:
                indicator = str(group["indicator"])
                fields = list(group["fields"])
                with st.expander(INDICATOR_LABELS[indicator], expanded=bool(fields)):
                    if not fields:
                        current_proxy = override_values.get(indicator, base_row.get(indicator))
                        proxy_score_values[indicator] = st.text_input(
                            "Proxy score override",
                            value=_editor_value(current_proxy),
                            key=f"scenario_{selected}_{indicator}_proxy_score",
                            help=(
                                "This indicator is derived from proxy factors. Edit the 0-100 proxy score here "
                                "when you want to override that derived value. Higher = stronger credit quality."
                            ),
                        )
                        continue
                    cols = st.columns(3)
                    for idx, spec in enumerate(fields):
                        col = str(spec["column"])
                        with cols[idx % 3]:
                            edited_values[col] = st.text_input(
                                str(spec["label"]),
                                value=_editor_value(base_row.get(col)),
                                key=f"scenario_{selected}_{col}",
                                help=(
                                    "Original latest-snapshot value. Use raw units. For returns/ratios, "
                                    "enter decimals such as 0.12 or percentages such as 12%."
                                ),
                            )

            b1, b2, _ = st.columns([1, 1, 4])
            apply_clicked = b1.form_submit_button("Apply scenario", use_container_width=True)
            reset_clicked = b2.form_submit_button("Reset company", use_container_width=True)

        if apply_clicked:
            overrides = dict(st.session_state["scenario_overrides"])
            next_values: Dict[str, object] = {
                "sector_group": sector_value,
            }
            for col, text in edited_values.items():
                next_values[col] = _parse_editor_number(text)
            for indicator, text in proxy_score_values.items():
                parsed = _parse_score_input(text)
                base_proxy = base_row.get(indicator)
                if pd.notna(parsed) and (pd.isna(base_proxy) or abs(float(parsed) - float(base_proxy)) > 1e-6):
                    next_values[indicator] = _bounded_score(parsed)
            overrides[selected] = next_values
            st.session_state["scenario_overrides"] = overrides
            st.rerun()

        if reset_clicked:
            overrides = dict(st.session_state["scenario_overrides"])
            overrides.pop(selected, None)
            st.session_state["scenario_overrides"] = overrides
            st.rerun()


def _render_composite_calc(row: pd.Series, period_label: str = "Current", table_key: str | None = None) -> None:
    rows = []
    for ind in INDICATOR_FIELDS:
        score = float(row[ind])
        weight = INDICATOR_WEIGHTS[ind]
        rows.append(
            {
                "Input": INDICATOR_LABELS[ind],
                "Origin data": "Indicator score calculated below",
                "Value": _fmt_calc(score, "score"),
                "Weight": f"{weight * 100:.0f}%",
                "Contribution": f"{score * weight:.2f}",
            }
        )
    st.markdown(
        f"""
        <div class="calc-shell">
            <div class="calc-title">
                <h4>Composite Surveillance Score - {period_label}</h4>
                <span class="calc-score" style="background:{_score_color(row['surveillance_score'])};">{_fmt_calc(row['surveillance_score'], "score")}</span>
            </div>
            <div class="calc-meta">The displayed score is the weighted average of absolute indicator scores. Higher = stronger credit quality / lower risk.</div>
            <div class="formula-box">surveillance_score = sum(indicator_score_i * indicator_weight_i)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.metric("Weighted score", _fmt_calc(row.get("raw_surveillance_score"), "score"))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, key=table_key)


def _score_audit_period_options(ticker: str, quarterly_scores_df: pd.DataFrame) -> tuple[List[str], Dict[str, pd.Series]]:
    if quarterly_scores_df.empty or "score_period" not in quarterly_scores_df.columns:
        return ["Current"], {}

    company_q = quarterly_scores_df[quarterly_scores_df["ticker"] == ticker].copy()
    if company_q.empty:
        return ["Current"], {}

    if "period_end" in company_q.columns:
        company_q["_period_sort"] = pd.to_datetime(company_q["period_end"], errors="coerce")
        company_q = company_q.sort_values(["_period_sort", "score_period"], ascending=[False, False])
    else:
        company_q = company_q.sort_values("score_period", ascending=False)

    period_rows = {
        str(qrow["score_period"]): qrow.drop(labels=["_period_sort"], errors="ignore")
        for _, qrow in company_q.iterrows()
    }
    return ["Current"] + list(period_rows.keys()), period_rows


def _render_score_calculations(row: pd.Series, quarterly_scores_df: pd.DataFrame) -> None:
    st.markdown("##### Score calculation audit")
    st.caption(
        "Scores run 0 - 100. Higher means stronger credit quality and lower risk. "
        "Sub-scores use the same direction as the final score, so each indicator is internally consistent. "
        "Sub-scores use absolute threshold curves instead of broad-universe cross-sectional percentiles. Missing values default to 50 or documented assumptions."
    )
    st.caption(
        "Peer ranks and percentiles are shown only as diagnostics. They do not drive the Surveillance Score."
    )

    period_options, period_rows = _score_audit_period_options(str(row["ticker"]), quarterly_scores_df)
    selected_period = st.selectbox(
        "Audit period",
        options=period_options,
        key=f"calc_period_{row['ticker']}",
        help="Current uses the latest dashboard row, including active scenario edits. Historical quarters use saved quarterly scores.",
    )
    audit_row = row if selected_period == "Current" else period_rows[selected_period]
    if selected_period == "Current":
        st.caption("Audit source: current latest snapshot.")
    else:
        period_end = audit_row.get("period_end", "")
        st.caption(f"Audit source: saved quarterly score for {selected_period}" + (f" ending {period_end}." if period_end else "."))

    options = ["Composite Score"] + [INDICATOR_LABELS[ind] for ind in INDICATOR_FIELDS] + ["FCF Sustainability Sub-score"]
    selected_calc = st.selectbox("Select score to explain", options=options, key=f"calc_selector_{row['ticker']}")

    row = audit_row
    period_key = str(selected_period).replace(" ", "_").replace("/", "_")
    calc_key = str(selected_calc).replace(" ", "_").replace("/", "_").replace("&", "and")

    indicator_snapshot = pd.DataFrame(
        [
            {
                "Indicator": INDICATOR_LABELS[ind],
                "Selected period score": _fmt_calc(row.get(ind), "score"),
            }
            for ind in INDICATOR_FIELDS
        ]
    )
    st.dataframe(
        indicator_snapshot,
        use_container_width=True,
        hide_index=True,
        key=f"calc_indicator_snapshot_{row['ticker']}_{period_key}",
    )

    def render_formula_card(
        title: str,
        score: float,
        formula: str,
        source: str,
        rows: List[Dict[str, str]],
        note: str = "",
    ) -> None:
        _render_formula_card(
            f"{title} - {selected_period}",
            score,
            formula,
            source,
            rows,
            note,
            table_key=f"calc_detail_table_{row['ticker']}_{period_key}_{calc_key}",
        )

    if selected_calc == "Composite Score":
        _render_composite_calc(
            row,
            period_label=selected_period,
            table_key=f"calc_composite_table_{row['ticker']}_{period_key}",
        )
        st.info(
            "The displayed Surveillance Score is no longer portfolio-rank calibrated. It is the auditable weighted average of the 9 indicator scores."
        )
        return

    if selected_calc == "Financial Performance":
        render_formula_card(
            "Financial Performance",
            row["financial_performance_score"],
            "mean(score_leverage, score_debt, score_net_debt_ebitda, score_interest_coverage, score_cash, score_margin, score_fcf, score_fcf_sustainability, score_current_ratio)",
            "Origin: SEC latest snapshot plus quarterly history for FCF sustainability; sector assumptions fill missing ratios where flagged.",
            [
                _calc_row("Liabilities / Assets", "SEC balance sheet", row.get("leverage_proxy"), "ratio", row.get("score_leverage"), "Lower leverage receives a higher sub-score"),
                _calc_row("Debt / Assets", "SEC debt and assets", row.get("debt_to_assets"), "ratio", row.get("score_debt"), "Lower debt burden receives a higher sub-score"),
                _calc_row("Net Debt / EBITDA", "SEC total debt, cash, NI, interest, depreciation", row.get("net_debt_to_ebitda"), "ratio", row.get("score_net_debt_ebitda"), "Lower multiple receives a higher sub-score"),
                _calc_row("EBITDA / Interest", "SEC NI, interest, depreciation", row.get("ebitda_interest_coverage"), "ratio", row.get("score_interest_coverage"), "Higher coverage receives a higher sub-score"),
                _calc_row("Cash / Assets", "SEC cash and assets", row.get("cash_to_assets"), "ratio", row.get("score_cash"), "Higher cash cushion receives a higher sub-score"),
                _calc_row("Net Margin", "SEC net income and revenue", row.get("net_margin"), "pct", row.get("score_margin"), "Higher margin receives a higher sub-score"),
                _calc_row("FCF / Assets", "SEC CFO, capex, assets", row.get("fcf_to_assets"), "pct", row.get("score_fcf"), "Higher FCF generation receives a higher sub-score"),
                _calc_row("FCF Sustainability", "Quarterly history derived from SEC CFO/capex/cash", row.get("score_fcf_sustainability"), "score", row.get("score_fcf_sustainability"), "Blend of negative FCF count, FCF/EBITDA conversion, and burn rate"),
                _calc_row("Current Ratio", "SEC current assets/liabilities", row.get("current_ratio"), "ratio", row.get("score_current_ratio"), "Higher liquidity receives a higher sub-score"),
            ],
            note="Financial Performance is the clearest public-data factor. A high value means the company ranks well on repayment capacity, liquidity, profitability, and cash-flow quality."
        )
        return

    if selected_calc == "FCF Sustainability Sub-score":
        render_formula_card(
            "FCF Sustainability Sub-score",
            row.get("score_fcf_sustainability"),
            "mean(score_negative_fcf_quarters, score_fcf_conversion, score_cash_burn)",
            "Origin: quarterly SEC history transformed into rolling cash-flow signals.",
            [
                _calc_row("Negative FCF quarters, L4Q", "Quarterly CFO - capex", row.get("negative_fcf_quarters_l4"), "int", row.get("score_negative_fcf_quarters"), "Fewer negative quarters receive a higher sub-score"),
                _calc_row("FCF / EBITDA conversion", "TTM FCF and EBITDA proxy", row.get("fcf_ebitda_conversion"), "ratio", row.get("score_fcf_conversion"), "Higher conversion receives a higher sub-score"),
                _calc_row("Cash burn / Cash", "Negative TTM FCF divided by cash", row.get("cash_burn_to_cash"), "ratio", row.get("score_cash_burn"), "Lower burn receives a higher sub-score"),
            ],
            note="This sub-score catches firms that look profitable but are not converting earnings into durable cash flow."
        )
        return

    if selected_calc == "Market-Implied Risk":
        render_formula_card(
            "Market-Implied Risk",
            row["market_implied_risk_score"],
            "mean(score_3m_return, score_volatility, score_drawdown)",
            "Origin: yfinance daily close series from the data prep run.",
            [
                _calc_row("3-month return", "yfinance close prices", row.get("ret_3m"), "pct", row.get("score_negative_momentum"), "Stronger return receives a higher sub-score"),
                _calc_row("3-month annualized volatility", "yfinance close prices", row.get("vol_3m"), "pct", row.get("score_volatility"), "Lower volatility receives a higher sub-score"),
                _calc_row("6-month drawdown", "yfinance close prices", row.get("drawdown_6m"), "pct", row.get("score_drawdown"), "Smaller drawdown receives a higher sub-score"),
            ],
            note="This is not a bond/CDS model yet. It uses public equity price behavior as the current market signal."
        )
        return

    if selected_calc == "Accounting Integrity":
        render_formula_card(
            "Accounting Integrity",
            row["accounting_integrity_score"],
            "If Beneish is available: average Sloan accruals score and Beneish diagnostic score. If Beneish is unavailable: 50% Financial score + 30% Data Quality score + 20% Sloan accruals score.",
            "Origin: public SEC forensic diagnostics. Beneish is estimated from quarterly history when the needed fields are available.",
            [
                _calc_row("Sloan accruals", "SEC (Net Income - CFO) / Assets", row.get("accrual_proxy"), "ratio", row.get("score_sloan_accruals"), "Lower accrual pressure receives a higher diagnostic sub-score"),
                _calc_row("Beneish M-Score", "SEC history, 8-variable Beneish model", row.get("beneish_m"), "ratio", row.get("score_beneish"), "Lower manipulation-risk M-Score receives a higher diagnostic sub-score"),
                _calc_row("Financial Performance score", "Fallback assumption input", row.get("financial_performance_score"), "score", None, "50% of fallback score when Beneish is unavailable"),
                _calc_row("Data Quality score", "Fallback assumption input", row.get("data_quality_score"), "score", None, "30% of fallback score when Beneish is unavailable"),
                _calc_row("Beneish unavailable flag", "Coverage flag", row.get("accounting_integrity_assumption_used"), "bool", None, "Yes means the fallback assumption formula is used"),
            ],
            note="Higher Accounting Integrity score means stronger earnings-quality signal."
        )
        return

    if selected_calc == "News & Sentiment":
        render_formula_card(
            "News & Sentiment",
            row["news_sentiment_score"],
            "If news is available: 40% general-news score + 60% risk-news score. If unavailable: 50% Market score + 30% Financial score + 20% Sector-quality score.",
            "Origin: public news pull when available; otherwise adjacent public signals are used as an assumption.",
            [
                _calc_row("News mentions, 30d", "GDELT/news feed if enabled", row.get("news_mentions_30d"), "int", None, "Volume component"),
                _calc_row("Risk-news mentions, 30d", "GDELT/news feed if enabled", row.get("risk_news_mentions_30d"), "int", None, "Risk-event component"),
                _calc_row("Market-Implied Risk score", "Fallback assumption input", row.get("market_implied_risk_score"), "score", None, "50% of assumed score when news is unavailable"),
                _calc_row("Financial Performance score", "Fallback assumption input", row.get("financial_performance_score"), "score", None, "30% of assumed score when news is unavailable"),
                _calc_row("Sector quality score", "Fallback assumption input", row.get("sector_quality_score"), "score", None, "20% of assumed score when news is unavailable"),
                _calc_row("Assumption used", "Data prep availability flag", row.get("news_sentiment_assumption_used"), "bool", None, "Yes means no reliable news pull was available"),
            ],
            note="Higher News & Sentiment score means lower observed or assumed event-pressure risk."
        )
        return

    if selected_calc == "Facility & Liquidity":
        render_formula_card(
            "Facility & Liquidity",
            row["facility_liquidity_score"],
            "Score = 55% Financial + 20% Cash/Assets sub-score + 15% Current Ratio sub-score + 10% Sector quality + 10. Risk proxy = 100 - score.",
            "Origin: public-data proxy; production version should use internal utilization, drawdown, liquidity, and covenant data.",
            [
                _calc_row("Financial Performance score", "Calculated indicator score", row.get("financial_performance_score"), "score", None, "55% of proxy score"),
                _calc_row("Cash / Assets sub-score", "SEC cash and assets", row.get("cash_to_assets"), "ratio", row.get("score_cash"), "20% of proxy score"),
                _calc_row("Current Ratio sub-score", "SEC current assets/liabilities", row.get("current_ratio"), "ratio", row.get("score_current_ratio"), "15% of proxy score"),
                _calc_row("Sector quality score", "SECTOR_BASE mapping in scoring.py", row.get("sector_quality_score"), "score", None, "10% of proxy score"),
                _calc_row("Risk proxy", "Formula inversion", row.get("facility_liquidity_risk_proxy"), "score", None, "Risk proxy = 100 - score"),
            ],
        )
        return

    if selected_calc == "Behavioral & Payment":
        render_formula_card(
            "Behavioral & Payment",
            row["behavioral_payment_score"],
            "Score = 55% Financial + 20% Accounting + 15% Data Quality + 10% Market + 8. Risk proxy = 100 - score.",
            "Origin: public-data proxy; production version should use payment timeliness, waivers, amendments, and reporting behavior.",
            [
                _calc_row("Financial Performance score", "Calculated indicator score", row.get("financial_performance_score"), "score", None, "55% of proxy score"),
                _calc_row("Accounting Integrity score", "Calculated indicator score", row.get("accounting_integrity_score"), "score", None, "20% of proxy score"),
                _calc_row("Data Quality score", "Data confidence input", row.get("data_quality_score"), "score", None, "15% of proxy score"),
                _calc_row("Market-Implied Risk score", "Calculated indicator score", row.get("market_implied_risk_score"), "score", None, "10% of proxy score"),
                _calc_row("Risk proxy", "Formula inversion", row.get("behavioral_payment_risk_proxy"), "score", None, "Risk proxy = 100 - score"),
            ],
        )
        return

    if selected_calc == "Connectivity & Contagion":
        render_formula_card(
            "Connectivity & Contagion",
            row["connectivity_contagion_score"],
            "Score = 45% Market + 25% Sector quality + 20% News + 10% Data Quality + 5. Scores are bounded between 0 and 100.",
            "Origin: public-data proxy; production version should use ownership, supply-chain, and cross-obligor exposure data.",
            [
                _calc_row("Market-Implied Risk score", "Calculated indicator score", row.get("market_implied_risk_score"), "score", None, "45% of proxy score"),
                _calc_row("Sector quality score", "SECTOR_BASE mapping in scoring.py", row.get("sector_quality_score"), "score", None, "25% of proxy score"),
                _calc_row("News & Sentiment score", "Calculated indicator score", row.get("news_sentiment_score"), "score", None, "20% of proxy score"),
                _calc_row("Data Quality score", "Data confidence input", row.get("data_quality_score"), "score", None, "10% of proxy score"),
                _calc_row("Sector peer count", "Loaded universe sector counts", row.get("sector_density_count"), "int", row.get("sector_density_score"), "Context only"),
            ],
        )
        return

    if selected_calc == "Governance & Strategic Discipline":
        render_formula_card(
            "Governance & Strategic Discipline",
            row["governance_discipline_score"],
            "Score = 45% Accounting + 25% News + 20% Financial + 10% Data Quality + 5. Risk proxy = 100 - score.",
            "Origin: public-data proxy; production version should add insider transactions, management turnover, governance events, and M&A behavior.",
            [
                _calc_row("Accounting Integrity score", "Calculated indicator score", row.get("accounting_integrity_score"), "score", None, "45% of proxy score"),
                _calc_row("News & Sentiment score", "Calculated indicator score", row.get("news_sentiment_score"), "score", None, "25% of proxy score"),
                _calc_row("Financial Performance score", "Calculated indicator score", row.get("financial_performance_score"), "score", None, "20% of proxy score"),
                _calc_row("Data Quality score", "Data confidence input", row.get("data_quality_score"), "score", None, "10% of proxy score"),
                _calc_row("Risk proxy", "Formula inversion", row.get("governance_discipline_risk_proxy"), "score", None, "Risk proxy = 100 - score"),
            ],
        )
        return

    if selected_calc == "Collateral & Recovery":
        render_formula_card(
            "Collateral & Recovery",
            row["collateral_recovery_score"],
            "Score = 40% Debt/Assets sub-score + 25% Net Debt/EBITDA sub-score + 20% Leverage sub-score + 15% Financial + 5. Risk proxy = 100 - score.",
            "Origin: public-data proxy; production version should use lien, collateral, appraisal, margining, and recovery data.",
            [
                _calc_row("Debt / Assets sub-score", "SEC debt and assets", row.get("debt_to_assets"), "ratio", row.get("score_debt"), "40% of proxy score"),
                _calc_row("Net Debt / EBITDA sub-score", "SEC debt, cash, EBITDA proxy", row.get("net_debt_to_ebitda"), "ratio", row.get("score_net_debt_ebitda"), "25% of proxy score"),
                _calc_row("Liabilities / Assets sub-score", "SEC balance sheet", row.get("leverage_proxy"), "ratio", row.get("score_leverage"), "20% of proxy score"),
                _calc_row("Financial Performance score", "Calculated indicator score", row.get("financial_performance_score"), "score", None, "15% of proxy score"),
                _calc_row("Risk proxy", "Formula inversion", row.get("collateral_recovery_risk_proxy"), "score", None, "Risk proxy = 100 - score"),
            ],
        )
        return


def _peer_group(row: pd.Series, scored_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    ticker = row.get("ticker")
    industry = row.get("industry_group")
    sector = row.get("sector_group")

    peers = scored_df[
        (scored_df["industry_group"] == industry) & (scored_df["ticker"] != ticker)
    ].copy()
    if len(peers) >= 3:
        return peers, f"Industry group: {industry}"

    peers = scored_df[
        (scored_df["sector_group"] == sector) & (scored_df["ticker"] != ticker)
    ].copy()
    return peers, f"Sector fallback: {sector}"


def _render_peer_comparison(row: pd.Series, scored_df: pd.DataFrame) -> None:
    peers, group_label = _peer_group(row, scored_df)
    if peers.empty:
        st.info("No peer group is available for this company in the loaded universe.")
        return

    st.markdown("##### Peer and industry comparison")
    st.caption(
        f"Benchmarking against {group_label} ({len(peers)} peers, excluding the selected company). "
        "Higher score means stronger credit quality."
    )

    score_peer_avg = float(peers["surveillance_score"].mean())
    industry_rank_df = pd.concat([peers, row.to_frame().T], ignore_index=True)
    rank = int(
        industry_rank_df["surveillance_score"]
        .rank(ascending=False, method="min")
        .iloc[-1]
    )
    percentile = float(
        industry_rank_df["surveillance_score"]
        .rank(pct=True, ascending=True)
        .iloc[-1]
        * 100
    )

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Company score", f"{float(row['surveillance_score']):.1f}")
    p2.metric("Peer average", f"{score_peer_avg:.1f}", f"{float(row['surveillance_score']) - score_peer_avg:+.1f}")
    p3.metric("Quality rank", f"{rank} of {len(industry_rank_df)}", help="1 = strongest score in comparison group")
    p4.metric("Peer percentile", f"{percentile:.0f}%", help="Higher percentile = stronger than more peers")

    radar_labels = [INDICATOR_LABELS[ind] for ind in INDICATOR_FIELDS]
    company_vals = [float(row[ind]) for ind in INDICATOR_FIELDS]
    peer_vals = [float(peers[ind].mean()) for ind in INDICATOR_FIELDS]
    radar_fig = go.Figure()
    radar_fig.add_trace(
        go.Scatterpolar(
            r=company_vals + [company_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=row["ticker"],
            line_color="#dc2626",
            opacity=0.78,
        )
    )
    radar_fig.add_trace(
        go.Scatterpolar(
            r=peer_vals + [peer_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name="Peer avg",
            line_color="#2563eb",
            opacity=0.45,
        )
    )
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        margin=dict(l=20, r=20, t=20, b=20),
        height=430,
        legend=dict(orientation="h", y=-0.08),
    )

    delta_rows = pd.DataFrame(
        {
            "Indicator": radar_labels,
            "Company": company_vals,
            "Peer average": peer_vals,
        }
    )
    delta_rows["Delta"] = delta_rows["Company"] - delta_rows["Peer average"]
    delta_rows = delta_rows.sort_values("Delta", ascending=False)
    delta_fig = px.bar(
        delta_rows,
        x="Delta",
        y="Indicator",
        orientation="h",
        color="Delta",
        color_continuous_scale="RdBu_r",
        text=delta_rows["Delta"].map(lambda v: f"{v:+.1f}"),
    )
    delta_fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=20, b=20),
        coloraxis_showscale=False,
        xaxis_title="Company minus peer average",
        yaxis_title="",
    )

    pc1, pc2 = st.columns([1, 1])
    with pc1:
        st.plotly_chart(radar_fig, use_container_width=True)
    with pc2:
        st.plotly_chart(delta_fig, use_container_width=True)

    st.markdown("###### Indicator benchmark")
    show_delta = delta_rows.copy()
    show_delta["Company"] = show_delta["Company"].map(lambda v: f"{v:.1f}")
    show_delta["Peer average"] = show_delta["Peer average"].map(lambda v: f"{v:.1f}")
    show_delta["Delta"] = show_delta["Delta"].map(lambda v: f"{v:+.1f}")
    st.dataframe(show_delta, use_container_width=True, hide_index=True)

    st.markdown("###### Key ratio benchmark")
    ratio_specs = [
        ("Net Debt / EBITDA", "net_debt_to_ebitda", "ratio", "Higher is weaker"),
        ("EBITDA / Interest", "ebitda_interest_coverage", "ratio", "Lower is weaker"),
        ("Debt / Assets", "debt_to_assets", "ratio", "Higher is weaker"),
        ("Cash / Assets", "cash_to_assets", "ratio", "Lower is weaker"),
        ("Net Margin", "net_margin", "pct", "Lower is weaker"),
        ("FCF / Assets", "fcf_to_assets", "pct", "Lower is weaker"),
        ("Current Ratio", "current_ratio", "ratio", "Lower is weaker"),
        ("3M Return", "ret_3m", "pct", "Lower is weaker"),
        ("3M Volatility", "vol_3m", "pct", "Higher is weaker"),
    ]
    ratio_rows = []
    for label, col, kind, interpretation in ratio_specs:
        company_value = row.get(col)
        peer_average = peers[col].mean() if col in peers.columns else np.nan
        ratio_rows.append(
            {
                "Metric": label,
                "Company": _fmt_calc(company_value, kind),
                "Peer average": _fmt_calc(peer_average, kind),
                "Delta": "-"
                if pd.isna(company_value) or pd.isna(peer_average)
                else _fmt_calc(float(company_value) - float(peer_average), kind),
                "Interpretation": interpretation,
            }
        )
    st.dataframe(pd.DataFrame(ratio_rows), use_container_width=True, hide_index=True)

    st.markdown("###### Lowest-scoring peers")
    peer_table = (
        peers.sort_values("surveillance_score", ascending=True)
        .head(8)[
            [
                "ticker",
                "company_name",
                "surveillance_score",
                "alert_tier",
                "financial_performance_score",
                "market_implied_risk_score",
                "data_quality_score",
            ]
        ]
        .rename(
            columns={
                "ticker": "Ticker",
                "company_name": "Company",
                "surveillance_score": "Score",
                "alert_tier": "Tier",
                "financial_performance_score": "Financial",
                "market_implied_risk_score": "Market",
                "data_quality_score": "DQ",
            }
        )
    )
    st.dataframe(
        peer_table.style.format({"Score": "{:.1f}", "Financial": "{:.1f}", "Market": "{:.1f}", "DQ": "{:.0f}"}),
        use_container_width=True,
        hide_index=True,
    )


def _top_driver_lines(row: pd.Series) -> List[str]:
    drivers = row.get("top_driver_objects", [])
    lines = []
    for ind, value in drivers[:3]:
        lines.append(f"- {INDICATOR_LABELS[ind]}: {float(value):.1f}")
    return lines


def _peer_context(row: pd.Series, scored_df: pd.DataFrame) -> Dict[str, object]:
    peers, group_label = _peer_group(row, scored_df)
    if peers.empty:
        return {
            "group_label": "No peer group available",
            "peer_count": 0,
            "peer_average": np.nan,
            "rank": np.nan,
            "group_size": 1,
            "percentile": np.nan,
        }

    group = pd.concat([peers[["ticker", "surveillance_score"]], row.to_frame().T], ignore_index=True)
    rank = int(group["surveillance_score"].rank(ascending=False, method="min").iloc[-1])
    percentile = float(group["surveillance_score"].rank(pct=True, ascending=True).iloc[-1] * 100)
    return {
        "group_label": group_label,
        "peer_count": len(peers),
        "peer_average": float(peers["surveillance_score"].mean()),
        "rank": rank,
        "group_size": len(group),
        "percentile": percentile,
    }


def _scenario_is_active_for_row(row: pd.Series, base_row: pd.Series) -> bool:
    return bool(abs(float(row["surveillance_score"]) - float(base_row["surveillance_score"])) > 1e-9)


def _build_executive_memo(row: pd.Series, base_row: pd.Series, scored_df: pd.DataFrame) -> str:
    peer = _peer_context(row, scored_df)
    scenario_active_for_row = _scenario_is_active_for_row(row, base_row)
    score_delta = float(row["surveillance_score"]) - float(base_row["surveillance_score"])
    tier_note = (
        f"Scenario impact: score changed {score_delta:+.1f} points from base "
        f"({float(base_row['surveillance_score']):.1f}) and tier moved from "
        f"{base_row['alert_tier']} to {row['alert_tier']}."
        if scenario_active_for_row
        else "No what-if scenario edits are active for this company."
    )

    driver_text = "\n".join(_top_driver_lines(row))
    peer_line = (
        f"Peer context: benchmarked against {peer['group_label']} "
        f"({peer['peer_count']} peers). The company ranks {peer['rank']} of "
        f"{peer['group_size']} by credit-quality score, with a score {float(row['surveillance_score']) - float(peer['peer_average']):+.1f} "
        f"points versus peer average ({float(peer['peer_average']):.1f})."
        if peer["peer_count"]
        else "Peer context: no comparable peer group is available in the loaded universe."
    )

    action = "Continue monitoring under normal cadence."
    if row["alert_tier"] == "Tier 1":
        action = "Recommended action: mandatory name-level review, document factor drivers, and evaluate PD/LGD/EAD overlays."
    elif row["alert_tier"] == "Tier 2":
        action = "Recommended action: targeted review of the drivers contributing to accelerating stress."
    elif row["alert_tier"] == "Tier 3":
        action = "Recommended action: place in monitoring queue and seek confirmatory evidence before escalation."

    memo = f"""Executive Credit Surveillance Memo

Company: {row['ticker']} - {row['company_name']}
Sector / Industry: {row['sector_group']} / {row['industry_group']}

Current Assessment
Surveillance Score: {float(row['surveillance_score']):.1f}
Alert Level: {row['alert_level']}
Action Tier: {row['alert_tier']}
Data Quality Score: {float(row['data_quality_score']):.0f}
Velocity Signals: {int(row.get('velocity_signal_count', 0))}/3

Summary
{row['company_name']} is currently classified as {row['alert_level']} with action tier {row['alert_tier']}. {peer_line} {tier_note}

Weakest Score Drivers
{driver_text}

Key Financial Snapshot
- Financial Performance score: {float(row['financial_performance_score']):.1f}
- Market-Implied Risk score: {float(row['market_implied_risk_score']):.1f}
- Accounting Integrity score: {float(row['accounting_integrity_score']):.1f}
- Net Debt / EBITDA: {_fmt_calc(row.get('net_debt_to_ebitda'), 'ratio')}
- EBITDA / Interest: {_fmt_calc(row.get('ebitda_interest_coverage'), 'ratio')}
- FCF / Assets: {_fmt_calc(row.get('fcf_to_assets'), 'pct')}
- 3-month return: {_fmt_calc(row.get('ret_3m'), 'pct')}

Management Action
{action}

Data Caveat
This public-data proof of concept uses SEC filings, public market data, and documented proxy factors where internal bank data is not available. Facility, behavioral, governance, contagion, and collateral factors should be replaced with internal production data before use in formal credit actions.
"""
    return memo


def _scenario_export_table(row: pd.Series, base_row: pd.Series) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []

    summary_fields = [
        ("Surveillance Score", "surveillance_score", "score"),
        ("Alert Level", "alert_level", "text"),
        ("Action Tier", "alert_tier", "text"),
        ("Financial Performance", "financial_performance_score", "score"),
        ("Market-Implied Risk", "market_implied_risk_score", "score"),
        ("Accounting Integrity", "accounting_integrity_score", "score"),
        ("Data Quality", "data_quality_score", "score"),
    ]
    metric_fields = [
        ("Revenue", "revenue", "money"),
        ("Net Income", "net_income", "money"),
        ("Assets", "assets", "money"),
        ("Liabilities", "liabilities", "money"),
        ("Cash", "cash", "money"),
        ("Total Debt", "total_debt", "money"),
        ("Net Debt / EBITDA", "net_debt_to_ebitda", "ratio"),
        ("EBITDA / Interest", "ebitda_interest_coverage", "ratio"),
        ("Debt / Assets", "debt_to_assets", "ratio"),
        ("Cash / Assets", "cash_to_assets", "ratio"),
        ("Net Margin", "net_margin", "pct"),
        ("FCF / Assets", "fcf_to_assets", "pct"),
        ("Current Ratio", "current_ratio", "ratio"),
        ("3M Return", "ret_3m", "pct"),
        ("3M Volatility", "vol_3m", "pct"),
        ("6M Drawdown", "drawdown_6m", "pct"),
    ]
    indicator_fields = [(INDICATOR_LABELS[ind], ind, "score") for ind in INDICATOR_FIELDS]

    for section, fields in [
        ("Summary", summary_fields),
        ("Key Metrics", metric_fields),
        ("Indicator Scores", indicator_fields),
    ]:
        for label, col, kind in fields:
            current = row.get(col)
            base = base_row.get(col)
            if kind == "text":
                delta = "" if current == base else f"{base} -> {current}"
                current_fmt = str(current)
                base_fmt = str(base)
            else:
                delta = "" if pd.isna(current) or pd.isna(base) else _fmt_calc(float(current) - float(base), kind)
                current_fmt = _fmt_calc(current, kind)
                base_fmt = _fmt_calc(base, kind)
            rows.append(
                {
                    "Ticker": row["ticker"],
                    "Company": row["company_name"],
                    "Section": section,
                    "Field": label,
                    "Base Value": base_fmt,
                    "Scenario / Current Value": current_fmt,
                    "Change": delta,
                }
            )
    return pd.DataFrame(rows)


def _pdf_from_text(title: str, body: str) -> bytes:
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        lines: List[str] = []
        for paragraph in body.splitlines():
            if not paragraph.strip():
                lines.append("")
            else:
                lines.extend(textwrap.wrap(paragraph, width=92) or [""])

        page_lines = 42
        for start in range(0, len(lines), page_lines):
            fig = plt.figure(figsize=(8.5, 11))
            fig.patch.set_facecolor("white")
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            if start == 0:
                ax.text(0.08, 0.95, title, fontsize=16, fontweight="bold", va="top")
                y = 0.90
            else:
                ax.text(0.08, 0.95, f"{title} (continued)", fontsize=14, fontweight="bold", va="top")
                y = 0.90
            for line in lines[start:start + page_lines]:
                ax.text(0.08, y, line, fontsize=9.5, va="top", family="monospace")
                y -= 0.0205
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return buffer.getvalue()


def _render_memo_and_exports(row: pd.Series, base_row: pd.Series, scored_df: pd.DataFrame) -> None:
    st.markdown("##### Executive memo and scenario export")
    memo = _build_executive_memo(row, base_row, scored_df)
    scenario_table = _scenario_export_table(row, base_row)
    safe_name = str(row["ticker"]).replace("/", "_")
    pdf_bytes = _pdf_from_text(f"{row['ticker']} Executive Credit Memo", memo)
    csv_bytes = scenario_table.to_csv(index=False).encode("utf-8")

    with st.expander("Generated executive memo", expanded=True):
        st.text_area(
            "Memo text",
            memo,
            height=420,
            key=f"memo_text_{row['ticker']}",
            label_visibility="collapsed",
        )

    e1, e2, e3 = st.columns([1, 1, 2])
    e1.download_button(
        "Download memo PDF",
        data=pdf_bytes,
        file_name=f"{safe_name}_executive_credit_memo.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
    e2.download_button(
        "Download scenario CSV",
        data=csv_bytes,
        file_name=f"{safe_name}_scenario_export.csv",
        mime="text/csv",
        use_container_width=True,
    )
    e3.caption(
        "Exports reflect the selected company after any active scenario edits. Source CSV remains unchanged."
    )


def _render_quarterly_company_section(ticker: str, quarterly_scores_df: pd.DataFrame) -> None:
    st.markdown("##### Quarterly score history")
    st.caption(
        "Company-level score history from SEC quarterly/TTM financial data for the most recent two years."
    )
    if quarterly_scores_df.empty:
        st.info("Quarterly scores file not found. Run `python build_surveillance_store.py` in the data prep package.")
        return

    company_q = quarterly_scores_df[quarterly_scores_df["ticker"] == ticker].copy()
    if company_q.empty:
        st.info("No quarterly score rows are available for this company.")
        return
    company_q["score_period"] = company_q["score_period"].astype(str)
    company_q = company_q.sort_values("score_period")

    q1, q2, q3 = st.columns(3)
    q1.metric("Quarterly rows", f"{len(company_q):,}")
    q2.metric("Periods", int(company_q["score_period"].nunique()))
    q3.metric("Latest period", company_q["score_period"].max())

    fig_q = px.line(
        company_q,
        x="score_period",
        y="surveillance_score",
        markers=True,
        hover_data=["period_end", "raw_surveillance_score", "financial_performance_score", "alert_level", "alert_tier"],
    )
    fig_q.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Quarter",
        yaxis_title="Surveillance Score",
        yaxis_range=[0, 100],
    )
    st.plotly_chart(fig_q, use_container_width=True)

    display_cols = [
        "score_period", "period_end", "surveillance_score", "raw_surveillance_score",
        "alert_level", "alert_tier", "financial_performance_score",
        "market_implied_risk_score", "accounting_integrity_score", "data_quality_score",
    ]
    st.dataframe(
        company_q[[c for c in display_cols if c in company_q.columns]]
        .sort_values("score_period", ascending=False)
        .rename(
            columns={
                "score_period": "Quarter",
                "period_end": "SEC Period End",
                "surveillance_score": "Score",
                "raw_surveillance_score": "Weighted Score",
                "alert_level": "Alert",
                "alert_tier": "Tier",
                "financial_performance_score": "Financial",
                "market_implied_risk_score": "Market",
                "accounting_integrity_score": "Accounting",
                "data_quality_score": "Data Quality",
            }
        )
        .style.format({
            "Score": "{:.1f}", "Weighted Score": "{:.1f}", "Financial": "{:.1f}",
            "Market": "{:.1f}", "Accounting": "{:.1f}", "Data Quality": "{:.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


def _render_fraud_summary_card(row: pd.Series) -> None:
    st.markdown("##### Fraud & Accounting")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accounting", _fmt_calc(row.get("accounting_integrity_score"), "score"))
    m2.metric("Fraud Risk", _fmt_calc(row.get("fraud_risk_subscore"), "score"))
    m3.metric("Sloan Score", _fmt_calc(row.get("score_sloan_accruals"), "score"))
    m4.metric("Watch Flag", "Yes" if bool(row.get("fraud_watch_flag", False)) else "No")

    if pd.notna(row.get("beneish_m")):
        if float(row["beneish_m"]) > -1.78:
            caption = "Beneish is above the -1.78 threshold; review accounting quality."
        else:
            caption = "Beneish is below the -1.78 elevated-risk threshold."
    else:
        caption = "Beneish is unavailable; Accounting Integrity uses the documented fallback."
    st.caption(caption)
    st.link_button(
        "Open fraud detail",
        f"?view=fraud&ticker={row['ticker']}",
        use_container_width=False,
    )


def _render_fraud_detail_page(scored_df: pd.DataFrame, ticker: str) -> None:
    if ticker not in set(scored_df["ticker"]):
        ticker = "AAPL" if "AAPL" in set(scored_df["ticker"]) else sorted(scored_df["ticker"])[0]
    fraud_row = scored_df[scored_df["ticker"] == ticker].iloc[0]

    st.link_button("Back to dashboard", "./", use_container_width=False)
    st.subheader(f"Fraud and accounting detail: {fraud_row['ticker']} - {fraud_row['company_name']}")
    st.caption(
        "Forensic accounting detail focused on Beneish M-Score, Sloan accruals, fallback assumptions, and accounting-quality review signals."
    )

    fraud_options = sorted(scored_df["ticker"].tolist())
    selected_detail = st.selectbox(
        "Fraud detail company",
        options=fraud_options,
        index=fraud_options.index(str(fraud_row["ticker"])),
        key="fraud_detail_company",
    )
    if selected_detail != fraud_row["ticker"]:
        st.link_button("Open selected company detail", f"?view=fraud&ticker={selected_detail}")

    fd1, fd2, fd3, fd4, fd5 = st.columns(5)
    fd1.metric("Accounting Integrity", f"{fraud_row['accounting_integrity_score']:.1f}")
    fd2.metric("Fraud Risk", _fmt_calc(fraud_row.get("fraud_risk_subscore"), "score"))
    fd3.metric("Sloan Score", _fmt_calc(fraud_row.get("score_sloan_accruals"), "score"))
    fd4.metric("Beneish M", _fmt_calc(fraud_row.get("beneish_m"), "ratio"))
    fd5.metric("Watch Flag", "Yes" if bool(fraud_row.get("fraud_watch_flag", False)) else "No")

    detail_rows = pd.DataFrame(
        [
            {
                "Signal": "Sloan accruals",
                "Raw value": _fmt_calc(fraud_row.get("accrual_proxy"), "ratio"),
                "Score": _fmt_calc(fraud_row.get("score_sloan_accruals"), "score"),
                "Interpretation": "Lower accrual pressure is stronger; high accrual pressure can indicate weaker earnings quality.",
            },
            {
                "Signal": "Beneish M-Score",
                "Raw value": _fmt_calc(fraud_row.get("beneish_m"), "ratio"),
                "Score": _fmt_calc(fraud_row.get("score_beneish"), "score"),
                "Interpretation": "M > -1.78 historically indicates elevated manipulation risk; unavailable values use fallback logic.",
            },
            {
                "Signal": "Fallback assumption",
                "Raw value": "Yes" if bool(fraud_row.get("accounting_integrity_assumption_used", False)) else "No",
                "Score": "-",
                "Interpretation": "When Beneish is unavailable, Accounting Integrity uses Financial Performance, Data Quality, and Sloan accruals.",
            },
            {
                "Signal": "Fraud risk sub-score",
                "Raw value": _fmt_calc(fraud_row.get("fraud_risk_subscore"), "score"),
                "Score": _fmt_calc(fraud_row.get("fraud_quality_subscore"), "score"),
                "Interpretation": "Fraud risk is 100 minus forensic quality; higher risk values should trigger accounting-quality review.",
            },
        ]
    )
    st.dataframe(detail_rows, use_container_width=True, hide_index=True)

    if pd.notna(fraud_row.get("beneish_m")):
        beneish_comps = pd.DataFrame({
            "Variable": ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "TATA", "LVGI"],
            "Value": [
                fraud_row.get("beneish_dsri"), fraud_row.get("beneish_gmi"), fraud_row.get("beneish_aqi"),
                fraud_row.get("beneish_sgi"), fraud_row.get("beneish_depi"), fraud_row.get("beneish_sgai"),
                fraud_row.get("beneish_tata"), fraud_row.get("beneish_lvgi"),
            ],
            "Meaning": [
                "Receivables growth vs sales",
                "Gross margin deterioration",
                "Soft asset growth",
                "Sales growth pressure",
                "Depreciation rate change",
                "SG&A growth vs sales",
                "Accruals vs assets",
                "Leverage increase",
            ],
        })
        beneish_comps["Value"] = beneish_comps["Value"].apply(lambda v: "-" if pd.isna(v) else f"{float(v):.3f}")
        st.markdown("##### Beneish component detail")
        st.dataframe(beneish_comps, use_container_width=True, hide_index=True)
    else:
        st.info("Beneish M-Score is unavailable for this company; fallback logic is shown above.")

    st.markdown("##### Portfolio fraud watchlist")
    watch_cols = [
        "ticker", "company_name", "sector_group", "surveillance_score", "alert_tier",
        "accounting_integrity_score", "fraud_risk_subscore", "beneish_m",
        "score_sloan_accruals", "score_beneish", "accounting_integrity_assumption_used",
    ]
    watch_table = (
        scored_df[[c for c in watch_cols if c in scored_df.columns]]
        .sort_values(["fraud_risk_subscore", "accounting_integrity_score"], ascending=[False, True])
        .head(25)
        .rename(
            columns={
                "ticker": "Ticker",
                "company_name": "Company",
                "sector_group": "Sector",
                "surveillance_score": "Score",
                "alert_tier": "Tier",
                "accounting_integrity_score": "Accounting",
                "fraud_risk_subscore": "Fraud Risk",
                "beneish_m": "Beneish M",
                "score_sloan_accruals": "Sloan Score",
                "score_beneish": "Beneish Score",
                "accounting_integrity_assumption_used": "Fallback Used",
            }
        )
    )
    st.dataframe(
        watch_table.style.format({
            "Score": "{:.1f}", "Accounting": "{:.1f}", "Fraud Risk": "{:.1f}",
            "Beneish M": "{:.2f}", "Sloan Score": "{:.1f}", "Beneish Score": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


STRESS_TEMPLATES: Dict[str, Dict[str, float]] = {
    "Credit downturn": {
        "revenue_decline": 0.10,
        "margin_compression": 0.03,
        "cash_decline": 0.10,
        "debt_increase": 0.10,
        "ret_3m_shock": 0.15,
        "vol_increase": 0.25,
        "drawdown_shock": 0.15,
    },
    "Liquidity stress": {
        "revenue_decline": 0.05,
        "margin_compression": 0.02,
        "cash_decline": 0.30,
        "debt_increase": 0.15,
        "current_liabilities_increase": 0.20,
        "ret_3m_shock": 0.10,
        "vol_increase": 0.20,
        "drawdown_shock": 0.10,
    },
    "Market shock": {
        "revenue_decline": 0.00,
        "margin_compression": 0.00,
        "cash_decline": 0.00,
        "debt_increase": 0.00,
        "ret_3m_shock": 0.30,
        "vol_increase": 0.75,
        "drawdown_shock": 0.25,
    },
    "Sector contagion stress": {
        "revenue_decline": 0.05,
        "margin_compression": 0.01,
        "cash_decline": 0.05,
        "debt_increase": 0.05,
        "ret_3m_shock": 0.12,
        "vol_increase": 0.25,
        "drawdown_shock": 0.12,
        "connectivity_score_shock": 15.0,
        "news_score_shock": 10.0,
    },
    "Accounting / fraud stress": {
        "revenue_decline": 0.00,
        "margin_compression": 0.00,
        "cash_decline": 0.00,
        "debt_increase": 0.00,
        "ret_3m_shock": 0.08,
        "vol_increase": 0.15,
        "drawdown_shock": 0.08,
        "accounting_score_shock": 25.0,
        "governance_score_shock": 15.0,
        "news_score_shock": 10.0,
    },
}


def _apply_stress_template(
    raw_snapshot: pd.DataFrame,
    scored_snapshot: pd.DataFrame,
    ticker: str,
    shocks: Dict[str, float],
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    stressed_raw = raw_snapshot.copy()
    mask = stressed_raw["ticker"] == ticker
    if not mask.any():
        return scored_snapshot.copy()

    base = scored_snapshot[scored_snapshot["ticker"] == ticker].iloc[0]
    revenue = pd.to_numeric(stressed_raw.loc[mask, "revenue"], errors="coerce")
    net_income = pd.to_numeric(stressed_raw.loc[mask, "net_income"], errors="coerce")
    cfo = pd.to_numeric(stressed_raw.loc[mask, "cfo"], errors="coerce")
    cash = pd.to_numeric(stressed_raw.loc[mask, "cash"], errors="coerce")
    total_debt = pd.to_numeric(stressed_raw.loc[mask, "total_debt"], errors="coerce")
    current_liabilities = pd.to_numeric(stressed_raw.loc[mask, "current_liabilities"], errors="coerce")

    revenue_decline = shocks.get("revenue_decline", 0.0)
    margin_compression = shocks.get("margin_compression", 0.0)
    cash_decline = shocks.get("cash_decline", 0.0)
    debt_increase = shocks.get("debt_increase", 0.0)
    current_liabilities_increase = shocks.get("current_liabilities_increase", 0.0)

    stressed_revenue = revenue * (1 - revenue_decline)
    base_margin = float(base.get("net_margin")) if pd.notna(base.get("net_margin")) else np.nan
    stressed_margin = np.nan if pd.isna(base_margin) else max(base_margin - margin_compression, -0.75)

    stressed_raw.loc[mask, "revenue"] = stressed_revenue
    if pd.notna(stressed_margin):
        stressed_raw.loc[mask, "net_income"] = stressed_revenue * stressed_margin
    stressed_raw.loc[mask, "cfo"] = cfo - (revenue * margin_compression * 0.75).fillna(0)
    stressed_raw.loc[mask, "cash"] = cash * (1 - cash_decline)
    stressed_raw.loc[mask, "total_debt"] = total_debt * (1 + debt_increase)
    stressed_raw.loc[mask, "current_liabilities"] = current_liabilities * (1 + current_liabilities_increase)

    for col in ["ret_1m", "ret_3m", "vol_3m", "drawdown_6m"]:
        if col not in stressed_raw.columns:
            stressed_raw[col] = np.nan
        stressed_raw[col] = pd.to_numeric(stressed_raw[col], errors="coerce")
    stressed_raw.loc[mask, "ret_3m"] = stressed_raw.loc[mask, "ret_3m"] - shocks.get("ret_3m_shock", 0.0)
    stressed_raw.loc[mask, "ret_1m"] = stressed_raw.loc[mask, "ret_1m"] - shocks.get("ret_3m_shock", 0.0) / 3
    stressed_raw.loc[mask, "vol_3m"] = stressed_raw.loc[mask, "vol_3m"] * (1 + shocks.get("vol_increase", 0.0))
    stressed_raw.loc[mask, "drawdown_6m"] = stressed_raw.loc[mask, "drawdown_6m"] - shocks.get("drawdown_shock", 0.0)

    stressed_raw = _recompute_latest_snapshot_ratios(stressed_raw)
    stressed_scored = compute_scores(stressed_raw, history_df=history_df)

    indicator_shocks: Dict[str, object] = {}
    score_shock_map = {
        "connectivity_score_shock": "connectivity_contagion_score",
        "news_score_shock": "news_sentiment_score",
        "accounting_score_shock": "accounting_integrity_score",
        "governance_score_shock": "governance_discipline_score",
    }
    stressed_row = stressed_scored[stressed_scored["ticker"] == ticker].iloc[0]
    for shock_key, indicator in score_shock_map.items():
        if shocks.get(shock_key, 0.0) > 0:
            indicator_shocks[indicator] = max(0.0, float(stressed_row[indicator]) - shocks[shock_key])
    if indicator_shocks:
        stressed_scored = apply_indicator_score_overrides(stressed_scored, {ticker: indicator_shocks})

    return stressed_scored


def _overlay_rationale(base_row: pd.Series, stressed_row: pd.Series) -> pd.DataFrame:
    score_delta = float(stressed_row["surveillance_score"]) - float(base_row["surveillance_score"])
    financial_delta = float(stressed_row["financial_performance_score"]) - float(base_row["financial_performance_score"])
    market_delta = float(stressed_row["market_implied_risk_score"]) - float(base_row["market_implied_risk_score"])
    accounting_delta = float(stressed_row["accounting_integrity_score"]) - float(base_row["accounting_integrity_score"])
    facility_delta = float(stressed_row["facility_liquidity_score"]) - float(base_row["facility_liquidity_score"])
    collateral_delta = float(stressed_row["collateral_recovery_score"]) - float(base_row["collateral_recovery_score"])

    severity = "Low"
    if score_delta <= -15 or stressed_row["alert_tier"] == "Tier 1":
        severity = "High"
    elif score_delta <= -8 or stressed_row["alert_tier"] in {"Tier 2", "Tier 3"}:
        severity = "Moderate"

    return pd.DataFrame(
        [
            {
                "Overlay": "PD",
                "Severity": severity,
                "Rationale": (
                    f"Score moves {score_delta:+.1f}; Financial {financial_delta:+.1f}, "
                    f"Market {market_delta:+.1f}, Accounting {accounting_delta:+.1f}. "
                    "Use as forward PD review trigger, not a direct regulatory PD calculation."
                ),
            },
            {
                "Overlay": "LGD",
                "Severity": "Moderate" if collateral_delta <= -8 else "Low",
                "Rationale": (
                    f"Collateral & Recovery moves {collateral_delta:+.1f}. "
                    "Review recovery assumptions, advance rates, collateral margining, and lien position where internal data is available."
                ),
            },
            {
                "Overlay": "EAD",
                "Severity": "Moderate" if facility_delta <= -8 else "Low",
                "Rationale": (
                    f"Facility & Liquidity moves {facility_delta:+.1f}. "
                    "Review stressed drawdown/utilization assumptions and availability under liquidity pressure."
                ),
            },
        ]
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Surveillance Tool")
st.caption(
    "S&P 500 + Nasdaq 100 + Dow universe — surveillance scores derived from public SEC + market data, "
    "with transparent assumptions where public data does not cover internal-bank signals."
)

if st.query_params.get("view", "") == "fraud":
    _render_fraud_detail_page(df, st.query_params.get("ticker", "AAPL"))
    st.stop()

tab_overview, tab_drilldown, tab_stress, tab_methodology, tab_quality = st.tabs(
    [
        "Page 1 - Executive Overview",
        "Page 2 - Company Drilldown",
        "Page 3 - Stress Testing",
        "Page 4 - Indicator Methodology",
        "Page 5 - Data Coverage",
    ]
)


# ===========================================================================
# Page 1 - Executive Portfolio Overview (doc Section 14, Page 1)
# ===========================================================================

with tab_overview:
    st.subheader("Portfolio overview")

    if fdf.empty:
        st.warning("No companies match the current filters.")
        st.stop()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Companies", len(fdf))
    k2.metric("Avg Surveillance Score", f"{fdf['surveillance_score'].mean():.1f}")
    k3.metric("Tier 1", int((fdf["alert_tier"] == "Tier 1").sum()),
              help="High-conviction deterioration: composite < 40 OR >=2 indicators <= 25")
    k4.metric("Tier 2", int((fdf["alert_tier"] == "Tier 2").sum()),
              help="Accelerating stress: composite 40-55 AND velocity trigger")
    k5.metric("Tier 3", int((fdf["alert_tier"] == "Tier 3").sum()),
              help="Monitoring queue: velocity trigger only, no level breach")
    k6.metric("Avg Data Quality", f"{fdf['data_quality_score'].mean():.1f}")

    st.markdown("---")

    col_tier, col2 = st.columns([1, 2])

    with col_tier:
        st.markdown("##### Action tier distribution")
        tier_counts = (
            fdf["alert_tier"]
            .value_counts()
            .reindex(["Tier 1", "Tier 2", "Tier 3", "None"], fill_value=0)
            .reset_index()
        )
        tier_counts.columns = ["alert_tier", "count"]
        fig_tier = px.bar(
            tier_counts,
            x="alert_tier",
            y="count",
            color="alert_tier",
            color_discrete_map=TIER_COLORS,
            text="count",
        )
        fig_tier.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            xaxis_title="",
            yaxis_title="Companies",
        )
        st.plotly_chart(fig_tier, use_container_width=True)

    with col2:
        st.markdown("##### Sector vs. action tier (count)")
        tier_focus = ["Tier 1", "Tier 2", "Tier 3"]
        heat = (
            fdf[fdf["alert_tier"].isin(tier_focus)]
            .groupby(["sector_group", "alert_tier"])
            .size()
            .reset_index(name="count")
        )
        heat_pivot = heat.pivot(
            index="sector_group", columns="alert_tier", values="count"
        ).reindex(columns=tier_focus, fill_value=0).fillna(0)
        if heat_pivot.empty:
            heat_pivot = pd.DataFrame(0, index=["No active tiers"], columns=tier_focus)
        fig_heat = px.imshow(
            heat_pivot.values,
            x=heat_pivot.columns,
            y=heat_pivot.index,
            color_continuous_scale="Reds",
            text_auto=True,
            aspect="auto",
        )
        fig_heat.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            xaxis_title="",
            yaxis_title="",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Top 10 lowest-scoring names")
    top10 = (
        fdf.sort_values("surveillance_score", ascending=True)
        .head(10)
        [
            [
                "ticker", "company_name", "sector_group",
                "surveillance_score", "alert_level", "alert_tier",
                "velocity_signal_count", "top_risk_drivers",
                "data_quality_score", "data_quality_note",
            ]
        ]
        .rename(
            columns={
                "ticker": "Ticker",
                "company_name": "Company",
                "sector_group": "Sector",
                "surveillance_score": "Surveillance Score",
                "alert_level": "Alert",
                "alert_tier": "Tier",
                "velocity_signal_count": "Velocity Signals",
                "top_risk_drivers": "Weakest Score Drivers",
                "data_quality_score": "Data Quality",
                "data_quality_note": "Data Quality Note",
            }
        )
    )
    st.dataframe(
        top10.style.format({"Surveillance Score": "{:.1f}", "Data Quality": "{:.0f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("##### Average Surveillance Score by sector")
    sector_avg = (
        fdf.groupby("sector_group")
        .agg(
            avg_score=("surveillance_score", "mean"),
            count=("ticker", "count"),
            avg_data_quality=("data_quality_score", "mean"),
        )
        .reset_index()
        .sort_values("avg_score", ascending=True)
    )
    fig_sector = px.bar(
        sector_avg,
        x="avg_score",
        y="sector_group",
        orientation="h",
        text=sector_avg["avg_score"].round(1),
        color="avg_score",
        color_continuous_scale="RdYlGn",
    )
    fig_sector.update_layout(
        coloraxis_showscale=False,
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Avg Surveillance Score",
        yaxis_title="",
    )
    st.plotly_chart(fig_sector, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Full filtered table")
    full_table = fdf[
        [c for c in [
            "ticker", "company_name", "index_memberships", "sector_group", "industry_group",
            "surveillance_score", "alert_level",
            "top_risk_drivers", "data_quality_score", "data_quality_note",
        ] if c in fdf.columns]
    ].sort_values("surveillance_score", ascending=True)
    st.dataframe(
        full_table.style.format({"surveillance_score": "{:.1f}", "data_quality_score": "{:.0f}"}),
        use_container_width=True,
        hide_index=True,
    )


# ===========================================================================
# Page 2 - Company Drilldown (doc Section 14, Page 2)
# ===========================================================================

with tab_drilldown:
    st.subheader("Company drilldown")

    company_options = sorted(df["ticker"].tolist())
    default_ticker = "AAPL" if "AAPL" in company_options else company_options[0]
    selected = st.selectbox(
        "Pick a ticker",
        options=company_options,
        index=company_options.index(default_ticker),
    )
    row = df[df["ticker"] == selected].iloc[0]
    base_row = base_df[base_df["ticker"] == selected].iloc[0]

    h1, h2, h3, h4, h5, h6 = st.columns([1.3, 1, 1, 1, 1, 1])
    h1.markdown(f"### {row['ticker']} - {row['company_name']}")
    h1.caption(
        f"{row['sector_group']} / {row['industry_group']} · "
        f"CIK: {row['cik']}"
    )
    h2.metric("Surveillance Score", f"{row['surveillance_score']:.1f}")
    h3.markdown(
        f"**Alert**<br>{alert_chip(row['alert_level'])}", unsafe_allow_html=True
    )
    h4.markdown(
        f"**Action Tier**<br>{tier_chip(row['alert_tier'])}", unsafe_allow_html=True
    )
    h4.caption(f"velocity signals: {int(row.get('velocity_signal_count', 0))}/3")
    h5.metric("Data Quality", f"{row['data_quality_score']:.0f}")
    h6.metric("Stale Facts", int(row.get("stale_fact_count", 0)))

    _render_scenario_editor(selected, row, base_row)

    st.markdown("---")
    _render_memo_and_exports(row, base_row, df)

    st.markdown("---")

    g1, g2 = st.columns([1, 1.7])
    with g1:
        st.markdown("##### Surveillance Score gauge")
        st.plotly_chart(
            gauge_chart(row["surveillance_score"], row["alert_level"]),
            use_container_width=True,
        )
    with g2:
        st.markdown("##### 9-indicator scores")
        st.plotly_chart(indicator_bar(row), use_container_width=True)

    st.markdown("---")
    _render_peer_comparison(row, df)

    st.markdown("---")
    _render_score_calculations(row, quarterly_scores)

    st.markdown("---")
    st.markdown("##### Why this name is flagged")
    st.markdown(explanation_for_company(row))

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### Key financial metrics")
        fin_table = pd.DataFrame(
            {
                "Metric": [
                    "Revenue", "Net income", "Assets", "Liabilities",
                    "Cash", "CFO", "Capex", "Free cash flow", "Total debt",
                    "EBITDA proxy", "Net debt", "Net debt / EBITDA",
                    "EBITDA / Interest", "Negative FCF quarters (L4Q)",
                    "FCF / EBITDA", "Cash burn / Cash",
                    "Leverage proxy", "Debt / Assets", "Cash / Assets",
                    "Net margin", "FCF / Assets", "Accrual proxy", "Current ratio",
                ],
                "Value": [
                    row.get("revenue"), row.get("net_income"), row.get("assets"),
                    row.get("liabilities"), row.get("cash"), row.get("cfo"),
                    row.get("capex"), row.get("fcf"), row.get("total_debt"),
                    row.get("ebitda_proxy"), row.get("net_debt"),
                    row.get("net_debt_to_ebitda"), row.get("ebitda_interest_coverage"),
                    row.get("negative_fcf_quarters_l4"), row.get("fcf_ebitda_conversion"),
                    row.get("cash_burn_to_cash"),
                    row.get("leverage_proxy"), row.get("debt_to_assets"),
                    row.get("cash_to_assets"), row.get("net_margin"),
                    row.get("fcf_to_assets"), row.get("accrual_proxy"), row.get("current_ratio"),
                ],
            }
        )

        def _fmt_fin(label, v):
            if pd.isna(v):
                return "—"
            v = float(v)
            if "quarters" in label.lower():
                return f"{int(v)}"
            if any(token in label.lower() for token in ["/", "ratio", "margin", "accrual", "coverage", "burn"]):
                return f"{v:.3f}"
            if abs(v) >= 1e9:
                return f"${v/1e9:,.2f}B"
            if abs(v) >= 1e6:
                return f"${v/1e6:,.2f}M"
            if abs(v) < 5:
                return f"{v:.3f}"
            return f"{v:,.2f}"

        fin_table["Value"] = [_fmt_fin(l, v) for l, v in zip(fin_table["Metric"], fin_table["Value"])]
        st.dataframe(fin_table, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("##### Market metrics")
        mkt_table = pd.DataFrame(
            {
                "Metric": [
                    "Last close", "1-month return", "3-month return",
                    "3-month annualized vol", "6-month drawdown",
                ],
                "Value": [
                    row.get("last_close"), row.get("ret_1m"), row.get("ret_3m"),
                    row.get("vol_3m"), row.get("drawdown_6m"),
                ],
            }
        )

        def _fmt_m(label, v):
            if pd.isna(v):
                return "—"
            if "return" in label.lower() or "drawdown" in label.lower() or "vol" in label.lower():
                return f"{float(v) * 100:+.2f}%"
            return f"${float(v):,.2f}"

        mkt_table["Value"] = [_fmt_m(l, v) for l, v in zip(mkt_table["Metric"], mkt_table["Value"])]
        st.dataframe(mkt_table, use_container_width=True, hide_index=True)

        st.markdown("##### Public-data flags")
        flag_rows = []
        for ind_field, label in INDICATOR_LABELS.items():
            score = float(row[ind_field])
            if ind_field in {"financial_performance_score", "market_implied_risk_score", "accounting_integrity_score"}:
                src = "Public data"
            elif ind_field == "news_sentiment_score":
                src = "Assumed from public signals" if bool(row.get("news_sentiment_assumption_used", False)) else "Public news data"
            else:
                src = "Public-data proxy"
            flag_rows.append({"Indicator": label, "Score": f"{score:.0f}", "Source": src})
        st.dataframe(pd.DataFrame(flag_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    _render_quarterly_company_section(selected, quarterly_scores)

    st.markdown("---")
    _render_fraud_summary_card(row)


# ===========================================================================
# Page 3 - Stress Testing
# ===========================================================================

with tab_stress:
    st.subheader("Scenario and stress testing")
    st.caption(
        "Session-only stress templates. These do not mutate source files; they recompute the selected company against the loaded universe and summarize PD/LGD/EAD review rationale."
    )

    stress_options = sorted(df["ticker"].tolist())
    default_stress_ticker = "AAPL" if "AAPL" in stress_options else stress_options[0]
    stress_ticker = st.selectbox(
        "Stress-test company",
        options=stress_options,
        index=stress_options.index(default_stress_ticker),
        key="stress_company",
    )
    template = st.selectbox(
        "Scenario template",
        options=list(STRESS_TEMPLATES.keys()),
        key="stress_template",
    )
    template_shocks = dict(STRESS_TEMPLATES[template])

    st.markdown("##### Scenario controls")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        template_shocks["revenue_decline"] = st.slider(
            "Revenue decline", 0.0, 0.50, float(template_shocks.get("revenue_decline", 0.0)), 0.01, format="%.2f"
        )
        template_shocks["margin_compression"] = st.slider(
            "Margin compression", 0.0, 0.25, float(template_shocks.get("margin_compression", 0.0)), 0.01, format="%.2f"
        )
    with s2:
        template_shocks["cash_decline"] = st.slider(
            "Cash decline", 0.0, 0.75, float(template_shocks.get("cash_decline", 0.0)), 0.01, format="%.2f"
        )
        template_shocks["debt_increase"] = st.slider(
            "Debt increase", 0.0, 0.75, float(template_shocks.get("debt_increase", 0.0)), 0.01, format="%.2f"
        )
    with s3:
        template_shocks["ret_3m_shock"] = st.slider(
            "3M return shock", 0.0, 0.75, float(template_shocks.get("ret_3m_shock", 0.0)), 0.01, format="%.2f"
        )
        template_shocks["vol_increase"] = st.slider(
            "Volatility increase", 0.0, 2.00, float(template_shocks.get("vol_increase", 0.0)), 0.05, format="%.2f"
        )
    with s4:
        template_shocks["drawdown_shock"] = st.slider(
            "Drawdown shock", 0.0, 0.75, float(template_shocks.get("drawdown_shock", 0.0)), 0.01, format="%.2f"
        )
        template_shocks["current_liabilities_increase"] = st.slider(
            "Current liabilities increase", 0.0, 0.75, float(template_shocks.get("current_liabilities_increase", 0.0)), 0.01, format="%.2f"
        )

    st.markdown("##### Direct factor stress overlays")
    o1, o2, o3, o4 = st.columns(4)
    with o1:
        template_shocks["accounting_score_shock"] = st.slider(
            "Accounting score shock", 0.0, 60.0, float(template_shocks.get("accounting_score_shock", 0.0)), 1.0
        )
    with o2:
        template_shocks["governance_score_shock"] = st.slider(
            "Governance score shock", 0.0, 60.0, float(template_shocks.get("governance_score_shock", 0.0)), 1.0
        )
    with o3:
        template_shocks["news_score_shock"] = st.slider(
            "News score shock", 0.0, 60.0, float(template_shocks.get("news_score_shock", 0.0)), 1.0
        )
    with o4:
        template_shocks["connectivity_score_shock"] = st.slider(
            "Connectivity score shock", 0.0, 60.0, float(template_shocks.get("connectivity_score_shock", 0.0)), 1.0
        )

    stressed_df = _apply_stress_template(raw_df, base_df, stress_ticker, template_shocks, hist)
    stress_base = base_df[base_df["ticker"] == stress_ticker].iloc[0]
    stress_row = stressed_df[stressed_df["ticker"] == stress_ticker].iloc[0]

    st.markdown("---")
    st.markdown("##### Scenario output")
    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
    sm1.metric("Base score", f"{stress_base['surveillance_score']:.1f}")
    sm2.metric("Scenario score", f"{stress_row['surveillance_score']:.1f}", f"{stress_row['surveillance_score'] - stress_base['surveillance_score']:+.1f}")
    sm3.metric("Base tier", stress_base["alert_tier"])
    sm4.metric("Scenario tier", stress_row["alert_tier"])
    sm5.metric("Velocity signals", f"{int(stress_row.get('velocity_signal_count', 0))}/3")

    stress_compare = pd.DataFrame(
        [
            {
                "Indicator": INDICATOR_LABELS[ind],
                "Base": float(stress_base[ind]),
                "Scenario": float(stress_row[ind]),
                "Change": float(stress_row[ind]) - float(stress_base[ind]),
            }
            for ind in INDICATOR_FIELDS
        ]
    ).sort_values("Change")
    fig_stress = px.bar(
        stress_compare,
        x="Change",
        y="Indicator",
        orientation="h",
        color="Change",
        color_continuous_scale="RdBu_r",
        text=stress_compare["Change"].map(lambda v: f"{v:+.1f}"),
    )
    fig_stress.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=20, b=20),
        coloraxis_showscale=False,
        xaxis_title="Scenario minus base score",
        yaxis_title="",
    )
    st.plotly_chart(fig_stress, use_container_width=True)

    cstress1, cstress2 = st.columns([1, 1])
    with cstress1:
        st.markdown("##### Indicator movement")
        st.dataframe(
            stress_compare.style.format({"Base": "{:.1f}", "Scenario": "{:.1f}", "Change": "{:+.1f}"}),
            use_container_width=True,
            hide_index=True,
        )
    with cstress2:
        st.markdown("##### PD / LGD / EAD overlay rationale")
        overlay_df = _overlay_rationale(stress_base, stress_row)
        st.dataframe(overlay_df, use_container_width=True, hide_index=True)

    export_df = pd.concat(
        [
            pd.DataFrame(
                [
                    {"Section": "Summary", "Field": "Template", "Base": "", "Scenario": template, "Change": ""},
                    {"Section": "Summary", "Field": "Surveillance Score", "Base": stress_base["surveillance_score"], "Scenario": stress_row["surveillance_score"], "Change": stress_row["surveillance_score"] - stress_base["surveillance_score"]},
                    {"Section": "Summary", "Field": "Action Tier", "Base": stress_base["alert_tier"], "Scenario": stress_row["alert_tier"], "Change": ""},
                ]
            ),
            stress_compare.rename(columns={"Indicator": "Field"}).assign(Section="Indicator")[["Section", "Field", "Base", "Scenario", "Change"]],
            overlay_df.assign(Section="Overlay", Field=overlay_df["Overlay"], Base="", Scenario=overlay_df["Severity"], Change=overlay_df["Rationale"])[["Section", "Field", "Base", "Scenario", "Change"]],
        ],
        ignore_index=True,
    )
    st.download_button(
        "Download stress output CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{stress_ticker}_stress_test.csv",
        mime="text/csv",
        use_container_width=False,
    )


# ===========================================================================
# Page 4 - Indicator Methodology (doc Section 14, Page 3)
# ===========================================================================

with tab_methodology:
    st.subheader("Indicator methodology")
    st.caption(
        "Each indicator is scored 0 - 100 (higher = stronger credit quality / lower risk). The composite Surveillance "
        "Score is the weighted average of the 9 indicator scores, without final portfolio-rank calibration."
    )

    weights_df = pd.DataFrame(
        [
            {"Indicator": INDICATOR_LABELS[k], "Weight": f"{w * 100:.0f}%"}
            for k, w in INDICATOR_WEIGHTS.items()
        ]
    )

    st.markdown("##### Composite weights")
    cw1, cw2 = st.columns([1, 2])
    with cw1:
        st.dataframe(weights_df, use_container_width=True, hide_index=True)
        st.caption(f"Sum of weights: {sum(INDICATOR_WEIGHTS.values()) * 100:.0f}%")
    with cw2:
        st.markdown("##### Continuous alert color")
        thresh_df = pd.DataFrame(
            [
                {"Surveillance Score": "55 - 100", "Alert": "Green",
                 "Interpretation": "Stronger credit quality; low current deterioration signal"},
                {"Surveillance Score": "45 - 55", "Alert": "Yellow",
                 "Interpretation": "Monitor; some emerging weakness"},
                {"Surveillance Score": "35 - 45", "Alert": "Orange",
                 "Interpretation": "Elevated risk; review recommended"},
                {"Surveillance Score": "0 - 35", "Alert": "Red",
                 "Interpretation": "Weak score; management attention recommended"},
            ]
        )
        st.dataframe(thresh_df, use_container_width=True, hide_index=True)

        st.markdown("##### Action tier")
        tier_df = pd.DataFrame(
            [
                {"Tier": "Tier 1",
                 "Trigger": "score < 40  OR  >=2 indicators <= 25",
                 "Action": "High-conviction deterioration: mandatory name-level review; PD/LGD/EAD uplifts; documentation by factor."},
                {"Tier": "Tier 2",
                 "Trigger": "40 <= score < 55  AND  velocity trigger",
                 "Action": "Accelerating stress: targeted perturbation of most-relevant drivers (utilization/EAD, collateral/LGD, ACI/PD)."},
                {"Tier": "Tier 3",
                 "Trigger": "score >= 55  AND  velocity trigger",
                 "Action": "Monitoring queue: confirmatory evidence required before escalation."},
                {"Tier": "None",
                 "Trigger": "no level breach AND no velocity trigger",
                 "Action": "Within tolerance — no overlay action."},
            ]
        )
        st.dataframe(tier_df, use_container_width=True, hide_index=True)
        st.caption(
            "Velocity trigger fires when >=2 of these YoY signals are present: "
            "(1) TTM revenue contracting, (2) net margin compressing >2pp, "
            "(3) debt/assets rising >5pp. The score convention is higher = better credit quality."
        )

    st.markdown("---")

    METHODOLOGY: List[Dict[str, str]] = [
        {
            "label": "Facility & Liquidity",
            "purpose": "Detects borrower liquidity stress through facility usage, draw behavior, liquidity cushion, and covenant pressure.",
            "inputs": "Financial score, Cash/Assets sub-score, Current Ratio sub-score, Sector quality score",
            "formula": "Score = 55% Financial + 20% Cash + 15% Current Ratio + 10% Sector quality + 10. Risk proxy = 100 - score. Scores are bounded 0-100.",
            "data_type": "Public proxy / demo assumption",
            "production": "Replace with internal commitment, utilization, drawdown, deposit, and covenant data.",
        },
        {
            "label": "Financial Performance",
            "purpose": "Measures deterioration in core repayment capacity, profitability, liquidity, and balance-sheet strength.",
            "inputs": "Leverage, Debt/Assets, Net Debt/EBITDA, EBITDA/Interest, Cash/Assets, Net margin, FCF/Assets, FCF sustainability, Current ratio",
            "formula": "average of absolute threshold_score for each component; FCF sustainability = average(negative L4Q FCF count, FCF/EBITDA conversion, cash burn / cash)",
            "data_type": "Public SEC + assumptions for missing fields",
            "production": "Add internal borrower spreading, management projections, normalized EBITDA, taxes/rent adjustments, and banker adjustments.",
        },
        {
            "label": "Behavioral & Payment",
            "purpose": "Captures early stress signs from late payments, amendments, waivers, delayed reporting.",
            "inputs": "Financial score, Accounting Integrity score, Data Quality score, Market-Implied Risk score",
            "formula": "Score = 55% Financial + 20% Accounting + 15% Data Quality + 10% Market + 8. Risk proxy = 100 - score. Scores are bounded 0-100.",
            "data_type": "Public proxy / demo assumption",
            "production": "Replace with internal servicing, payment timeliness, waivers, amendments.",
        },
        {
            "label": "Market-Implied Risk",
            "purpose": "Captures market-based deterioration signals (negative momentum, volatility, drawdown).",
            "inputs": "3-month return, 3-month annualized vol, 6-month drawdown",
            "formula": "average of absolute threshold_score(3-month return, 3-month annualized volatility, 6-month drawdown)",
            "data_type": "Public market data (yfinance)",
            "production": "Replace with approved Bloomberg/Refinitiv/FactSet feed.",
        },
        {
            "label": "News & Sentiment",
            "purpose": "Detects negative public events affecting credit quality.",
            "inputs": "News counts when available; otherwise Market score, Financial score, and Sector quality score",
            "formula": "If news is available: 40% general-news score + 60% risk-news score. If unavailable: 50% Market score + 30% Financial score + 20% Sector-quality score.",
            "data_type": "Public news when available; otherwise documented public-data assumption",
            "production": "GDELT or vendor news/sentiment feed; activate event extraction.",
        },
        {
            "label": "Accounting Integrity",
            "purpose": "Detects quality-of-earnings concerns or concealed deterioration.",
            "inputs": "Sloan accruals and Beneish M-Score where available; Financial Performance and Data Quality for fallback",
            "formula": "If Beneish is available: average Sloan accruals score and Beneish score. If unavailable: 50% Financial score + 30% Data Quality score + 20% Sloan score.",
            "data_type": "Public SEC forensic diagnostics with explicit fallback when Beneish inputs are unavailable.",
            "production": "Add Dechow F-Score, AR/Inventory divergence, reserve adequacy, restatement flags, auditor changes.",
        },
        {
            "label": "Connectivity & Contagion",
            "purpose": "Captures risk from sector concentration and cross-obligor links.",
            "inputs": "Market score, Sector quality score, News score, Data Quality score; sector density shown as context",
            "formula": "Score = 45% Market + 25% Sector quality + 20% News + 10% Data Quality + 5. Scores are bounded 0-100.",
            "data_type": "Public proxy / demo assumption",
            "production": "Replace with legal-entity, ownership, supply-chain, cross-obligor exposure data.",
        },
        {
            "label": "Governance & Strategic Discipline",
            "purpose": "Captures governance, capital allocation, management instability.",
            "inputs": "Accounting Integrity score, News & Sentiment score, Financial score, Data Quality score",
            "formula": "Score = 45% Accounting + 25% News + 20% Financial + 10% Data Quality + 5. Risk proxy = 100 - score. Scores are bounded 0-100.",
            "data_type": "Public proxy / demo assumption",
            "production": "Add management turnover, insider transactions, governance events, M&A behavior.",
        },
        {
            "label": "Collateral & Recovery",
            "purpose": "Captures loss-severity / recovery risk if borrower deteriorates.",
            "inputs": "Debt/Assets sub-score, Net Debt/EBITDA sub-score, Liabilities/Assets sub-score, Financial score",
            "formula": "Score = 40% Debt/Assets + 25% Net Debt/EBITDA + 20% Leverage + 15% Financial + 5. Risk proxy = 100 - score. Scores are bounded 0-100.",
            "data_type": "Public proxy / demo assumption",
            "production": "Replace with internal collateral, lien, appraisal, margining, recovery data.",
        },
    ]

    for meth in METHODOLOGY:
        with st.expander(f"{meth['label']}", expanded=False):
            st.markdown(f"**Purpose:** {meth['purpose']}")
            st.markdown(f"**Inputs:** {meth['inputs']}")
            st.markdown(f"**Formula:**")
            st.code(meth["formula"], language="text")
            st.markdown(f"**Current data type:** {meth['data_type']}")
            st.markdown(f"**Production enhancement:** {meth['production']}")

# ===========================================================================
# Page 5 - Data Coverage & Assumptions (doc Section 14, Page 4)
# ===========================================================================

with tab_quality:
    st.subheader("Data coverage and assumptions")
    st.caption(
        "Transparency layer: shows where the score relies on real public data "
        "vs. sector-level assumptions or stale SEC facts."
    )

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Avg data quality", f"{fdf['data_quality_score'].mean():.1f}")
    q2.metric("Companies with stale facts",
              int((fdf["stale_fact_count"] > 0).sum()))
    q3.metric("Companies with assumptions",
              int((fdf["assumption_metric_count"] > 0).sum()))
    q4.metric("SEC unavailable", int((fdf["sec_available"] == False).sum()))

    st.markdown("---")

    cq1, cq2 = st.columns(2)
    with cq1:
        st.markdown("##### Data quality distribution")
        fig_dq = px.histogram(
            fdf, x="data_quality_score", nbins=20, range_x=[0, 100]
        )
        fig_dq.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            xaxis_title="Data Quality Score",
            yaxis_title="Companies",
        )
        st.plotly_chart(fig_dq, use_container_width=True)

    with cq2:
        st.markdown("##### Avg data quality by sector")
        sector_dq = (
            fdf.groupby("sector_group")
            .agg(avg_dq=("data_quality_score", "mean"), n=("ticker", "count"))
            .reset_index()
            .sort_values("avg_dq")
        )
        fig_sdq = px.bar(
            sector_dq, x="avg_dq", y="sector_group", orientation="h",
            color="avg_dq", color_continuous_scale="Blues",
            text=sector_dq["avg_dq"].round(0),
        )
        fig_sdq.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            xaxis_title="Avg Data Quality",
            yaxis_title="",
        )
        st.plotly_chart(fig_sdq, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Per-company data quality")
    coverage_table = (
        fdf[
            [
                "ticker", "company_name", "sector_group",
                "sec_available", "assumption_metric_count", "stale_fact_count",
                "data_quality_score", "data_quality_note",
            ]
        ]
        .sort_values("data_quality_score")
        .rename(
            columns={
                "ticker": "Ticker",
                "company_name": "Company",
                "sector_group": "Sector",
                "sec_available": "SEC Available",
                "assumption_metric_count": "Assumption Count",
                "stale_fact_count": "Stale Fact Count",
                "data_quality_score": "Data Quality Score",
                "data_quality_note": "Data Quality Note",
            }
        )
    )
    st.dataframe(
        coverage_table.style.format({"Data Quality Score": "{:.0f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("##### Flag detail (assumption / stale fact)")
    if diag.empty:
        st.info("Diagnostics file not found — re-run the data prep script to generate it.")
    else:
        diag_filt = diag[diag["ticker"].isin(fdf["ticker"])].copy()
        st.dataframe(
            diag_filt.rename(
                columns={
                    "ticker": "Ticker",
                    "company_name": "Company",
                    "sector_group": "Sector",
                    "metric": "Metric",
                    "flag_type": "Flag",
                    "fact_period_end": "Fact Period End",
                    "reference_period_end": "Reference Period End",
                }
            ).sort_values(["Flag", "Ticker"]),
            use_container_width=True,
            hide_index=True,
        )


st.markdown("---")
st.caption(
    "Built from public data only. Internal-bank-only indicators "
    "(Facility & Liquidity, Behavioral & Payment, Connectivity & Contagion, "
    "Governance, Collateral & Recovery) are public-data proxies and should be "
    "replaced with internal data in production. See the project documentation "
    "for the full methodology and known limitations."
)
