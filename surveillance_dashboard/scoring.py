"""
Surveillance Tool — Scoring Module
==================================

Implements the 9 indicator scores, composite Surveillance Score, alert level,
and top-3 driver logic exactly as defined in
`surveillance_tool_project_documentation.md` Sections 9 - 13.

This module is intentionally separate from the dashboard UI so the formulas
can be unit-tested and reviewed by audit / model governance independently.

All inputs come from `surveillance_dashboard_input.csv` produced by
`build_surveillance_data.py`. No I/O happens here — pass in a DataFrame,
get back a DataFrame with score columns appended.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants from documentation
# ---------------------------------------------------------------------------

# Sector base risk used by the Facility & Liquidity proxy (doc 9.1).
SECTOR_BASE: Dict[str, float] = {
    "Financials": 45,
    "Technology & Communications": 50,
    "Technology": 42,
    "Utilities": 48,
    "Energy": 44,
    "Transportation": 58,
    "Autos": 55,
    "Aerospace & Defense": 48,
    "Industrials": 45,
    "Consumer Staples": 38,
    "Consumer Discretionary": 48,
    "Communication Services": 45,
    "Healthcare": 40,
    "Real Estate": 52,
    "Materials": 50,
}

INDICATOR_FIELDS: List[str] = [
    "facility_liquidity_score",
    "financial_performance_score",
    "behavioral_payment_score",
    "market_implied_risk_score",
    "news_sentiment_score",
    "accounting_integrity_score",
    "connectivity_contagion_score",
    "governance_discipline_score",
    "collateral_recovery_score",
]

INDICATOR_LABELS: Dict[str, str] = {
    "facility_liquidity_score": "Facility & Liquidity",
    "financial_performance_score": "Financial Performance",
    "behavioral_payment_score": "Behavioral & Payment",
    "market_implied_risk_score": "Market-Implied Risk",
    "news_sentiment_score": "News & Sentiment",
    "accounting_integrity_score": "Accounting Integrity",
    "connectivity_contagion_score": "Connectivity & Contagion",
    "governance_discipline_score": "Governance & Strategic Discipline",
    "collateral_recovery_score": "Collateral & Recovery",
}

# Composite weights from EWIF v1.0 Section V.A "Proposed Factor Weighting".
# Sum = 100. Previously summed to 102 because Financial Performance was set
# to 17 instead of the doc's 15.
INDICATOR_WEIGHTS: Dict[str, float] = {
    "facility_liquidity_score": 0.17,
    "financial_performance_score": 0.15,
    "behavioral_payment_score": 0.08,
    "market_implied_risk_score": 0.10,
    "news_sentiment_score": 0.10,
    "accounting_integrity_score": 0.15,
    "connectivity_contagion_score": 0.10,
    "governance_discipline_score": 0.10,
    "collateral_recovery_score": 0.05,
}
assert abs(sum(INDICATOR_WEIGHTS.values()) - 1.0) < 1e-9, "Composite weights must sum to 1.0"

ALERT_BINS: List[Tuple[float, float, str]] = [
    (0.0, 35.0, "Red"),
    (35.0, 45.0, "Orange"),
    (45.0, 55.0, "Yellow"),
    (55.0, 100.01, "Green"),
]

ALERT_COLORS: Dict[str, str] = {
    "Green": "#16a34a",
    "Yellow": "#eab308",
    "Orange": "#f97316",
    "Red": "#dc2626",
}

# ---------------------------------------------------------------------------
# EWIF v1.0 Section V.C — Alert Thresholds and Escalation
#
# Score convention: higher Surveillance Score = better credit quality / lower risk.
#
#   Tier 1  high-conviction deterioration   composite < 40   OR  >=2 indicators <= 25
#   Tier 2  accelerating stress              40 <= composite < 55  AND velocity_trigger
#   Tier 3  monitoring queue                 composite >= 55        AND velocity_trigger
#   None    no concerns                      otherwise
#
# Velocity trigger (computed from quarterly history): >=2 of {YoY revenue
# decline, margin contraction >2pp YoY, debt/assets up >5pp YoY}.
# ---------------------------------------------------------------------------

TIER1_LEVEL = 40.0
TIER2_LEVEL = 55.0
TIER1_INDICATOR_SEVERE = 25.0
TIER1_MIN_SEVERE_COUNT = 2

TIER_LABELS: Dict[str, str] = {
    "Tier 1": "Tier 1 — High-Conviction Deterioration",
    "Tier 2": "Tier 2 — Accelerating Stress",
    "Tier 3": "Tier 3 — Monitoring Queue",
    "None": "No Tier — Within Tolerance",
}

TIER_COLORS: Dict[str, str] = {
    "Tier 1": "#dc2626",
    "Tier 2": "#f97316",
    "Tier 3": "#eab308",
    "None": "#16a34a",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def percentile_score(series: pd.Series, higher_is_risk: bool = True) -> pd.Series:
    """Cross-sectional score scaled to 0-100.

    Output convention is always higher = better credit quality / lower risk.
    If a raw input is riskier when higher, high raw values receive low scores.
    If a raw input is stronger when higher, high raw values receive high scores.
    Missing values default to neutral 50.
    """
    s = pd.to_numeric(series, errors="coerce")
    pct = s.rank(pct=True, method="average") * 100
    if higher_is_risk:
        pct = 100 - pct
    return pct.fillna(50).clip(0, 100)


def threshold_score(series: pd.Series, breakpoints: List[Tuple[float, float]]) -> pd.Series:
    """Absolute threshold score scaled to 0-100.

    Breakpoints are ordered as (raw_value, quality_score). Values between
    breakpoints are linearly interpolated and outside values are clamped. This
    avoids making a fundamentally strong company look weak simply because it is
    compared cross-sectionally against a broad mixed-sector universe.
    """
    s = pd.to_numeric(series, errors="coerce")
    points = sorted(breakpoints, key=lambda x: x[0])
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    scored = pd.Series(np.interp(s, x, y), index=series.index)
    return scored.fillna(50).clip(0, 100)


def assign_alert(score: float) -> str:
    if pd.isna(score):
        return "Yellow"
    for lo, hi, name in ALERT_BINS:
        if lo <= score < hi:
            return name
    return "Red"


# ---------------------------------------------------------------------------
# Beneish M-Score (EWIF v1.0 Section VI.B)
# ---------------------------------------------------------------------------
#
# 8-variable model:
#   M = -4.84
#       + 0.92  * DSRI    Days Sales in Receivables Index
#       + 0.528 * GMI     Gross Margin Index (prior / current)
#       + 0.404 * AQI     Asset Quality Index
#       + 0.892 * SGI     Sales Growth Index
#       + 0.115 * DEPI    Depreciation Index (prior rate / current rate)
#       - 0.172 * SGAI    SG&A Index
#       + 4.679 * TATA    Total Accruals to Total Assets
#       - 0.327 * LVGI    Leverage Index
#
# M > -1.78 indicates a higher likelihood of earnings manipulation (the
# "M-Score threshold" from Beneish 1999). Our scoring converts the raw M to
# a 0-100 risk percentile so it can be blended with the other indicators.

_BENEISH_INSTANT_METRICS = ("accounts_receivable", "ppe_net", "current_assets", "assets", "liabilities")
_BENEISH_FLOW_METRICS = ("revenue", "cogs", "depreciation", "sga", "net_income", "cfo")


def _wide_history(history_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long history into ticker x period_end x metric values, picking
    instant facts for balance-sheet metrics and TTM (else quarterly) for flows."""
    if history_df.empty:
        return pd.DataFrame()

    history_df = history_df.copy()
    history_df["period_end"] = pd.to_datetime(history_df["period_end"], errors="coerce")

    wanted_instant = list(_BENEISH_INSTANT_METRICS)
    wanted_flow = list(_BENEISH_FLOW_METRICS)

    inst = history_df[
        (history_df["metric"].isin(wanted_instant)) & (history_df["value_type"] == "instant")
    ]
    ttm = history_df[
        (history_df["metric"].isin(wanted_flow)) & (history_df["value_type"] == "ttm")
    ]
    qtr_fallback = history_df[
        (history_df["metric"].isin(wanted_flow)) & (history_df["value_type"] == "quarterly")
    ]

    parts = []
    for chunk in (inst, ttm):
        if not chunk.empty:
            parts.append(
                chunk.pivot_table(
                    index=["ticker", "period_end"], columns="metric", values="value", aggfunc="last"
                )
            )
    if not parts:
        return pd.DataFrame()
    wide = parts[0]
    for extra in parts[1:]:
        wide = wide.join(extra, how="outer")

    if not qtr_fallback.empty:
        # Fill missing TTM cells with quarterly-derived TTM via 4-quarter rolling sum.
        q_wide = qtr_fallback.pivot_table(
            index=["ticker", "period_end"], columns="metric", values="value", aggfunc="last"
        ).sort_index()
        rolled = q_wide.groupby(level="ticker", group_keys=False).apply(
            lambda g: g.rolling(window=4, min_periods=4).sum()
        )
        for col in rolled.columns:
            if col in wide.columns:
                wide[col] = wide[col].combine_first(rolled[col])
            else:
                wide[col] = rolled[col]

    return wide.sort_index()


def _safe_ratio(num: float, den: float) -> float:
    if den is None or pd.isna(den) or den == 0:
        return np.nan
    if num is None or pd.isna(num):
        return np.nan
    return float(num) / float(den)


def compute_beneish_m_score(history_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the Beneish M-Score and components for each ticker, using the
    latest year-over-year comparable TTM/instant snapshot pair (period_end vs.
    ~365 days earlier).

    Returns one row per ticker with columns:
      ticker, beneish_m, beneish_period_t, beneish_period_t_minus_1,
      beneish_dsri, beneish_gmi, beneish_aqi, beneish_sgi,
      beneish_depi, beneish_sgai, beneish_tata, beneish_lvgi
    """
    wide = _wide_history(history_df)
    if wide.empty:
        return pd.DataFrame(columns=["ticker", "beneish_m"])

    rows: List[Dict[str, float]] = []

    for ticker, sub in wide.groupby(level="ticker"):
        # Reduce to (period_end -> row dict).
        sub = sub.reset_index().set_index("period_end").sort_index()
        # Need every input metric for both t and t-4Q.
        required = list(_BENEISH_INSTANT_METRICS) + ["revenue", "cogs", "depreciation", "sga", "net_income", "cfo"]
        # Keep periods where at least revenue + assets exist.
        valid = sub.dropna(subset=["revenue", "assets"], how="any")
        if len(valid) < 5:
            continue

        # Walk backwards from the most recent period until we find a pair that
        # has all required values for both t and t-1 (~4 quarters earlier).
        latest_pair = None
        for i in range(len(valid) - 1, -1, -1):
            period_t = valid.index[i]
            target_prior = period_t - pd.Timedelta(days=365)
            # Match the closest period within +/- 60 days of one year prior.
            mask = (sub.index >= target_prior - pd.Timedelta(days=60)) & (sub.index <= target_prior + pd.Timedelta(days=60))
            candidates = sub.loc[mask]
            if candidates.empty:
                continue
            period_tm1 = candidates.index[(candidates.index - target_prior).map(abs).argmin()]
            row_t = sub.loc[period_t]
            row_tm1 = sub.loc[period_tm1]
            ok = all(pd.notna(row_t.get(m)) and pd.notna(row_tm1.get(m)) for m in required if m != "cogs" and m != "sga" and m != "depreciation")
            if ok:
                latest_pair = (period_t, period_tm1, row_t, row_tm1)
                break

        if latest_pair is None:
            continue

        period_t, period_tm1, t, tm1 = latest_pair

        # Variable definitions.
        dsri = _safe_ratio(_safe_ratio(t["accounts_receivable"], t["revenue"]),
                           _safe_ratio(tm1["accounts_receivable"], tm1["revenue"]))

        if pd.notna(t.get("cogs")) and pd.notna(tm1.get("cogs")):
            gm_t = _safe_ratio(t["revenue"] - t["cogs"], t["revenue"])
            gm_tm1 = _safe_ratio(tm1["revenue"] - tm1["cogs"], tm1["revenue"])
            gmi = _safe_ratio(gm_tm1, gm_t)
        else:
            gmi = np.nan  # Banks/insurance don't report COGS; will neutralize below.

        # AQI: 1 - (CurrentAssets + PP&E)/TotalAssets, all instant
        if all(pd.notna(t.get(k)) for k in ("current_assets", "ppe_net", "assets")) and \
           all(pd.notna(tm1.get(k)) for k in ("current_assets", "ppe_net", "assets")):
            aq_t = 1 - (t["current_assets"] + t["ppe_net"]) / t["assets"]
            aq_tm1 = 1 - (tm1["current_assets"] + tm1["ppe_net"]) / tm1["assets"]
            aqi = _safe_ratio(aq_t, aq_tm1)
        else:
            aqi = np.nan

        sgi = _safe_ratio(t["revenue"], tm1["revenue"])

        if pd.notna(t.get("depreciation")) and pd.notna(tm1.get("depreciation")):
            dep_rate_t = _safe_ratio(t["depreciation"], t["depreciation"] + t["ppe_net"])
            dep_rate_tm1 = _safe_ratio(tm1["depreciation"], tm1["depreciation"] + tm1["ppe_net"])
            depi = _safe_ratio(dep_rate_tm1, dep_rate_t)
        else:
            depi = np.nan

        if pd.notna(t.get("sga")) and pd.notna(tm1.get("sga")):
            sgai = _safe_ratio(_safe_ratio(t["sga"], t["revenue"]),
                               _safe_ratio(tm1["sga"], tm1["revenue"]))
        else:
            sgai = np.nan

        tata = _safe_ratio(t["net_income"] - t["cfo"], t["assets"])
        lvgi = _safe_ratio(_safe_ratio(t["liabilities"], t["assets"]),
                           _safe_ratio(tm1["liabilities"], tm1["assets"]))

        # Neutralize missing components by setting them to 1.0 (no change YoY).
        # For TATA we set to 0.0 if missing.
        comps = {
            "dsri": dsri if pd.notna(dsri) else 1.0,
            "gmi": gmi if pd.notna(gmi) else 1.0,
            "aqi": aqi if pd.notna(aqi) else 1.0,
            "sgi": sgi if pd.notna(sgi) else 1.0,
            "depi": depi if pd.notna(depi) else 1.0,
            "sgai": sgai if pd.notna(sgai) else 1.0,
            "tata": tata if pd.notna(tata) else 0.0,
            "lvgi": lvgi if pd.notna(lvgi) else 1.0,
        }

        # Apply the Beneish (1999) coefficients.
        m = (
            -4.84
            + 0.92 * comps["dsri"]
            + 0.528 * comps["gmi"]
            + 0.404 * comps["aqi"]
            + 0.892 * comps["sgi"]
            + 0.115 * comps["depi"]
            - 0.172 * comps["sgai"]
            + 4.679 * comps["tata"]
            - 0.327 * comps["lvgi"]
        )

        rows.append({
            "ticker": ticker,
            "beneish_m": float(m),
            "beneish_period_t": pd.Timestamp(period_t).strftime("%Y-%m-%d"),
            "beneish_period_t_minus_1": pd.Timestamp(period_tm1).strftime("%Y-%m-%d"),
            "beneish_dsri": float(comps["dsri"]),
            "beneish_gmi": float(comps["gmi"]),
            "beneish_aqi": float(comps["aqi"]),
            "beneish_sgi": float(comps["sgi"]),
            "beneish_depi": float(comps["depi"]),
            "beneish_sgai": float(comps["sgai"]),
            "beneish_tata": float(comps["tata"]),
            "beneish_lvgi": float(comps["lvgi"]),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Velocity trigger (used by Tier 2 / Tier 3 alert classification)
# ---------------------------------------------------------------------------


def compute_velocity_signals(history_df: pd.DataFrame) -> pd.DataFrame:
    """Compute a YoY deterioration count per ticker. >=2 deteriorating
    fundamentals constitutes a 'velocity trigger' per EWIF v1.0 Section V.C.

    Signals tracked:
      1. TTM revenue YoY change < 0
      2. TTM net margin YoY change < -2 percentage points
      3. Debt/assets YoY change > +5 percentage points
    """
    if history_df.empty:
        return pd.DataFrame(columns=["ticker", "velocity_signal_count", "velocity_trigger"])

    rows: List[Dict[str, float]] = []
    h = history_df.copy()
    h["period_end"] = pd.to_datetime(h["period_end"], errors="coerce")

    rev = h[(h["metric"] == "revenue") & (h["value_type"] == "ttm")]
    nm = h[(h["metric"] == "net_margin") & (h["value_type"] == "ratio")]
    dta = h[(h["metric"] == "debt_to_assets") & (h["value_type"] == "ratio")]

    def _yoy(sub: pd.DataFrame) -> Tuple[float, float]:
        if sub.empty:
            return (np.nan, np.nan)
        s = sub.sort_values("period_end")
        latest = s.iloc[-1]
        target_prior = latest["period_end"] - pd.Timedelta(days=365)
        prior_idx = (s["period_end"] - target_prior).abs().idxmin()
        prior = s.loc[prior_idx]
        if abs((prior["period_end"] - target_prior).days) > 60:
            return (latest["value"], np.nan)
        return (latest["value"], prior["value"])

    for ticker in h["ticker"].dropna().unique():
        rev_t, rev_tm1 = _yoy(rev[rev["ticker"] == ticker])
        nm_t, nm_tm1 = _yoy(nm[nm["ticker"] == ticker])
        dta_t, dta_tm1 = _yoy(dta[dta["ticker"] == ticker])

        signals = 0
        # Revenue contraction.
        if pd.notna(rev_t) and pd.notna(rev_tm1) and rev_tm1 != 0:
            if (rev_t - rev_tm1) / abs(rev_tm1) < 0:
                signals += 1
        # Margin compression > 2pp.
        if pd.notna(nm_t) and pd.notna(nm_tm1):
            if (nm_t - nm_tm1) < -0.02:
                signals += 1
        # Leverage build > 5pp.
        if pd.notna(dta_t) and pd.notna(dta_tm1):
            if (dta_t - dta_tm1) > 0.05:
                signals += 1

        rows.append({
            "ticker": ticker,
            "velocity_signal_count": signals,
            "velocity_trigger": bool(signals >= 2),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Financial Performance enhancement signals
# ---------------------------------------------------------------------------


def compute_fcf_sustainability(history_df: pd.DataFrame) -> pd.DataFrame:
    """Compute FCF sustainability sub-signals from quarterly history.

    These are public-data approximations of the EWIF v1.0 FCF Sustainability
    sub-index:
      1. count of negative quarterly FCF periods in the latest four quarters
      2. TTM FCF / EBITDA conversion
      3. cash burn rate = abs(TTM FCF) / cash when TTM FCF is negative
    """
    if history_df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "negative_fcf_quarters_l4",
                "fcf_ebitda_conversion",
                "cash_burn_to_cash",
                "fcf_sustainability_period",
            ]
        )

    h = history_df.copy()
    h["period_end"] = pd.to_datetime(h["period_end"], errors="coerce")

    ttm_metrics = ["fcf", "net_income", "interest_expense", "depreciation"]
    ttm = h[(h["metric"].isin(ttm_metrics)) & (h["value_type"] == "ttm")]
    instant = h[(h["metric"] == "cash") & (h["value_type"] == "instant")]
    q_flows = h[(h["metric"].isin(["cfo", "capex"])) & (h["value_type"] == "quarterly")]

    if ttm.empty:
        return pd.DataFrame(columns=["ticker"])

    ttm_wide = ttm.pivot_table(
        index=["ticker", "period_end"], columns="metric", values="value", aggfunc="last"
    ).sort_index()
    cash_wide = instant.pivot_table(
        index=["ticker", "period_end"], columns="metric", values="value", aggfunc="last"
    ).sort_index()
    wide = ttm_wide.join(cash_wide, how="outer").sort_index()

    q_fcf = pd.DataFrame()
    if not q_flows.empty:
        q_wide = q_flows.pivot_table(
            index=["ticker", "period_end"], columns="metric", values="value", aggfunc="last"
        ).sort_index()
        if {"cfo", "capex"}.issubset(q_wide.columns):
            q_fcf = (q_wide["cfo"] - q_wide["capex"]).rename("quarterly_fcf").reset_index()

    rows: List[Dict[str, float]] = []
    for ticker, sub in wide.groupby(level="ticker"):
        sub = sub.reset_index().set_index("period_end").sort_index()
        valid = sub.dropna(subset=["fcf"], how="any")
        if valid.empty:
            continue

        latest_period = valid.index[-1]
        latest = sub.loc[latest_period]
        net_income = latest.get("net_income", np.nan)
        ebitda = np.nan
        if pd.notna(net_income):
            ebitda = (
                net_income
                + (latest.get("interest_expense", 0.0) if pd.notna(latest.get("interest_expense", np.nan)) else 0.0)
                + (latest.get("depreciation", 0.0) if pd.notna(latest.get("depreciation", np.nan)) else 0.0)
            )
        fcf = latest.get("fcf", np.nan)
        cash = latest.get("cash", np.nan)

        if pd.notna(ebitda) and ebitda > 0:
            fcf_conversion = _safe_ratio(fcf, ebitda)
        else:
            fcf_conversion = np.nan

        if pd.notna(fcf) and pd.notna(cash) and cash > 0 and fcf < 0:
            cash_burn = abs(float(fcf)) / float(cash)
        elif pd.notna(fcf) and fcf >= 0:
            cash_burn = 0.0
        else:
            cash_burn = np.nan

        neg_quarters = np.nan
        if not q_fcf.empty:
            tq = q_fcf[q_fcf["ticker"] == ticker].sort_values("period_end")
            tq = tq[tq["period_end"] <= latest_period].tail(4)
            if len(tq) > 0:
                neg_quarters = int((tq["quarterly_fcf"] < 0).sum())

        rows.append(
            {
                "ticker": ticker,
                "negative_fcf_quarters_l4": neg_quarters,
                "fcf_ebitda_conversion": fcf_conversion,
                "cash_burn_to_cash": cash_burn,
                "fcf_sustainability_period": pd.Timestamp(latest_period).strftime("%Y-%m-%d"),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tier classification (EWIF v1.0 Section V.C)
# ---------------------------------------------------------------------------


def assign_tier(score: float, indicator_scores: Dict[str, float], velocity_trigger: bool) -> str:
    """Convert composite score + indicator-level severity + velocity flag into
    a Tier 1/2/3 label. Higher score means better credit quality."""
    if pd.isna(score):
        return "None"

    severe_count = sum(1 for v in indicator_scores.values() if pd.notna(v) and v <= TIER1_INDICATOR_SEVERE)
    if score < TIER1_LEVEL or severe_count >= TIER1_MIN_SEVERE_COUNT:
        return "Tier 1"
    if score < TIER2_LEVEL and velocity_trigger:
        return "Tier 2"
    if velocity_trigger:
        return "Tier 3"
    return "None"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def compute_scores(df: pd.DataFrame, history_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute all 9 indicator scores, the composite Surveillance Score, the
    alert level, and the top 3 drivers per company.

    Expects columns from `surveillance_dashboard_input.csv`:
        leverage_proxy, debt_to_assets, cash_to_assets, net_margin, fcf_to_assets,
        accrual_proxy, current_ratio,
        ret_3m, vol_3m, drawdown_6m,
        news_mentions_30d, risk_news_mentions_30d, news_data_available,
        sector_group
    """
    df = df.copy()

    # --- Beneish M-Score (if quarterly history is available) -----------------
    # Per EWIF v1.0 Section VI.B, Beneish lives inside Accounting Integrity
    # but it depends on YoY snapshots that only the history file carries, so
    # we compute it here and merge in.
    if history_df is not None and not history_df.empty and "beneish_m" not in df.columns:
        m_scores = compute_beneish_m_score(history_df)
        if not m_scores.empty:
            df = df.merge(m_scores, on="ticker", how="left")
        else:
            df["beneish_m"] = np.nan
    elif "beneish_m" not in df.columns:
        df["beneish_m"] = np.nan

    # --- Velocity signals (used by Tier classification) ----------------------
    if history_df is not None and not history_df.empty and "velocity_signal_count" not in df.columns:
        v = compute_velocity_signals(history_df)
        if not v.empty:
            df = df.merge(v, on="ticker", how="left")
    if "velocity_signal_count" not in df.columns:
        df["velocity_signal_count"] = 0
        df["velocity_trigger"] = False
    df["velocity_signal_count"] = df["velocity_signal_count"].fillna(0).astype(int)
    df["velocity_trigger"] = df["velocity_trigger"].fillna(False).astype(bool)

    # --- FCF sustainability (EWIF v1.0 Financial Performance sub-index) ------
    if history_df is not None and not history_df.empty and "negative_fcf_quarters_l4" not in df.columns:
        fcf_sustainability = compute_fcf_sustainability(history_df)
        if not fcf_sustainability.empty:
            df = df.merge(fcf_sustainability, on="ticker", how="left")
    for col in ["negative_fcf_quarters_l4", "fcf_ebitda_conversion", "cash_burn_to_cash"]:
        if col not in df.columns:
            df[col] = np.nan

    # --- 9.2 Financial Performance Indicator ---------------------------------
    # The base v1 implementation used six broad public ratios. This EWIF
    # alignment pass adds repayment-capacity signals available from current
    # SEC fields: net debt / EBITDA, EBITDA / interest, and an FCF
    # sustainability sub-score from quarterly history when present.
    numeric_cols = [
        "net_income", "interest_expense", "depreciation", "total_debt", "cash"
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ebitda_proxy"] = np.where(
        df["net_income"].notna(),
        df["net_income"] + df["interest_expense"].fillna(0.0) + df["depreciation"].fillna(0.0),
        np.nan,
    )
    df["net_debt"] = df["total_debt"] - df["cash"]
    positive_ebitda = df["ebitda_proxy"] > 0
    positive_interest = df["interest_expense"] > 0
    df["net_debt_to_ebitda"] = np.where(
        positive_ebitda,
        df["net_debt"] / df["ebitda_proxy"],
        np.nan,
    )
    df["ebitda_interest_coverage"] = np.where(
        positive_interest,
        df["ebitda_proxy"] / df["interest_expense"],
        np.nan,
    )

    df["score_leverage"] = threshold_score(
        df["leverage_proxy"],
        [(0.00, 100), (0.35, 90), (0.55, 70), (0.75, 40), (0.90, 20), (1.00, 0)],
    )
    df["score_debt"] = threshold_score(
        df["debt_to_assets"],
        [(0.00, 100), (0.20, 95), (0.40, 80), (0.60, 55), (0.80, 30), (1.00, 10)],
    )
    df["score_net_debt_ebitda"] = threshold_score(
        df["net_debt_to_ebitda"],
        [(0.00, 100), (1.00, 95), (2.50, 80), (4.00, 55), (6.00, 30), (8.00, 10)],
    )
    df.loc[(~positive_ebitda) & (df["net_debt"] > 0), "score_net_debt_ebitda"] = 0.0
    df["score_interest_coverage"] = threshold_score(
        df["ebitda_interest_coverage"],
        [(0.00, 0), (1.00, 20), (2.00, 40), (4.00, 65), (8.00, 85), (12.00, 100)],
    )
    df.loc[(df["ebitda_proxy"] <= 0) & positive_interest, "score_interest_coverage"] = 0.0
    df["score_cash"] = threshold_score(
        df["cash_to_assets"],
        [(0.00, 20), (0.03, 40), (0.08, 65), (0.15, 85), (0.25, 100)],
    )
    df["score_margin"] = threshold_score(
        df["net_margin"],
        [(-0.20, 0), (0.00, 40), (0.05, 60), (0.10, 75), (0.20, 90), (0.30, 100)],
    )
    df["score_fcf"] = threshold_score(
        df["fcf_to_assets"],
        [(-0.10, 0), (0.00, 45), (0.03, 65), (0.07, 85), (0.12, 100)],
    )
    df["score_current_ratio"] = threshold_score(
        df["current_ratio"],
        [(0.00, 0), (0.75, 25), (1.00, 50), (1.50, 75), (2.50, 95), (4.00, 100)],
    )
    df["score_negative_fcf_quarters"] = threshold_score(
        df["negative_fcf_quarters_l4"],
        [(0.00, 100), (1.00, 75), (2.00, 50), (3.00, 25), (4.00, 0)],
    )
    df["score_fcf_conversion"] = threshold_score(
        df["fcf_ebitda_conversion"],
        [(-0.50, 0), (0.00, 40), (0.40, 65), (0.80, 85), (1.20, 100)],
    )
    df["score_cash_burn"] = threshold_score(
        df["cash_burn_to_cash"],
        [(0.00, 100), (0.25, 80), (0.50, 60), (1.00, 35), (2.00, 10)],
    )
    df["score_fcf_sustainability"] = df[
        ["score_negative_fcf_quarters", "score_fcf_conversion", "score_cash_burn"]
    ].mean(axis=1)
    df["financial_performance_score"] = df[
        [
            "score_leverage",
            "score_debt",
            "score_net_debt_ebitda",
            "score_interest_coverage",
            "score_cash",
            "score_margin",
            "score_fcf",
            "score_fcf_sustainability",
            "score_current_ratio",
        ]
    ].mean(axis=1)

    # --- 9.4 Market-Implied Risk Indicator -----------------------------------
    df["score_negative_momentum"] = threshold_score(
        df["ret_3m"],
        [(-0.30, 0), (-0.15, 25), (0.00, 50), (0.10, 75), (0.25, 100)],
    )
    df["score_volatility"] = threshold_score(
        df["vol_3m"],
        [(0.00, 100), (0.15, 85), (0.25, 65), (0.40, 40), (0.60, 15), (1.00, 0)],
    )
    df["score_drawdown"] = threshold_score(
        df["drawdown_6m"],
        [(-0.60, 0), (-0.40, 20), (-0.25, 45), (-0.10, 70), (0.00, 100)],
    )
    df["market_implied_risk_score"] = df[
        ["score_negative_momentum", "score_volatility", "score_drawdown"]
    ].mean(axis=1)

    # --- Sector inputs for proxy indicators and assumptions ------------------
    # External credit-rating style buckets are intentionally not used in
    # scoring. Most available values in this public demo are synthetic, so
    # using them would add false precision to the score.
    sector_base = df["sector_group"].map(SECTOR_BASE).fillna(45)
    df["sector_base_risk"] = sector_base
    df["sector_quality_score"] = threshold_score(
        sector_base,
        [(35, 92), (40, 85), (45, 75), (50, 65), (55, 55), (60, 45), (65, 35)],
    )

    # --- 9.6 Accounting Integrity Indicator ----------------------------------
    # Blend two forensic signals:
    #   (a) Sloan Accruals       (existing accrual_proxy = (NI - CFO)/Assets)
    #   (b) Beneish M-Score      (8-variable, computed from history if available)
    # If Beneish is not available for a company, use a documented public-data
    # assumption built from financial strength, data quality, and Sloan
    # accruals. This keeps Accounting Integrity directional without treating a
    # missing Beneish model as a clean bill of health.
    df["score_sloan_accruals"] = threshold_score(
        df["accrual_proxy"],
        [(-0.20, 100), (-0.10, 90), (0.00, 80), (0.05, 65), (0.10, 45), (0.20, 20), (0.35, 0)],
    )
    if "beneish_m" in df.columns and df["beneish_m"].notna().any():
        df["score_beneish"] = threshold_score(
            df["beneish_m"],
            [(-3.50, 95), (-2.50, 85), (-2.00, 70), (-1.78, 55), (-1.50, 35), (-1.00, 15), (0.00, 0)],
        )
        beneish_present = df["beneish_m"].notna()
        df["accounting_integrity_assumption_used"] = ~beneish_present
    else:
        df["score_beneish"] = np.nan
        df["accounting_integrity_assumption_used"] = True
        beneish_present = pd.Series(False, index=df.index)

    data_quality = pd.to_numeric(df.get("data_quality_score", pd.Series([75.0] * len(df))), errors="coerce").fillna(75.0)
    beneish_based = df[["score_sloan_accruals", "score_beneish"]].mean(axis=1)
    fallback_assumption = (
        0.50 * df["financial_performance_score"]
        + 0.30 * data_quality
        + 0.20 * df["score_sloan_accruals"]
    ).clip(0, 100)
    df["accounting_integrity_score"] = np.where(
        beneish_present,
        beneish_based,
        fallback_assumption,
    )
    df["fraud_quality_subscore"] = np.where(
        beneish_present,
        beneish_based,
        df["score_sloan_accruals"],
    )
    df["fraud_risk_subscore"] = (100 - df["fraud_quality_subscore"]).clip(0, 100)
    df["fraud_watch_flag"] = (
        (pd.to_numeric(df.get("beneish_m", pd.Series(np.nan, index=df.index)), errors="coerce") > -1.78)
        | (df["score_sloan_accruals"] <= 25)
        | (df["accounting_integrity_score"] <= 35)
    )

    # --- 9.5 News & Sentiment Indicator --------------------------------------
    # If real news data is unavailable, estimate the score from adjacent public
    # signals rather than holding every company flat at 50.
    news_available = df.get("news_data_available", pd.Series([False] * len(df)))
    has_news_var = (
        news_available.fillna(False).any()
        and df["news_mentions_30d"].fillna(0).std() > 0
    )
    if has_news_var:
        df["news_sentiment_score"] = (
            0.4 * threshold_score(
                df["news_mentions_30d"],
                [(0, 100), (10, 90), (25, 80), (50, 65), (100, 50), (250, 30), (500, 10)],
            )
            + 0.6 * threshold_score(
                df["risk_news_mentions_30d"],
                [(0, 100), (1, 90), (3, 75), (8, 55), (15, 35), (30, 15), (60, 0)],
            )
        )
        df["news_sentiment_assumption_used"] = False
    else:
        df["news_sentiment_score"] = (
            0.50 * df["market_implied_risk_score"]
            + 0.30 * df["financial_performance_score"]
            + 0.20 * df["sector_quality_score"]
        ).clip(0, 100)
        df["news_sentiment_assumption_used"] = True

    # --- 9.1 Facility & Liquidity Indicator (proxy) --------------------------
    df["facility_liquidity_score"] = (
        0.55 * df["financial_performance_score"]
        + 0.20 * df["score_cash"]
        + 0.15 * df["score_current_ratio"]
        + 0.10 * df["sector_quality_score"]
        + 10
    ).clip(0, 100)
    df["facility_liquidity_risk_proxy"] = 100 - df["facility_liquidity_score"]

    # --- 9.3 Behavioral & Payment Indicator (proxy) --------------------------
    df["behavioral_payment_score"] = (
        0.55 * df["financial_performance_score"]
        + 0.20 * df["accounting_integrity_score"]
        + 0.15 * data_quality
        + 0.10 * df["market_implied_risk_score"]
        + 8
    ).clip(0, 100)
    df["behavioral_payment_risk_proxy"] = 100 - df["behavioral_payment_score"]

    # --- 9.7 Connectivity & Contagion Indicator (proxy) ----------------------
    sector_density = df["sector_group"].map(df["sector_group"].value_counts())
    sector_density_score = threshold_score(
        sector_density,
        [(0, 100), (10, 90), (25, 80), (50, 70), (100, 60), (150, 50), (250, 40)],
    )
    df["sector_density_count"] = sector_density
    df["sector_density_score"] = sector_density_score
    df["connectivity_contagion_score"] = (
        0.45 * df["market_implied_risk_score"]
        + 0.25 * df["sector_quality_score"]
        + 0.20 * df["news_sentiment_score"]
        + 0.10 * data_quality
        + 5
    ).clip(0, 100)
    df["connectivity_contagion_risk_proxy"] = 100 - df["connectivity_contagion_score"]

    # --- 9.8 Governance & Strategic Discipline Indicator (proxy) -------------
    df["governance_discipline_score"] = (
        0.45 * df["accounting_integrity_score"]
        + 0.25 * df["news_sentiment_score"]
        + 0.20 * df["financial_performance_score"]
        + 0.10 * data_quality
        + 5
    ).clip(0, 100)
    df["governance_discipline_risk_proxy"] = 100 - df["governance_discipline_score"]

    # --- 9.9 Collateral & Recovery Indicator (proxy) -------------------------
    df["collateral_recovery_score"] = (
        0.40 * df["score_debt"]
        + 0.25 * df["score_net_debt_ebitda"]
        + 0.20 * df["score_leverage"]
        + 0.15 * df["financial_performance_score"]
        + 5
    ).clip(0, 100)
    df["collateral_recovery_risk_proxy"] = 100 - df["collateral_recovery_score"]

    # --- 11. Composite Surveillance Score ------------------------------------
    # The displayed score is the weighted average of absolute indicator scores.
    # The final score and scoring sub-components use absolute threshold curves;
    # peer-relative views are kept as dashboard diagnostics only.
    df["raw_surveillance_score"] = sum(
        df[ind] * w for ind, w in INDICATOR_WEIGHTS.items()
    ).clip(0, 100)
    df["surveillance_score"] = df["raw_surveillance_score"]
    df["ewif_score"] = df["surveillance_score"]

    # --- 12. Alert Level -----------------------------------------------------
    df["alert_level"] = df["surveillance_score"].apply(assign_alert)

    # --- EWIF v1.0 Section V.C  Tier assignment (composite + velocity) -------
    tiers: List[str] = []
    for _, row in df.iterrows():
        ind_scores = {ind: row.get(ind) for ind in INDICATOR_FIELDS}
        tiers.append(
            assign_tier(
                score=row["surveillance_score"],
                indicator_scores=ind_scores,
                velocity_trigger=bool(row.get("velocity_trigger", False)),
            )
        )
    df["alert_tier"] = tiers

    # --- 13. Top 3 Drivers ---------------------------------------------------
    top_drivers: List[str] = []
    top_driver_lists: List[List[Tuple[str, float]]] = []
    for _, row in df.iterrows():
        scored = sorted(
            ((ind, float(row[ind])) for ind in INDICATOR_FIELDS),
            key=lambda x: x[1],
            reverse=False,
        )[:3]
        top_driver_lists.append(scored)
        top_drivers.append(
            "; ".join(f"{INDICATOR_LABELS[ind]}: {val:.0f}" for ind, val in scored)
        )
    df["top_risk_drivers"] = top_drivers
    df["top_driver_objects"] = top_driver_lists

    df = df.drop(columns=["rating_seed", "rating_bucket", "rating_source"], errors="ignore")

    return df


# ---------------------------------------------------------------------------
# Plain-English explanation (doc Section 16)
# ---------------------------------------------------------------------------


INDICATOR_EXPLANATION_TEMPLATES: Dict[str, str] = {
    "facility_liquidity_score": (
        "Facility & Liquidity is a public-data proxy combining sector base risk "
        "and financial weakness. In production this should be replaced "
        "with internal facility utilization, drawdowns, and covenant data."
    ),
    "financial_performance_score": (
        "Financial Performance is computed from public SEC data using absolute "
        "threshold curves across leverage, net debt / EBITDA, EBITDA / interest "
        "coverage, cash cushion, profitability, free cash flow, FCF sustainability, "
        "and short-term liquidity."
    ),
    "behavioral_payment_score": (
        "Behavioral & Payment is a placeholder proxy using financial weakness as a "
        "stand-in for true payment behavior; in production this should be "
        "driven by servicing data, payment timeliness, and waiver/amendment activity."
    ),
    "market_implied_risk_score": (
        "Market-Implied Risk averages absolute threshold scores for 3-month return, "
        "3-month annualized volatility, and 6-month drawdown. A low score means "
        "markets are pricing more downside or uncertainty."
    ),
    "news_sentiment_score": (
        "News & Sentiment uses actual news counts when available. When news data is "
        "unavailable, it is estimated from market-implied risk, financial performance, "
        "and sector quality so the factor remains directional."
    ),
    "accounting_integrity_score": (
        "Accounting Integrity blends Sloan accruals and the 8-variable Beneish "
        "M-Score where available. If Beneish is unavailable, the score uses Sloan "
        "accruals as the directional public-data proxy."
    ),
    "connectivity_contagion_score": (
        "Connectivity & Contagion is a public-data proxy approximating sector-cluster "
        "risk using sector concentration and current market stress."
    ),
    "governance_discipline_score": (
        "Governance & Strategic Discipline is a proxy combining accounting-integrity "
        "pressure and news pressure. In production it should include "
        "management turnover, insider transactions, and capital allocation signals."
    ),
    "collateral_recovery_score": (
        "Collateral & Recovery is assumption-based in the public demo because lien and "
        "appraisal data are internal. Current scoring uses leverage as a rough public "
        "proxy for recovery risk."
    ),
}


def explanation_for_company(row: pd.Series) -> str:
    """Generate the plain-English explanation paragraph from doc Section 16."""
    name = row["company_name"]
    alert = row["alert_level"]
    score = row["surveillance_score"]
    drivers = row["top_driver_objects"]
    dq = row.get("data_quality_score", float("nan"))
    dq_note = row.get("data_quality_note", "")

    lines: List[str] = [
        f"**{name}** is currently flagged **{alert}** with a Surveillance Score of "
        f"**{score:.1f}**. Higher scores indicate stronger credit quality.",
        "",
        f"The weakest score drivers are **{INDICATOR_LABELS[drivers[0][0]]}** "
        f"({drivers[0][1]:.0f}), **{INDICATOR_LABELS[drivers[1][0]]}** "
        f"({drivers[1][1]:.0f}), and **{INDICATOR_LABELS[drivers[2][0]]}** "
        f"({drivers[2][1]:.0f}).",
        "",
    ]
    for ind, val in drivers:
        lines.append(f"- *{INDICATOR_LABELS[ind]} ({val:.0f}):* "
                     f"{INDICATOR_EXPLANATION_TEMPLATES[ind]}")
    lines.append("")
    lines.append(f"**Data quality:** {dq:.0f} - {dq_note}.")
    return "\n".join(lines)
