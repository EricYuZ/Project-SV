"""
Build structured Surveillance Tool storage.

This script turns the existing CSV artifacts into a SQLite database and creates
quarterly company scores for a complete company-by-quarter grid. Defaults are
the restart baseline requested for this project: 2024Q1 through 2026Q1,
inclusive. Quarterly market metrics are pulled with yfinance when available;
missing financial or market inputs are filled with documented assumptions and
recorded in `assumption_audit`.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "output"
DB_PATH = OUTPUT_DIR / "surveillance_store.sqlite"
QUARTERLY_SCORE_PATH = OUTPUT_DIR / "surveillance_quarterly_scores.csv"
SCORE_START_PERIOD = os.getenv("SCORE_START_PERIOD", "2024Q1")
SCORE_END_PERIOD = os.getenv("SCORE_END_PERIOD", "2026Q1")
RUN_QUARTERLY_MARKET = os.getenv("RUN_QUARTERLY_MARKET", "1") == "1"

sys.path.insert(0, str(PROJECT_DIR / "surveillance_dashboard"))
from scoring import compute_scores  # noqa: E402

sys.path.insert(0, str(BASE_DIR))
from build_surveillance_data import DEFAULT_ASSUMPTIONS, SECTOR_ASSUMPTIONS  # noqa: E402


RATIO_FIELDS = [
    "leverage_proxy",
    "debt_to_assets",
    "cash_to_assets",
    "net_margin",
    "fcf_to_assets",
    "accrual_proxy",
    "current_ratio",
]
INSTANT_FIELDS = [
    "assets",
    "liabilities",
    "cash",
    "current_assets",
    "current_liabilities",
    "total_debt",
    "accounts_receivable",
    "ppe_net",
]
TTM_FIELDS = [
    "revenue",
    "net_income",
    "cfo",
    "capex",
    "fcf",
    "interest_expense",
    "depreciation",
    "cogs",
    "sga",
]
QUARTERLY_MARKET_DEFAULTS = {
    "last_close": np.nan,
    "ret_1m": 0.0,
    "ret_3m": 0.0,
    "vol_3m": 0.25,
    "drawdown_6m": -0.10,
}

ASSUMPTION_FACTOR_MAP = {
    "assets": "Financial Performance",
    "liabilities": "Financial Performance",
    "cash": "Financial Performance",
    "current_assets": "Financial Performance",
    "current_liabilities": "Financial Performance",
    "total_debt": "Financial Performance",
    "revenue": "Financial Performance",
    "net_income": "Financial Performance",
    "cfo": "Financial Performance",
    "capex": "Financial Performance",
    "fcf": "Financial Performance",
    "interest_expense": "Financial Performance",
    "depreciation": "Financial Performance",
    "last_close": "Market-Implied Risk",
    "ret_1m": "Market-Implied Risk",
    "ret_3m": "Market-Implied Risk",
    "vol_3m": "Market-Implied Risk",
    "drawdown_6m": "Market-Implied Risk",
    "news_mentions_30d": "News & Sentiment",
    "risk_news_mentions_30d": "News & Sentiment",
    "news_sentiment": "News & Sentiment",
    "accounting_integrity": "Accounting Integrity",
}


def _note_for_quality(score: float) -> str:
    if score >= 85:
        return "Strong quarterly public-data coverage"
    if score >= 70:
        return "Usable quarterly coverage; some missing fields"
    if score >= 50:
        return "Quarterly assumption-heavy; use with caution"
    return "Weak quarterly coverage; needs manual review or better data source"


def _get_pivot_column(wide: pd.DataFrame, value_type: str, metric: str) -> pd.Series:
    key = (value_type, metric)
    if key in wide.columns:
        return wide[key]
    return pd.Series(np.nan, index=wide.index)


def score_periods() -> pd.PeriodIndex:
    return pd.period_range(SCORE_START_PERIOD, SCORE_END_PERIOD, freq="Q")


def _quarter_end(period: str | pd.Period) -> pd.Timestamp:
    return pd.Period(str(period), freq="Q").end_time.normalize()


def _empty_market_frame(tickers: list[str], periods: pd.PeriodIndex) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        for period in periods:
            rows.append(
                {
                    "ticker": ticker,
                    "score_period": str(period),
                    **{col: val for col, val in QUARTERLY_MARKET_DEFAULTS.items()},
                    **{f"{col}_is_assumption": True for col in QUARTERLY_MARKET_DEFAULTS},
                    "market_data_available": False,
                }
            )
    return pd.DataFrame(rows)


def _metrics_from_close_at(close: pd.Series, period_end: pd.Timestamp) -> dict[str, Any]:
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close[close.index <= period_end]
    if close.empty:
        return {**QUARTERLY_MARKET_DEFAULTS, "market_data_available": False}

    ret = close.pct_change().dropna()
    out: dict[str, Any] = {f: np.nan for f in QUARTERLY_MARKET_DEFAULTS}
    out["last_close"] = float(close.iloc[-1])
    if len(close) > 21:
        out["ret_1m"] = float(close.iloc[-1] / close.iloc[-21] - 1)
    if len(close) > 63:
        out["ret_3m"] = float(close.iloc[-1] / close.iloc[-63] - 1)
    if len(ret) > 20:
        out["vol_3m"] = float(ret.tail(63).std() * np.sqrt(252))
    if len(close) > 126:
        out["drawdown_6m"] = float(close.iloc[-1] / close.tail(126).max() - 1)
    out["market_data_available"] = pd.notna(out["last_close"])
    return out


def fetch_quarterly_market(tickers: list[str], periods: pd.PeriodIndex) -> pd.DataFrame:
    if not RUN_QUARTERLY_MARKET or yf is None or not tickers:
        return _empty_market_frame(tickers, periods)

    start = (_quarter_end(periods[0]) - pd.Timedelta(days=230)).strftime("%Y-%m-%d")
    end = (_quarter_end(periods[-1]) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    try:
        hist = yf.download(
            tickers=" ".join(tickers),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception as exc:
        print(f"WARN: quarterly yfinance download failed: {exc}")
        hist = None

    found: dict[str, pd.Series] = {}
    if hist is not None and not getattr(hist, "empty", True):
        if isinstance(hist.columns, pd.MultiIndex):
            for ticker in tickers:
                if ticker in hist.columns.get_level_values(0):
                    sub = hist[ticker]
                    if "Close" in sub.columns:
                        found[ticker] = sub["Close"]
        elif len(tickers) == 1 and "Close" in hist.columns:
            found[tickers[0]] = hist["Close"]

    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        close = found.get(ticker, pd.Series(dtype=float))
        for period in periods:
            metrics = _metrics_from_close_at(close, _quarter_end(period))
            row = {"ticker": ticker, "score_period": str(period), **metrics}
            for col, default in QUARTERLY_MARKET_DEFAULTS.items():
                row[f"{col}_is_assumption"] = pd.isna(row.get(col))
                if pd.isna(row.get(col)):
                    row[col] = default
            rows.append(row)
    return pd.DataFrame(rows)


def _fill_ratio_assumptions(out: pd.DataFrame) -> pd.DataFrame:
    for col in RATIO_FIELDS:
        out[f"{col}_is_assumption"] = out[col].isna()
        missing = out[col].isna()
        for idx, row in out[missing].iterrows():
            sector = row.get("sector_group", "")
            out.at[idx, col] = SECTOR_ASSUMPTIONS.get(sector, DEFAULT_ASSUMPTIONS).get(
                col, DEFAULT_ASSUMPTIONS[col]
            )
    return out


def _fill_core_financial_assumptions(out: pd.DataFrame) -> pd.DataFrame:
    core_fields = [
        "assets",
        "liabilities",
        "cash",
        "current_assets",
        "current_liabilities",
        "total_debt",
        "revenue",
        "net_income",
        "cfo",
        "capex",
        "fcf",
        "interest_expense",
        "depreciation",
    ]
    for col in core_fields:
        if col not in out.columns:
            out[col] = np.nan
        out[f"{col}_is_assumption"] = out[col].isna()

    for col in core_fields:
        sector_median = out.groupby(["score_period", "sector_group"])[col].transform("median")
        global_median = out.groupby("score_period")[col].transform("median")
        latest_col = f"{col}_latest"
        if latest_col in out.columns:
            out[col] = out[col].combine_first(out[latest_col])
        out[col] = out[col].combine_first(sector_median).combine_first(global_median)

    # Deterministic fallbacks when a whole sector/period is missing a field.
    out["assets"] = out["assets"].fillna(1_000_000_000.0)
    out["liabilities"] = out["liabilities"].fillna(out["leverage_proxy"] * out["assets"])
    out["cash"] = out["cash"].fillna(out["cash_to_assets"] * out["assets"])
    out["total_debt"] = out["total_debt"].fillna(out["debt_to_assets"] * out["assets"])
    out["current_liabilities"] = out["current_liabilities"].fillna(out["liabilities"] * 0.35)
    out["current_assets"] = out["current_assets"].fillna(out["current_ratio"] * out["current_liabilities"])
    out["revenue"] = out["revenue"].fillna(out["assets"] * 0.8)
    out["net_income"] = out["net_income"].fillna(out["net_margin"] * out["revenue"])
    out["fcf"] = out["fcf"].fillna(out["fcf_to_assets"] * out["assets"])
    out["cfo"] = out["cfo"].fillna(out["net_income"] - out["accrual_proxy"] * out["assets"])
    out["capex"] = out["capex"].fillna(out["cfo"] - out["fcf"])
    out["interest_expense"] = out["interest_expense"].fillna((out["total_debt"] * 0.05).clip(lower=0))
    out["depreciation"] = out["depreciation"].fillna((out["assets"] * 0.03).clip(lower=0))
    return out


def build_quarterly_inputs(latest: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    periods = score_periods()
    tickers = latest["ticker"].dropna().astype(str).str.upper().drop_duplicates().tolist()

    history = history.copy()
    history["period_end"] = pd.to_datetime(history["period_end"], errors="coerce")
    history["value"] = pd.to_numeric(history["value"], errors="coerce")
    history = history.dropna(subset=["ticker", "period_end", "value"])
    if not history.empty:
        history["score_period"] = pd.PeriodIndex(history["period_end"], freq="Q").astype(str)
        history = history[history["score_period"].isin([str(p) for p in periods])].copy()

    if history.empty:
        out = pd.DataFrame(columns=["ticker", "period_end", "score_period"] + RATIO_FIELDS + INSTANT_FIELDS + TTM_FIELDS)
    else:
        wide = history.pivot_table(
            index=["ticker", "period_end"],
            columns=["value_type", "metric"],
            values="value",
            aggfunc="last",
        )

        out = pd.DataFrame(index=wide.index)
        for col in RATIO_FIELDS:
            out[col] = _get_pivot_column(wide, "ratio", col)
        for col in INSTANT_FIELDS:
            out[col] = _get_pivot_column(wide, "instant", col)
        for col in TTM_FIELDS:
            out[col] = _get_pivot_column(wide, "ttm", col)

        out = out.reset_index()
        out["score_period"] = pd.PeriodIndex(pd.to_datetime(out["period_end"]), freq="Q").astype(str)
        out = (
            out.sort_values(["ticker", "score_period", "period_end"])
            .drop_duplicates(["ticker", "score_period"], keep="last")
            .reset_index(drop=True)
        )

    grid = pd.MultiIndex.from_product(
        [tickers, [str(p) for p in periods]],
        names=["ticker", "score_period"],
    ).to_frame(index=False)
    out = grid.merge(out, on=["ticker", "score_period"], how="left")
    out["period_end"] = pd.to_datetime(out["period_end"], errors="coerce")
    out["period_end"] = out["period_end"].fillna(out["score_period"].map(_quarter_end))

    identity_cols = [
        c
        for c in ["ticker", "company_name", "sector_group", "industry_group", "index_memberships", "cik", "sec_available"]
        if c in latest.columns
    ]
    identity = latest[identity_cols].drop_duplicates("ticker")
    out = out.merge(identity, on="ticker", how="left")

    latest_financial_cols = [c for c in INSTANT_FIELDS + TTM_FIELDS if c in latest.columns]
    if latest_financial_cols:
        latest_values = latest[["ticker"] + latest_financial_cols].drop_duplicates("ticker")
        latest_values = latest_values.rename(columns={c: f"{c}_latest" for c in latest_financial_cols})
        out = out.merge(latest_values, on="ticker", how="left")

    if "total_debt" in out.columns:
        out["total_debt"] = out["total_debt"].combine_first(out["debt_to_assets"] * out["assets"])
    if "fcf" in out.columns:
        out["fcf"] = out["fcf"].combine_first(out["cfo"] - out["capex"])

    out = _fill_ratio_assumptions(out)
    out = _fill_core_financial_assumptions(out)

    market = fetch_quarterly_market(tickers, periods)
    out = out.merge(market, on=["ticker", "score_period"], how="left")
    for col, val in QUARTERLY_MARKET_DEFAULTS.items():
        flag_col = f"{col}_is_assumption"
        if flag_col not in out.columns:
            out[flag_col] = out[col].isna()
        out[col] = out[col].fillna(val)
        out[flag_col] = out[flag_col].fillna(True)
    out["market_data_available"] = out["market_data_available"].fillna(False)

    out["reference_period_end"] = out["period_end"].dt.strftime("%Y-%m-%d")
    out["period_end"] = out["period_end"].dt.strftime("%Y-%m-%d")
    out["news_mentions_30d"] = 0
    out["risk_news_mentions_30d"] = 0
    out["news_data_available"] = False
    out["news_mentions_30d_is_assumption"] = True
    out["risk_news_mentions_30d_is_assumption"] = True

    assumption_cols = [c for c in out.columns if c.endswith("_is_assumption")]
    out["assumption_metric_count"] = out[assumption_cols].sum(axis=1)
    out["stale_fact_count"] = 0
    out["data_quality_score"] = (
        100.0 - 70.0 * out["assumption_metric_count"] / max(1, len(assumption_cols))
    ).clip(0, 100).round(1)
    out["data_quality_note"] = out["data_quality_score"].apply(_note_for_quality)
    return (
        out.sort_values(["ticker", "score_period"])
        .reset_index(drop=True)
    )


def build_quarterly_scores(quarterly_inputs: pd.DataFrame) -> pd.DataFrame:
    if quarterly_inputs.empty:
        return pd.DataFrame()

    scored_parts = []
    for score_period, part in quarterly_inputs.groupby("score_period", sort=True):
        scored = compute_scores(part.copy(), history_df=None)
        scored["score_period"] = score_period
        scored_parts.append(scored)
    return pd.concat(scored_parts, ignore_index=True)


def _sql_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].map(
                lambda v: json.dumps(v) if isinstance(v, (list, dict, tuple)) else v
            )
    return out.where(pd.notna(out), None)


def _write_table(conn: sqlite3.Connection, name: str, df: pd.DataFrame) -> None:
    _sql_safe(df).to_sql(name, conn, if_exists="replace", index=False)


def build_assumption_audit(scored: pd.DataFrame, period_type: str) -> pd.DataFrame:
    """Create a structured assumption audit table for DB consumers.

    This table intentionally mirrors the management-review proposal: one row per
    company/field assumption with method, source, confidence, and affected score
    area. It includes both data-prep assumptions (`*_is_assumption`) and scoring
    assumptions such as fallback News and Accounting Integrity.
    """
    rows: list[dict[str, Any]] = []
    if scored.empty:
        return pd.DataFrame()

    flag_cols = [c for c in scored.columns if c.endswith("_is_assumption")]
    scoring_flags = [
        c
        for c in ["news_sentiment_assumption_used", "accounting_integrity_assumption_used"]
        if c in scored.columns
    ]

    for _, row in scored.iterrows():
        base = {
            "period_type": period_type,
            "score_period": row.get("score_period"),
            "period_end": row.get("period_end"),
            "ticker": row.get("ticker"),
            "company_name": row.get("company_name"),
            "sector_group": row.get("sector_group"),
            "industry_group": row.get("industry_group"),
            "peer_group_used": row.get("industry_group") or row.get("sector_group"),
            "peer_count": None,
            "assumption_confidence": row.get("data_quality_score"),
        }
        for flag_col in flag_cols:
            if not bool(row.get(flag_col, False)):
                continue
            field_name = flag_col[: -len("_is_assumption")]
            rows.append(
                {
                    **base,
                    "field_name": field_name,
                    "reported_value": None,
                    "final_value_used": row.get(field_name),
                    "assumption_used": True,
                    "assumption_method": "data-prep fallback",
                    "assumption_source": flag_col,
                    "assumption_formula": "Prepared by build_surveillance_data/build_surveillance_store fallback logic.",
                    "affected_factor": ASSUMPTION_FACTOR_MAP.get(field_name, "Data Confidence"),
                    "affected_ratio": field_name,
                }
            )
        for flag_col in scoring_flags:
            if not bool(row.get(flag_col, False)):
                continue
            field_name = flag_col.replace("_assumption_used", "")
            rows.append(
                {
                    **base,
                    "field_name": field_name,
                    "reported_value": None,
                    "final_value_used": row.get(f"{field_name}_score"),
                    "assumption_used": True,
                    "assumption_method": "scoring fallback",
                    "assumption_source": flag_col,
                    "assumption_formula": (
                        "News fallback uses Market, Financial, and Sector quality. "
                        "Accounting fallback uses Financial, Data Quality, and Sloan accruals."
                    ),
                    "affected_factor": ASSUMPTION_FACTOR_MAP.get(field_name, field_name.replace("_", " ").title()),
                    "affected_ratio": f"{field_name}_score",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    latest_path = OUTPUT_DIR / "surveillance_dashboard_input.csv"
    history_path = OUTPUT_DIR / "surveillance_quarterly_history.csv"
    diagnostics_path = OUTPUT_DIR / "surveillance_data_diagnostics.csv"

    latest = pd.read_csv(latest_path)
    history = pd.read_csv(history_path)
    diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path.exists() else pd.DataFrame()

    quarterly_inputs = build_quarterly_inputs(latest, history)
    quarterly_scores = build_quarterly_scores(quarterly_inputs)
    quarterly_scores.to_csv(QUARTERLY_SCORE_PATH, index=False)

    latest_scores = compute_scores(latest, history_df=history)
    latest_assumptions = build_assumption_audit(latest_scores, "latest")
    quarterly_assumptions = build_assumption_audit(quarterly_scores, "quarterly")
    assumption_audit = pd.concat(
        [latest_assumptions, quarterly_assumptions],
        ignore_index=True,
    )
    companies = latest[
        [c for c in ["ticker", "company_name", "sector_group", "industry_group", "index_memberships", "cik", "sec_available"] if c in latest.columns]
    ].drop_duplicates("ticker")

    with sqlite3.connect(DB_PATH) as conn:
        _write_table(conn, "companies", companies)
        _write_table(conn, "latest_inputs", latest)
        _write_table(conn, "latest_scores", latest_scores)
        _write_table(conn, "quarterly_history", history)
        _write_table(conn, "quarterly_inputs", quarterly_inputs)
        _write_table(conn, "quarterly_scores", quarterly_scores)
        _write_table(conn, "data_diagnostics", diagnostics)
        _write_table(conn, "assumption_audit", assumption_audit)

    print(f"Wrote quarterly scores -> {QUARTERLY_SCORE_PATH}")
    print(f"Wrote SQLite store     -> {DB_PATH}")
    print(f"Score period range     -> {SCORE_START_PERIOD} to {SCORE_END_PERIOD}")
    print(f"Quarterly score rows   -> {len(quarterly_scores):,}")
    print(f"Assumption audit rows  -> {len(assumption_audit):,}")
    print(f"Companies in store     -> {companies['ticker'].nunique():,}")


if __name__ == "__main__":
    main()
