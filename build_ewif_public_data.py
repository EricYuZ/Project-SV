"""
EWIF public-data ETL for U.S.-listed company demo universe.

What it does
------------
1) Reads ewif_company_universe_us_100.csv
2) Pulls real public financial statement facts from SEC EDGAR companyfacts API.
3) Optionally pulls market price metrics from yfinance if installed.
4) Optionally pulls public news-volume proxy from GDELT if RUN_GDELT=1.
5) Fills missing dashboard ratios with transparent demo assumptions.
6) Outputs dashboard-ready CSV files.

Setup
-----
python -m pip install pandas requests numpy yfinance

SEC request header
------------------
Set an identifying user agent before running, ideally with your work email:
  Mac/Linux: export SEC_USER_AGENT="Your Name your.email@company.com"
  Windows:   set SEC_USER_AGENT=Your Name your.email@company.com

Run
---
python build_ewif_public_data.py

Outputs
-------
./output/ewif_raw_financials.csv
./output/ewif_dashboard_input.csv
./output/ewif_missing_data_report.csv
./output/ewif_data_quality_summary.csv
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf  # optional
except Exception:  # pragma: no cover
    yf = None

BASE_DIR = Path(__file__).resolve().parent
UNIVERSE_FILE = BASE_DIR / "ewif_company_universe_us_100.csv"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = BASE_DIR / "cache_sec"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "EWIF demo research contact@example.com")
SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}
SEC_HEADERS_FILES = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

# Keep SEC requests comfortably under fair-access limits.
SEC_SLEEP_SECONDS = float(os.getenv("SEC_SLEEP_SECONDS", "0.15"))
RUN_MARKET = os.getenv("RUN_MARKET", "1") == "1"
RUN_GDELT = os.getenv("RUN_GDELT", "0") == "1"  # set to 1 if you want news counts

# US-GAAP tags by dashboard field. Tags are intentionally broad because companies label XBRL items differently.
TAG_MAP = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
        "InterestAndDividendIncomeOperating",
        "InterestIncomeExpenseNonOperatingNet",
        "NoninterestIncome",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "PartnersCapital",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    "cfo": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets",
    ],
    "interest_expense": [
        "InterestExpenseNonOperating",
        "InterestExpense",
        "InterestAndDebtExpense",
    ],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "accounts_receivable": [
        "AccountsReceivableNetCurrent",
        "AccountsNotesAndLoansReceivableNetCurrent",
        "ReceivablesNetCurrent",
    ],
    "inventory": ["InventoryNet", "InventoryFinishedGoodsNetOfReserves"],
    "accounts_payable": ["AccountsPayableCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent"],
    "long_term_debt": [
        "LongTermDebt",
        "LongTermDebtAndFinanceLeaseObligations",
        "LongTermDebtNoncurrent",
    ],
    "short_term_debt": [
        "ShortTermBorrowings",
        "ShortTermDebt",
        "CurrentPortionOfLongTermDebt",
        "LongTermDebtCurrent",
    ],
}

# Conservative demo assumptions used only for missing scoring ratios.
# These are not replacements for real financial data; they keep the dashboard running when SEC tags are unavailable.
SECTOR_ASSUMPTIONS = {
    "Financials": {"leverage_proxy": 0.88, "debt_to_assets": 0.25, "cash_to_assets": 0.08, "net_margin": 0.18, "fcf_to_assets": 0.015, "accrual_proxy": 0.00, "current_ratio": 1.00},
    "Technology": {"leverage_proxy": 0.55, "debt_to_assets": 0.18, "cash_to_assets": 0.12, "net_margin": 0.18, "fcf_to_assets": 0.08, "accrual_proxy": 0.02, "current_ratio": 1.50},
    "Technology & Communications": {"leverage_proxy": 0.70, "debt_to_assets": 0.38, "cash_to_assets": 0.04, "net_margin": 0.10, "fcf_to_assets": 0.04, "accrual_proxy": 0.02, "current_ratio": 0.90},
    "Utilities": {"leverage_proxy": 0.74, "debt_to_assets": 0.45, "cash_to_assets": 0.02, "net_margin": 0.10, "fcf_to_assets": -0.01, "accrual_proxy": 0.02, "current_ratio": 0.80},
    "Energy": {"leverage_proxy": 0.48, "debt_to_assets": 0.22, "cash_to_assets": 0.05, "net_margin": 0.12, "fcf_to_assets": 0.05, "accrual_proxy": 0.01, "current_ratio": 1.20},
    "Transportation": {"leverage_proxy": 0.78, "debt_to_assets": 0.42, "cash_to_assets": 0.07, "net_margin": 0.06, "fcf_to_assets": 0.01, "accrual_proxy": 0.02, "current_ratio": 0.85},
    "Autos": {"leverage_proxy": 0.78, "debt_to_assets": 0.38, "cash_to_assets": 0.08, "net_margin": 0.05, "fcf_to_assets": 0.02, "accrual_proxy": 0.01, "current_ratio": 1.15},
    "Aerospace & Defense": {"leverage_proxy": 0.78, "debt_to_assets": 0.30, "cash_to_assets": 0.06, "net_margin": 0.07, "fcf_to_assets": 0.04, "accrual_proxy": 0.03, "current_ratio": 1.10},
    "Industrials": {"leverage_proxy": 0.66, "debt_to_assets": 0.28, "cash_to_assets": 0.06, "net_margin": 0.09, "fcf_to_assets": 0.04, "accrual_proxy": 0.02, "current_ratio": 1.20},
    "Consumer Staples": {"leverage_proxy": 0.70, "debt_to_assets": 0.30, "cash_to_assets": 0.04, "net_margin": 0.10, "fcf_to_assets": 0.05, "accrual_proxy": 0.02, "current_ratio": 0.95},
    "Consumer Discretionary": {"leverage_proxy": 0.72, "debt_to_assets": 0.32, "cash_to_assets": 0.05, "net_margin": 0.08, "fcf_to_assets": 0.03, "accrual_proxy": 0.02, "current_ratio": 1.05},
    "Communication Services": {"leverage_proxy": 0.62, "debt_to_assets": 0.28, "cash_to_assets": 0.08, "net_margin": 0.13, "fcf_to_assets": 0.05, "accrual_proxy": 0.02, "current_ratio": 1.10},
    "Healthcare": {"leverage_proxy": 0.65, "debt_to_assets": 0.30, "cash_to_assets": 0.06, "net_margin": 0.12, "fcf_to_assets": 0.05, "accrual_proxy": 0.02, "current_ratio": 1.10},
    "Real Estate": {"leverage_proxy": 0.70, "debt_to_assets": 0.45, "cash_to_assets": 0.03, "net_margin": 0.08, "fcf_to_assets": 0.02, "accrual_proxy": 0.02, "current_ratio": 0.90},
}
DEFAULT_ASSUMPTIONS = {"leverage_proxy": 0.65, "debt_to_assets": 0.30, "cash_to_assets": 0.06, "net_margin": 0.08, "fcf_to_assets": 0.03, "accrual_proxy": 0.02, "current_ratio": 1.00}

RATING_RISK_BIAS = {
    "AAA": -10, "AA": -8, "AA-": -7, "A+": -5, "A": -4, "A-": -3,
    "BBB+": 0, "BBB": 2, "BBB-": 5,
    "BB+": 10, "BB": 13, "BB-": 16,
    "B+": 22, "B": 25, "B-": 30,
    "Caa2": 40, "NR": 10,
}


def fetch_json(url: str, headers: Optional[Dict[str, str]] = None, cache_path: Optional[Path] = None) -> Dict[str, Any]:
    if cache_path and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    resp = requests.get(url, headers=headers or SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if cache_path:
        cache_path.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(SEC_SLEEP_SECONDS)
    return data


def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_FILE)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df


def get_sec_ticker_map() -> pd.DataFrame:
    url = "https://www.sec.gov/files/company_tickers.json"
    data = fetch_json(url, headers=SEC_HEADERS_FILES, cache_path=CACHE_DIR / "company_tickers.json")
    rows = []
    for _, v in data.items():
        rows.append({
            "ticker": str(v["ticker"]).upper(),
            "sec_title": v["title"],
            "cik": str(v["cik_str"]).zfill(10),
        })
    return pd.DataFrame(rows)


def get_companyfacts(cik: str) -> Optional[Dict[str, Any]]:
    cache_path = CACHE_DIR / f"CIK{cik}_companyfacts.json"
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        return fetch_json(url, headers=SEC_HEADERS, cache_path=cache_path)
    except Exception as exc:
        print(f"WARN: companyfacts failed for CIK {cik}: {exc}")
        return None


def select_latest_fact(facts: Dict[str, Any], tags: Iterable[str], unit: str = "USD", prefer_form: str = "10-K") -> Tuple[float, str, str, str, str]:
    """Return value, tag, form, end, filed. Returns NaNs/empty strings if missing."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {}) if facts else {}
    candidates = []
    for tag in tags:
        node = us_gaap.get(tag, {})
        units = node.get("units", {})
        records = units.get(unit, [])
        if not records and unit == "USD":
            # Some companies report pure ratios/shares under other units, but keep financial fields USD-only.
            continue
        for rec in records:
            if rec.get("val") is None:
                continue
            form = rec.get("form", "")
            if form not in {"10-K", "10-Q", "20-F", "40-F"}:
                continue
            end = rec.get("end", "")
            filed = rec.get("filed", "")
            # Keep only normal, filed facts. Avoid amended facts ordering issues by using latest filed/end.
            candidates.append({"val": rec.get("val"), "tag": tag, "form": form, "end": end, "filed": filed})
    if not candidates:
        return (np.nan, "", "", "", "")
    preferred = [c for c in candidates if c["form"] == prefer_form]
    if preferred:
        candidates = preferred
    candidates.sort(key=lambda x: (x.get("end") or "", x.get("filed") or ""), reverse=True)
    c = candidates[0]
    return (float(c["val"]), c["tag"], c["form"], c["end"], c["filed"])


def safe_div(num, den):
    try:
        if den is None or pd.isna(den) or den == 0:
            return np.nan
        if num is None or pd.isna(num):
            return np.nan
        return num / den
    except Exception:
        return np.nan


def parse_rating_bucket(text: str) -> str:
    text = str(text)
    candidates = ["Caa2", "AAA", "AA-", "AA", "A+", "A-", "A", "BBB+", "BBB-", "BBB", "BB+", "BB-", "BB", "B+", "B-", "B", "NR"]
    for c in candidates:
        if c in text:
            return c
    return "NR"


def financials_for_company(row: pd.Series, cik: Optional[str]) -> Dict[str, Any]:
    result = {
        "ticker": row["ticker"],
        "company_name": row["company_name"],
        "sector_group": row["sector_group"],
        "industry_group": row["industry_group"],
        "rating_seed": row["rating_seed"],
        "rating_bucket": parse_rating_bucket(row["rating_seed"]),
        "rating_source": row["rating_source"],
        "cik": cik,
        "sec_available": bool(cik),
    }
    if not cik:
        return result
    facts = get_companyfacts(cik)
    if not facts:
        result["sec_available"] = False
        return result

    for field, tags in TAG_MAP.items():
        value, tag, form, end, filed = select_latest_fact(facts, tags, unit="USD", prefer_form="10-K")
        result[field] = value
        result[f"{field}_tag"] = tag
        result[f"{field}_form"] = form
        result[f"{field}_period_end"] = end
        result[f"{field}_filed"] = filed

    # Derived values
    result["total_debt"] = np.nansum([result.get("long_term_debt"), result.get("short_term_debt")])
    if pd.isna(result.get("long_term_debt")) and pd.isna(result.get("short_term_debt")):
        result["total_debt"] = np.nan
    result["fcf"] = result.get("cfo") - result.get("capex") if pd.notna(result.get("cfo")) and pd.notna(result.get("capex")) else np.nan
    result["leverage_proxy"] = safe_div(result.get("liabilities"), result.get("assets"))
    result["debt_to_assets"] = safe_div(result.get("total_debt"), result.get("assets"))
    result["cash_to_assets"] = safe_div(result.get("cash"), result.get("assets"))
    result["net_margin"] = safe_div(result.get("net_income"), result.get("revenue"))
    result["fcf_to_assets"] = safe_div(result.get("fcf"), result.get("assets"))
    result["accrual_proxy"] = safe_div((result.get("net_income") - result.get("cfo")) if pd.notna(result.get("net_income")) and pd.notna(result.get("cfo")) else np.nan, result.get("assets"))
    result["current_ratio"] = safe_div(result.get("current_assets"), result.get("current_liabilities"))
    result["receivables_to_revenue"] = safe_div(result.get("accounts_receivable"), result.get("revenue"))
    result["inventory_to_revenue"] = safe_div(result.get("inventory"), result.get("revenue"))
    result["ap_to_revenue"] = safe_div(result.get("accounts_payable"), result.get("revenue"))
    return result


def market_metrics(ticker: str) -> Dict[str, Any]:
    result = {"ticker": ticker, "market_data_available": False}
    if not RUN_MARKET or yf is None:
        return result
    try:
        hist = yf.download(ticker, period="1y", auto_adjust=True, progress=False, threads=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return result
        close = hist["Close"].dropna()
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        ret = close.pct_change().dropna()
        result["market_data_available"] = True
        result["last_close"] = float(close.iloc[-1])
        result["ret_1m"] = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 21 else np.nan
        result["ret_3m"] = float(close.iloc[-1] / close.iloc[-63] - 1) if len(close) > 63 else np.nan
        result["vol_3m"] = float(ret.tail(63).std() * math.sqrt(252)) if len(ret) > 20 else np.nan
        result["drawdown_6m"] = float(close.iloc[-1] / close.tail(126).max() - 1) if len(close) > 126 else np.nan
    except Exception as exc:
        print(f"WARN: market data failed for {ticker}: {exc}")
    return result


def gdelt_mentions(company_name: str, days: int = 30) -> Dict[str, Any]:
    result = {"company_name": company_name, "gdelt_available": False}
    if not RUN_GDELT:
        return result
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    def count(query: str) -> float:
        params = (
            f"query={quote(query)}&mode=timelinevol&format=json"
            f"&startdatetime={start.strftime('%Y%m%d%H%M%S')}"
            f"&enddatetime={end.strftime('%Y%m%d%H%M%S')}"
        )
        try:
            data = requests.get(f"{base}?{params}", timeout=25).json()
            timeline = data.get("timeline", [])
            return float(sum(x.get("value", 0) for x in timeline))
        except Exception:
            return np.nan
    clean_name = company_name.replace(", Inc.", "").replace(" Inc.", "").replace(" Corp", "").replace(" Corporation", "")
    all_mentions = count(f'"{clean_name}"')
    risk_mentions = count(f'"{clean_name}" (bankruptcy OR restructuring OR lawsuit OR investigation OR default OR downgrade OR delinquency OR liquidity OR fraud)')
    result.update({"gdelt_available": True, "news_mentions_30d": all_mentions, "risk_news_mentions_30d": risk_mentions})
    return result


def fill_assumptions(df: pd.DataFrame) -> pd.DataFrame:
    ratio_cols = ["leverage_proxy", "debt_to_assets", "cash_to_assets", "net_margin", "fcf_to_assets", "accrual_proxy", "current_ratio"]
    for col in ratio_cols:
        df[f"{col}_is_assumption"] = df[col].isna()
        for idx, row in df[df[col].isna()].iterrows():
            sector = row.get("sector_group", "")
            val = SECTOR_ASSUMPTIONS.get(sector, DEFAULT_ASSUMPTIONS).get(col, DEFAULT_ASSUMPTIONS[col])
            df.at[idx, col] = val
    # Market assumptions for optional/unavailable price metrics
    market_defaults = {"ret_1m": 0.0, "ret_3m": 0.0, "vol_3m": 0.25, "drawdown_6m": -0.10}
    for col, val in market_defaults.items():
        if col not in df.columns:
            df[col] = np.nan
        df[f"{col}_is_assumption"] = df[col].isna()
        df[col] = df[col].fillna(val)
    # News assumptions
    for col in ["news_mentions_30d", "risk_news_mentions_30d"]:
        if col not in df.columns:
            df[col] = np.nan
        df[f"{col}_is_assumption"] = df[col].isna()
        df[col] = df[col].fillna(0.0)
    return df


def percentile_score(series: pd.Series, higher_is_risk: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    pct = s.rank(pct=True, method="average") * 100
    if not higher_is_risk:
        pct = 100 - pct
    return pct.fillna(50).clip(0, 100)


def score_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    # Financial factor: 100 means higher credit deterioration risk.
    df["score_leverage"] = percentile_score(df["leverage_proxy"], True)
    df["score_debt"] = percentile_score(df["debt_to_assets"], True)
    df["score_cash"] = percentile_score(df["cash_to_assets"], False)
    df["score_margin"] = percentile_score(df["net_margin"], False)
    df["score_fcf"] = percentile_score(df["fcf_to_assets"], False)
    df["score_current_ratio"] = percentile_score(df["current_ratio"], False)
    df["financial_performance_score"] = df[["score_leverage", "score_debt", "score_cash", "score_margin", "score_fcf", "score_current_ratio"]].mean(axis=1)

    # Market factor
    df["score_negative_momentum"] = percentile_score(-df["ret_3m"], True)
    df["score_volatility"] = percentile_score(df["vol_3m"], True)
    df["score_drawdown"] = percentile_score(-df["drawdown_6m"], True)
    df["market_implied_risk_score"] = df[["score_negative_momentum", "score_volatility", "score_drawdown"]].mean(axis=1)

    # News factor
    df["news_sentiment_score"] = (
        0.4 * percentile_score(df["news_mentions_30d"], True) +
        0.6 * percentile_score(df["risk_news_mentions_30d"], True)
    )

    # Accounting factor
    df["accounting_integrity_score"] = percentile_score(df["accrual_proxy"], True)

    # Internal-bank-only factors are placeholder assumptions with rating/sector bias.
    rating_bias = df["rating_bucket"].map(RATING_RISK_BIAS).fillna(10)
    sector_base = df["sector_group"].map({
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
    }).fillna(45)

    # These are not pretending to be real facility/payment/collateral data.
    df["facility_liquidity_score"] = (sector_base + rating_bias + 0.25 * df["financial_performance_score"] - 10).clip(0, 100)
    df["behavioral_payment_score"] = (35 + rating_bias + 0.20 * df["financial_performance_score"]).clip(0, 100)
    df["connectivity_contagion_score"] = (35 + 0.4 * percentile_score(df["sector_group"].map(df["sector_group"].value_counts()), True) + 0.10 * df["market_implied_risk_score"]).clip(0, 100)
    df["governance_discipline_score"] = (35 + rating_bias + 0.15 * df["accounting_integrity_score"] + 0.10 * df["news_sentiment_score"]).clip(0, 100)
    df["collateral_recovery_score"] = (30 + rating_bias + 0.25 * df["leverage_proxy"] * 100).clip(0, 100)

    df["ewif_score"] = (
        0.17 * df["facility_liquidity_score"] +
        0.17 * df["financial_performance_score"] +
        0.08 * df["behavioral_payment_score"] +
        0.10 * df["market_implied_risk_score"] +
        0.10 * df["news_sentiment_score"] +
        0.15 * df["accounting_integrity_score"] +
        0.10 * df["connectivity_contagion_score"] +
        0.10 * df["governance_discipline_score"] +
        0.05 * df["collateral_recovery_score"]
    ).clip(0, 100)

    df["alert_level"] = pd.cut(
        df["ewif_score"], bins=[-0.01, 35, 50, 70, 100], labels=["Green", "Yellow", "Orange", "Red"]
    ).astype(str)

    factor_cols = [
        "facility_liquidity_score", "financial_performance_score", "behavioral_payment_score",
        "market_implied_risk_score", "news_sentiment_score", "accounting_integrity_score",
        "connectivity_contagion_score", "governance_discipline_score", "collateral_recovery_score"
    ]
    top_factors = []
    for _, row in df.iterrows():
        sorted_f = sorted([(f, row[f]) for f in factor_cols], key=lambda x: x[1], reverse=True)
        top_factors.append("; ".join([f"{name.replace('_score','').replace('_',' ').title()}: {val:.0f}" for name, val in sorted_f[:3]]))
    df["top_risk_drivers"] = top_factors

    assumption_cols = [c for c in df.columns if c.endswith("_is_assumption")]
    df["assumption_metric_count"] = df[assumption_cols].sum(axis=1) if assumption_cols else 0
    df["data_quality_score"] = (100 - df["assumption_metric_count"] / max(1, len(assumption_cols)) * 100).round(1)
    return df


def main() -> None:
    universe = load_universe()
    sec_map = get_sec_ticker_map()
    universe = universe.merge(sec_map[["ticker", "cik", "sec_title"]], on="ticker", how="left")

    rows = []
    for _, row in universe.iterrows():
        print(f"SEC: {row['ticker']} {row['company_name']}")
        rows.append(financials_for_company(row, row.get("cik") if pd.notna(row.get("cik")) else None))
    financials = pd.DataFrame(rows)

    if RUN_MARKET:
        market_rows = []
        for ticker in universe["ticker"]:
            print(f"Market: {ticker}")
            market_rows.append(market_metrics(ticker))
        market = pd.DataFrame(market_rows)
        financials = financials.merge(market, on="ticker", how="left")

    if RUN_GDELT:
        news_rows = []
        for company in universe["company_name"]:
            print(f"GDELT: {company}")
            news_rows.append(gdelt_mentions(company))
            time.sleep(0.15)
        news = pd.DataFrame(news_rows)
        financials = financials.merge(news, on="company_name", how="left")

    financials.to_csv(OUTPUT_DIR / "ewif_raw_financials.csv", index=False)

    dash = fill_assumptions(financials.copy())
    dash = score_dashboard(dash)

    # Keep dashboard table concise but with enough auditability.
    dashboard_cols = [
        "ticker", "company_name", "sector_group", "industry_group", "rating_seed", "rating_bucket", "rating_source", "cik", "sec_available",
        "revenue", "net_income", "assets", "liabilities", "cash", "cfo", "capex", "fcf", "total_debt",
        "leverage_proxy", "debt_to_assets", "cash_to_assets", "net_margin", "fcf_to_assets", "accrual_proxy", "current_ratio",
        "last_close", "ret_1m", "ret_3m", "vol_3m", "drawdown_6m", "news_mentions_30d", "risk_news_mentions_30d",
        "facility_liquidity_score", "financial_performance_score", "behavioral_payment_score", "market_implied_risk_score", "news_sentiment_score",
        "accounting_integrity_score", "connectivity_contagion_score", "governance_discipline_score", "collateral_recovery_score",
        "ewif_score", "alert_level", "top_risk_drivers", "assumption_metric_count", "data_quality_score",
    ]
    for col in dashboard_cols:
        if col not in dash.columns:
            dash[col] = np.nan
    dash[dashboard_cols].to_csv(OUTPUT_DIR / "ewif_dashboard_input.csv", index=False)

    assumption_cols = [c for c in dash.columns if c.endswith("_is_assumption")]
    missing = dash[["ticker", "company_name", "sector_group", "rating_seed", "sec_available"] + assumption_cols].copy()
    missing.to_csv(OUTPUT_DIR / "ewif_missing_data_report.csv", index=False)

    summary = dash.groupby(["sector_group", "alert_level"], dropna=False).agg(
        company_count=("ticker", "count"),
        avg_ewif_score=("ewif_score", "mean"),
        avg_data_quality=("data_quality_score", "mean"),
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / "ewif_data_quality_summary.csv", index=False)

    print("Done. Files written to:")
    for f in ["ewif_raw_financials.csv", "ewif_dashboard_input.csv", "ewif_missing_data_report.csv", "ewif_data_quality_summary.csv"]:
        print(" -", OUTPUT_DIR / f)


if __name__ == "__main__":
    main()
