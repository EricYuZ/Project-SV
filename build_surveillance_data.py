"""
Surveillance Tool — Public Data Preparation Pipeline
=====================================================

Pulls public financial, market, and news data for the 100-company demo
universe and prepares dashboard-ready CSVs. Scoring (the 9 indicators,
composite Surveillance Score, alert level, top drivers) is intentionally
NOT done here — that lives in the dashboard layer so weights, thresholds,
and peer groups can be tuned without re-running the data pull.

What this script does
---------------------
1. Reads the major-index universe CSV (S&P 500 + Nasdaq 100 + Dow)
2. Pulls SEC EDGAR companyfacts for each company (cached on disk)
3. Pulls market price metrics from yfinance (optional, ON by default)
4. Pulls GDELT 30-day news + risk-news mention counts (ON by default)
5. Flags stale facts using the 550-day rule (per documentation 7.2)
6. Fills missing/stale ratios with sector-level assumptions and flags them
7. Computes data_quality_score (70% assumptions + 30% stale) and a note
8. Writes 3 CSVs prefixed `surveillance_*` for the dashboard to consume

Setup
-----
python -m pip install pandas requests numpy yfinance
export SEC_USER_AGENT="Your Name your.email@company.com"

Run
---
python build_surveillance_data.py

Env vars
--------
SEC_USER_AGENT       Required by SEC fair-use policy
RUN_MARKET           "1" (default) to pull yfinance market data
RUN_GDELT            "0" (default) to skip GDELT. The API is slow/empty in the
                     current run, so per documentation 6.3 the dashboard holds
                     news at neutral 50. Set to "1" only when GDELT is healthy.
STALE_DAYS           "550" (default) — fact is stale if older than this
SEC_SLEEP_SECONDS    "0.15" (default) between SEC requests
GDELT_SLEEP_SECONDS  "0.5" (default) between GDELT requests

Outputs (in ./output/)
----------------------
surveillance_raw_financials.csv     Raw SEC facts with tag/form/period_end/filed
surveillance_dashboard_input.csv    Clean per-company row for the dashboard
surveillance_data_diagnostics.csv   Long-format flag detail for the data-quality page
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except Exception:
    yf = None


BASE_DIR = Path(__file__).resolve().parent
# Default to the major-index universe; override with UNIVERSE_FILE env var.
DEFAULT_UNIVERSE = BASE_DIR / "ewif_company_universe_major_indices.csv"
SP500_UNIVERSE = BASE_DIR / "ewif_company_universe_sp500.csv"
LEGACY_UNIVERSE = BASE_DIR / "ewif_company_universe_us_100.csv"
UNIVERSE_FILE = Path(
    os.getenv("UNIVERSE_FILE",
              str(
                  DEFAULT_UNIVERSE
                  if DEFAULT_UNIVERSE.exists()
                  else SP500_UNIVERSE
                  if SP500_UNIVERSE.exists()
                  else LEGACY_UNIVERSE
              ))
)
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = BASE_DIR / "cache_sec"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT", "Surveillance demo research contact@example.com"
)
SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}
SEC_HEADERS_FILES = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

SEC_SLEEP_SECONDS = float(os.getenv("SEC_SLEEP_SECONDS", "0.15"))
GDELT_SLEEP_SECONDS = float(os.getenv("GDELT_SLEEP_SECONDS", "0.15"))
GDELT_TIMEOUT = float(os.getenv("GDELT_TIMEOUT", "8"))
RUN_MARKET = os.getenv("RUN_MARKET", "1") == "1"
RUN_GDELT = os.getenv("RUN_GDELT", "0") == "1"
STALE_DAYS = int(os.getenv("STALE_DAYS", "550"))


def log(msg: str) -> None:
    """Print with immediate flush so progress is visible during long pulls."""
    print(msg, flush=True)


# US-GAAP tags by dashboard field. Tags are intentionally broad because companies
# report XBRL items under different tag names.
TAG_MAP: Dict[str, List[str]] = {
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
    # Added for Beneish M-Score (EWIF v1.0 Section VI.B):
    # GMI needs gross margin -> needs COGS or CostOfRevenue.
    # AQI / DEPI need PP&E (instant) and Depreciation (duration).
    # SGAI needs Selling, General & Administrative.
    "cogs": [
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfRevenue",
        "CostOfServices",
    ],
    "ppe_net": [
        "PropertyPlantAndEquipmentNet",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
    ],
    "depreciation": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "sga": [
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
        "SellingAndMarketingExpense",
    ],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "accounts_receivable": [
        "AccountsReceivableNetCurrent",
        "AccountsNotesAndLoansReceivableNetCurrent",
        "ReceivablesNetCurrent",
    ],
    "inventory": ["InventoryNet", "InventoryFinishedGoodsNetOfReserves"],
    "accounts_payable": [
        "AccountsPayableCurrent",
        "AccountsPayableAndAccruedLiabilitiesCurrent",
    ],
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

SEC_FIELDS = list(TAG_MAP.keys())

# Ratios the dashboard expects to be present in surveillance_dashboard_input.csv.
# If a ratio cannot be computed from raw SEC data (or its inputs are stale),
# we substitute the sector-level assumption below and flag it.
RATIO_FIELDS = [
    "leverage_proxy",
    "debt_to_assets",
    "cash_to_assets",
    "net_margin",
    "fcf_to_assets",
    "accrual_proxy",
    "current_ratio",
]

MARKET_FIELDS = ["last_close", "ret_1m", "ret_3m", "vol_3m", "drawdown_6m"]
NEWS_FIELDS = ["news_mentions_30d", "risk_news_mentions_30d"]


# Sector-level fallbacks used only to keep the dashboard scorable when a
# ratio cannot be computed from public SEC data. Each filled value is flagged
# via <ratio>_is_assumption so the dashboard layer can mark it transparently.
SECTOR_ASSUMPTIONS: Dict[str, Dict[str, float]] = {
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
    "Materials": {"leverage_proxy": 0.60, "debt_to_assets": 0.30, "cash_to_assets": 0.05, "net_margin": 0.08, "fcf_to_assets": 0.04, "accrual_proxy": 0.02, "current_ratio": 1.40},
}
DEFAULT_ASSUMPTIONS = {
    "leverage_proxy": 0.65, "debt_to_assets": 0.30, "cash_to_assets": 0.06,
    "net_margin": 0.08, "fcf_to_assets": 0.03, "accrual_proxy": 0.02, "current_ratio": 1.00,
}

# Neutral defaults used only when yfinance is unavailable for a ticker.
MARKET_DEFAULTS = {"ret_1m": 0.0, "ret_3m": 0.0, "vol_3m": 0.25, "drawdown_6m": -0.10}


# -----------------------------------------------------------------------------
# HTTP helpers
# -----------------------------------------------------------------------------


def fetch_json(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    cache_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if cache_path and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    resp = requests.get(url, headers=headers or SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if cache_path:
        cache_path.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(SEC_SLEEP_SECONDS)
    return data


# -----------------------------------------------------------------------------
# Universe + CIK mapping
# -----------------------------------------------------------------------------


def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_FILE)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df


def get_sec_ticker_map() -> pd.DataFrame:
    url = "https://www.sec.gov/files/company_tickers.json"
    data = fetch_json(
        url,
        headers=SEC_HEADERS_FILES,
        cache_path=CACHE_DIR / "company_tickers.json",
    )
    rows = []
    for _, v in data.items():
        rows.append(
            {
                "ticker": str(v["ticker"]).upper(),
                "sec_title": v["title"],
                "cik": str(v["cik_str"]).zfill(10),
            }
        )
    return pd.DataFrame(rows)


def get_companyfacts(cik: str) -> Optional[Dict[str, Any]]:
    cache_path = CACHE_DIR / f"CIK{cik}_companyfacts.json"
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        return fetch_json(url, headers=SEC_HEADERS, cache_path=cache_path)
    except Exception as exc:
        log(f"  WARN: companyfacts failed for CIK {cik}: {exc}")
        return None


# -----------------------------------------------------------------------------
# SEC fact selection
# -----------------------------------------------------------------------------


def select_latest_fact(
    facts: Dict[str, Any],
    tags: Iterable[str],
    unit: str = "USD",
    prefer_form: str = "10-K",
) -> Tuple[float, str, str, str, str]:
    """Return (value, tag, form, period_end, filed) for the latest matching fact.

    Returns NaN/empty strings if no usable fact is found.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {}) if facts else {}
    candidates = []
    for tag in tags:
        node = us_gaap.get(tag, {})
        units = node.get("units", {})
        records = units.get(unit, [])
        if not records:
            continue
        for rec in records:
            if rec.get("val") is None:
                continue
            form = rec.get("form", "")
            if form not in {"10-K", "10-Q", "20-F", "40-F"}:
                continue
            candidates.append(
                {
                    "val": rec.get("val"),
                    "tag": tag,
                    "form": form,
                    "end": rec.get("end", ""),
                    "filed": rec.get("filed", ""),
                }
            )
    if not candidates:
        return (np.nan, "", "", "", "")
    preferred = [c for c in candidates if c["form"] == prefer_form]
    if preferred:
        candidates = preferred
    candidates.sort(
        key=lambda x: (x.get("end") or "", x.get("filed") or ""), reverse=True
    )
    c = candidates[0]
    return (float(c["val"]), c["tag"], c["form"], c["end"], c["filed"])


def parse_rating_bucket(text: str) -> str:
    text = str(text)
    candidates = [
        "Caa2", "AAA", "AA-", "AA", "A+", "A-", "A",
        "BBB+", "BBB-", "BBB", "BB+", "BB-", "BB",
        "B+", "B-", "B", "NR",
    ]
    for c in candidates:
        if c in text:
            return c
    return "NR"


def derive_synthetic_rating(
    debt_to_assets: Optional[float],
    leverage_proxy: Optional[float],
    net_margin: Optional[float],
    fcf_to_assets: Optional[float],
) -> str:
    """Bucket an unrated ticker into an investment-grade-style rating using
    leverage + profitability heuristics. NOT a substitute for a real S&P /
    Moody's rating; flagged in `rating_source` so the dashboard can surface
    that this is derived, not vendor-sourced."""
    if debt_to_assets is None or pd.isna(debt_to_assets):
        return "NR"

    score = 0.0
    score += float(debt_to_assets) * 50
    if leverage_proxy is not None and pd.notna(leverage_proxy):
        score += max(0.0, float(leverage_proxy) - 0.5) * 30
    if net_margin is not None and pd.notna(net_margin):
        score -= float(net_margin) * 20
    if fcf_to_assets is not None and pd.notna(fcf_to_assets):
        score -= float(fcf_to_assets) * 25

    if score < 5:
        return "A"
    if score < 8:
        return "A-"
    if score < 11:
        return "BBB+"
    if score < 14:
        return "BBB"
    if score < 17:
        return "BBB-"
    if score < 20:
        return "BB+"
    if score < 24:
        return "BB"
    if score < 28:
        return "BB-"
    if score < 33:
        return "B+"
    if score < 38:
        return "B"
    return "B-"


def safe_div(num, den):
    try:
        if den is None or pd.isna(den) or den == 0:
            return np.nan
        if num is None or pd.isna(num):
            return np.nan
        return num / den
    except Exception:
        return np.nan


# -----------------------------------------------------------------------------
# Per-company SEC pull
# -----------------------------------------------------------------------------


def financials_for_company(row: pd.Series, cik: Optional[str]) -> Dict[str, Any]:
    """Pull one company's SEC facts. Returns a flat dict with raw values plus
    `<field>_tag`, `<field>_form`, `<field>_period_end`, `<field>_filed`
    columns for every field in TAG_MAP."""
    result: Dict[str, Any] = {
        "ticker": row["ticker"],
        "company_name": row["company_name"],
        "sector_group": row["sector_group"],
        "industry_group": row["industry_group"],
        "index_memberships": row.get("index_memberships", ""),
        "rating_seed": row["rating_seed"],
        "rating_bucket": parse_rating_bucket(row["rating_seed"]),
        "rating_source": row["rating_source"],
        "cik": cik,
        "sec_available": bool(cik),
    }
    for field in SEC_FIELDS:
        result[field] = np.nan
        result[f"{field}_tag"] = ""
        result[f"{field}_form"] = ""
        result[f"{field}_period_end"] = ""
        result[f"{field}_filed"] = ""

    if not cik:
        return result

    facts = get_companyfacts(cik)
    if not facts:
        result["sec_available"] = False
        return result

    for field, tags in TAG_MAP.items():
        value, tag, form, end, filed = select_latest_fact(
            facts, tags, unit="USD", prefer_form="10-K"
        )
        result[field] = value
        result[f"{field}_tag"] = tag
        result[f"{field}_form"] = form
        result[f"{field}_period_end"] = end
        result[f"{field}_filed"] = filed
    return result


# -----------------------------------------------------------------------------
# Stale-fact detection (per documentation 7.2)
# -----------------------------------------------------------------------------


def _to_date(s: str) -> Optional[datetime]:
    if not s or pd.isna(s):
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return None


def detect_stale_facts(df: pd.DataFrame) -> pd.DataFrame:
    """For each row, find the most recent fact period_end across SEC_FIELDS as
    the company's reference anchor. Then flag any individual fact whose
    period_end is older than the reference by more than STALE_DAYS.

    Adds two columns per SEC field:
        <field>_is_stale         (bool)
        <field>_period_end_dt    (date string of the fact for the dashboard)

    Adds one column per row:
        reference_period_end     (latest period_end across all pulled facts)
    """
    period_cols = [f"{f}_period_end" for f in SEC_FIELDS]

    ref_dates: List[str] = []
    for _, row in df.iterrows():
        dates = [_to_date(row.get(c, "")) for c in period_cols]
        dates = [d for d in dates if d is not None]
        if dates:
            ref_dates.append(max(dates).strftime("%Y-%m-%d"))
        else:
            ref_dates.append("")
    df["reference_period_end"] = ref_dates

    for field in SEC_FIELDS:
        flag_col = f"{field}_is_stale"
        flags: List[bool] = []
        for _, row in df.iterrows():
            ref = _to_date(row["reference_period_end"])
            this = _to_date(row.get(f"{field}_period_end", ""))
            if ref is None or this is None:
                flags.append(False)
                continue
            flags.append((ref - this).days > STALE_DAYS)
        df[flag_col] = flags

    return df


# -----------------------------------------------------------------------------
# Ratio construction (uses stale-aware values)
# -----------------------------------------------------------------------------


def _maybe_stale(row: pd.Series, field: str) -> float:
    """Return the raw value for `field` unless flagged stale, in which case NaN.
    Used by the ratio builder so stale facts do not flow into ratios."""
    if bool(row.get(f"{field}_is_stale", False)):
        return np.nan
    val = row.get(field, np.nan)
    if pd.isna(val):
        return np.nan
    return float(val)


def build_ratios_and_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Compute total_debt, fcf, and the dashboard ratios. Stale inputs are
    treated as missing so they do not silently corrupt ratios."""
    derived_cols = {
        "total_debt": [],
        "fcf": [],
    }
    for r in RATIO_FIELDS:
        derived_cols[r] = []

    for _, row in df.iterrows():
        ltd = _maybe_stale(row, "long_term_debt")
        std = _maybe_stale(row, "short_term_debt")
        if pd.isna(ltd) and pd.isna(std):
            total_debt = np.nan
        else:
            total_debt = np.nansum([ltd, std])
        derived_cols["total_debt"].append(total_debt)

        cfo = _maybe_stale(row, "cfo")
        capex = _maybe_stale(row, "capex")
        fcf = (cfo - capex) if (pd.notna(cfo) and pd.notna(capex)) else np.nan
        derived_cols["fcf"].append(fcf)

        liabilities = _maybe_stale(row, "liabilities")
        assets = _maybe_stale(row, "assets")
        cash = _maybe_stale(row, "cash")
        revenue = _maybe_stale(row, "revenue")
        net_income = _maybe_stale(row, "net_income")
        current_assets = _maybe_stale(row, "current_assets")
        current_liabilities = _maybe_stale(row, "current_liabilities")

        derived_cols["leverage_proxy"].append(safe_div(liabilities, assets))
        derived_cols["debt_to_assets"].append(safe_div(total_debt, assets))
        derived_cols["cash_to_assets"].append(safe_div(cash, assets))
        derived_cols["net_margin"].append(safe_div(net_income, revenue))
        derived_cols["fcf_to_assets"].append(safe_div(fcf, assets))
        accrual_num = (net_income - cfo) if (pd.notna(net_income) and pd.notna(cfo)) else np.nan
        derived_cols["accrual_proxy"].append(safe_div(accrual_num, assets))
        derived_cols["current_ratio"].append(
            safe_div(current_assets, current_liabilities)
        )

    for k, v in derived_cols.items():
        df[k] = v
    return df


# -----------------------------------------------------------------------------
# Market data (yfinance)
# -----------------------------------------------------------------------------


def _empty_market_row(ticker: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {"ticker": ticker, "market_data_available": False}
    for f in MARKET_FIELDS:
        row[f] = np.nan
    return row


def _metrics_from_close(close: pd.Series) -> Dict[str, Any]:
    """Compute the dashboard market metrics from a daily close-price series."""
    out: Dict[str, Any] = {f: np.nan for f in MARKET_FIELDS}
    close = close.dropna()
    if close.empty:
        return out
    ret = close.pct_change().dropna()
    out["last_close"] = float(close.iloc[-1])
    if len(close) > 21:
        out["ret_1m"] = float(close.iloc[-1] / close.iloc[-21] - 1)
    if len(close) > 63:
        out["ret_3m"] = float(close.iloc[-1] / close.iloc[-63] - 1)
    if len(ret) > 20:
        out["vol_3m"] = float(ret.tail(63).std() * math.sqrt(252))
    if len(close) > 126:
        out["drawdown_6m"] = float(close.iloc[-1] / close.tail(126).max() - 1)
    return out


def fetch_market_batch(tickers: List[str]) -> pd.DataFrame:
    """Pull 1y daily closes for ALL tickers in a single batched yfinance call,
    then derive per-ticker metrics. Falls back to per-ticker pulls only for
    tickers that the batch could not resolve."""
    rows: List[Dict[str, Any]] = []
    if not RUN_MARKET or yf is None:
        for t in tickers:
            rows.append(_empty_market_row(t))
        return pd.DataFrame(rows)

    log(f"  Batched yfinance download for {len(tickers)} tickers (1y daily)...")
    try:
        hist = yf.download(
            tickers=" ".join(tickers),
            period="1y",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception as exc:
        log(f"  WARN: batched yfinance download failed: {exc}")
        hist = None

    found: Dict[str, pd.Series] = {}
    if hist is not None and not getattr(hist, "empty", True):
        if isinstance(hist.columns, pd.MultiIndex):
            for t in tickers:
                if t in hist.columns.get_level_values(0):
                    sub = hist[t]
                    if "Close" in sub.columns:
                        found[t] = sub["Close"]
        else:
            if len(tickers) == 1 and "Close" in hist.columns:
                found[tickers[0]] = hist["Close"]

    missing = [t for t in tickers if t not in found or found[t].dropna().empty]
    if missing:
        log(f"  Retrying {len(missing)} tickers individually: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        for t in missing:
            try:
                sub = yf.download(
                    t, period="1y", auto_adjust=True, progress=False, threads=False
                )
                if sub is not None and not sub.empty and "Close" in sub.columns:
                    close = sub["Close"]
                    if hasattr(close, "columns"):
                        close = close.iloc[:, 0]
                    found[t] = close
            except Exception as exc:
                log(f"    WARN: market data failed for {t}: {exc}")

    for t in tickers:
        row = _empty_market_row(t)
        if t in found and not found[t].dropna().empty:
            metrics = _metrics_from_close(found[t])
            row.update(metrics)
            row["market_data_available"] = pd.notna(metrics.get("last_close"))
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# News data (GDELT)
# -----------------------------------------------------------------------------


def gdelt_mentions(company_name: str, days: int = 30) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "company_name": company_name,
        "news_data_available": False,
        "news_mentions_30d": np.nan,
        "risk_news_mentions_30d": np.nan,
    }
    if not RUN_GDELT:
        return result

    end = datetime.now()
    start = end - timedelta(days=days)
    base = "https://api.gdeltproject.org/api/v2/doc/doc"

    def count(query: str) -> float:
        params = (
            f"query={quote(query)}&mode=timelinevol&format=json"
            f"&startdatetime={start.strftime('%Y%m%d%H%M%S')}"
            f"&enddatetime={end.strftime('%Y%m%d%H%M%S')}"
        )
        try:
            resp = requests.get(f"{base}?{params}", timeout=GDELT_TIMEOUT)
            if resp.status_code != 200:
                return np.nan
            data = resp.json()
            timeline = data.get("timeline", [])
            if not timeline:
                return 0.0
            total = 0.0
            for entry in timeline:
                for point in entry.get("data", []):
                    total += float(point.get("value", 0))
            return total
        except Exception:
            return np.nan

    clean_name = (
        company_name.replace(", Inc.", "")
        .replace(" Inc.", "")
        .replace(" Corp", "")
        .replace(" Corporation", "")
    )
    all_mentions = count(f'"{clean_name}"')
    time.sleep(GDELT_SLEEP_SECONDS)
    risk_mentions = count(
        f'"{clean_name}" (bankruptcy OR restructuring OR lawsuit OR investigation '
        "OR default OR downgrade OR delinquency OR liquidity OR fraud)"
    )

    if pd.notna(all_mentions) or pd.notna(risk_mentions):
        result["news_data_available"] = True
    result["news_mentions_30d"] = all_mentions
    result["risk_news_mentions_30d"] = risk_mentions
    return result


# -----------------------------------------------------------------------------
# Assumption fill (sector-level fallback for missing/stale ratios)
# -----------------------------------------------------------------------------


def fill_assumptions(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ratios with sector-level assumptions and flag each fill.

    A ratio is considered missing if NaN for any reason: SEC unavailable,
    raw input missing, or a stale input that we dropped during ratio building.
    """
    for col in RATIO_FIELDS:
        if col not in df.columns:
            df[col] = np.nan
        df[f"{col}_is_assumption"] = df[col].isna()
        for idx, row in df[df[col].isna()].iterrows():
            sector = row.get("sector_group", "")
            val = SECTOR_ASSUMPTIONS.get(sector, DEFAULT_ASSUMPTIONS).get(
                col, DEFAULT_ASSUMPTIONS[col]
            )
            df.at[idx, col] = val

    for col, val in MARKET_DEFAULTS.items():
        if col not in df.columns:
            df[col] = np.nan
        df[f"{col}_is_assumption"] = df[col].isna()
        df[col] = df[col].fillna(val)

    if "last_close" not in df.columns:
        df["last_close"] = np.nan
    df["last_close_is_assumption"] = df["last_close"].isna()

    for col in NEWS_FIELDS:
        if col not in df.columns:
            df[col] = np.nan
        df[f"{col}_is_assumption"] = df[col].isna()

    return df


# -----------------------------------------------------------------------------
# Data quality (per documentation 7.3)
# -----------------------------------------------------------------------------


def compute_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Implements the doc 7.3 weighted score:

        data_quality_score = clip(100
                                  - 70 * (assumption_metric_count / total_assumption_checks)
                                  - 30 * (stale_fact_count       / total_stale_checks),
                                  0, 100)

    Assumption checks count = ratio assumption flags + market metric assumption flags.
    Stale checks count       = number of SEC fields we attempted (TAG_MAP).
    """
    ratio_assumption_cols = [f"{r}_is_assumption" for r in RATIO_FIELDS]
    market_assumption_cols = [f"{m}_is_assumption" for m in MARKET_DEFAULTS.keys()]
    assumption_cols = ratio_assumption_cols + market_assumption_cols
    stale_cols = [f"{f}_is_stale" for f in SEC_FIELDS]

    total_assumption_checks = max(1, len(assumption_cols))
    total_stale_checks = max(1, len(stale_cols))

    df["assumption_metric_count"] = (
        df[assumption_cols].sum(axis=1) if assumption_cols else 0
    )
    df["stale_fact_count"] = (
        df[stale_cols].sum(axis=1) if stale_cols else 0
    )

    raw_score = (
        100.0
        - 70.0 * df["assumption_metric_count"] / total_assumption_checks
        - 30.0 * df["stale_fact_count"] / total_stale_checks
    )
    df["data_quality_score"] = raw_score.clip(0, 100).round(1)

    def note_for(score: float) -> str:
        if score >= 85:
            return "Strong public-data coverage"
        if score >= 70:
            return "Usable for demo; some stale/missing fields"
        if score >= 50:
            return "Assumption-heavy; use with caution"
        return "Weak coverage; needs manual review or better data source"

    df["data_quality_note"] = df["data_quality_score"].apply(note_for)
    return df


# -----------------------------------------------------------------------------
# Diagnostics (long format for dashboard data-quality page)
# -----------------------------------------------------------------------------


def build_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format per-company flag detail. One row per (company, flag) where
    a flag fired. Useful for the doc 14 page-4 Data Coverage view."""
    rows: List[Dict[str, Any]] = []
    flag_cols = [c for c in df.columns if c.endswith("_is_assumption") or c.endswith("_is_stale")]
    for _, r in df.iterrows():
        for c in flag_cols:
            if bool(r.get(c, False)):
                metric = c.rsplit("_is_", 1)[0]
                kind = "assumption" if c.endswith("_is_assumption") else "stale"
                rows.append(
                    {
                        "ticker": r["ticker"],
                        "company_name": r["company_name"],
                        "sector_group": r["sector_group"],
                        "metric": metric,
                        "flag_type": kind,
                        "fact_period_end": r.get(f"{metric}_period_end", ""),
                        "reference_period_end": r.get("reference_period_end", ""),
                    }
                )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


DASHBOARD_COLUMNS: List[str] = (
    [
        "ticker", "company_name", "sector_group", "industry_group",
        "index_memberships",
        "rating_seed", "rating_bucket", "rating_source",
        "cik", "sec_available", "reference_period_end",
    ]
    + SEC_FIELDS
    + ["total_debt", "fcf"]
    + RATIO_FIELDS
    + MARKET_FIELDS
    + ["market_data_available"]
    + NEWS_FIELDS
    + ["news_data_available"]
    + [f"{r}_is_assumption" for r in RATIO_FIELDS]
    + [f"{m}_is_assumption" for m in MARKET_DEFAULTS.keys()]
    + ["last_close_is_assumption"]
    + [f"{n}_is_assumption" for n in NEWS_FIELDS]
    + [f"{f}_is_stale" for f in SEC_FIELDS]
    + ["assumption_metric_count", "stale_fact_count", "data_quality_score", "data_quality_note"]
)


def main() -> None:
    t_start = time.time()
    log("Surveillance data prep starting...")
    log(f"  SEC_USER_AGENT  = {SEC_USER_AGENT}")
    log(f"  RUN_MARKET      = {RUN_MARKET}")
    log(f"  RUN_GDELT       = {RUN_GDELT}")
    log(f"  STALE_DAYS      = {STALE_DAYS}")
    log("")

    universe = load_universe()
    log(f"Loaded universe: {len(universe)} companies")

    log("Fetching SEC ticker -> CIK map...")
    sec_map = get_sec_ticker_map()
    universe = universe.merge(
        sec_map[["ticker", "cik", "sec_title"]], on="ticker", how="left"
    )
    n_with_cik = int(universe["cik"].notna().sum())
    log(f"  Mapped {n_with_cik}/{len(universe)} tickers to CIKs")

    log("\nPulling SEC companyfacts (cached on disk)...")
    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    for i, (_, row) in enumerate(universe.iterrows(), start=1):
        cik = row.get("cik") if pd.notna(row.get("cik")) else None
        if i % 10 == 0 or i == len(universe):
            log(f"  SEC progress: {i}/{len(universe)} (last: {row['ticker']})")
        rows.append(financials_for_company(row, cik))
    financials = pd.DataFrame(rows)
    log(f"  SEC pull done in {time.time() - t0:.1f}s")

    if RUN_MARKET:
        log("\nPulling market data (yfinance, batched)...")
        t0 = time.time()
        market = fetch_market_batch(list(universe["ticker"]))
        financials = financials.merge(market, on="ticker", how="left")
        log(f"  Market pull done in {time.time() - t0:.1f}s "
            f"({int(market['market_data_available'].sum())}/{len(market)} resolved)")
    else:
        for col in MARKET_FIELDS:
            financials[col] = np.nan
        financials["market_data_available"] = False

    raw_path = OUTPUT_DIR / "surveillance_raw_financials.csv"
    financials.to_csv(raw_path, index=False)
    log(f"\nWrote partial raw financials (pre-news) -> {raw_path}")

    if RUN_GDELT:
        log("\nPulling GDELT news mentions...")
        t0 = time.time()
        news_rows: List[Dict[str, Any]] = []
        total = len(universe)
        for i, company in enumerate(universe["company_name"], start=1):
            t_co = time.time()
            res = gdelt_mentions(company)
            news_rows.append(res)
            elapsed = time.time() - t_co
            if i % 5 == 0 or i == total or elapsed > 2.0:
                log(
                    f"  GDELT {i:>3}/{total} {company[:40]:<40} "
                    f"all={res['news_mentions_30d']!s:>8} "
                    f"risk={res['risk_news_mentions_30d']!s:>8} "
                    f"({elapsed:.1f}s)"
                )
            time.sleep(GDELT_SLEEP_SECONDS)
        news = pd.DataFrame(news_rows)
        financials = financials.merge(news, on="company_name", how="left")
        log(f"  GDELT pull done in {time.time() - t0:.1f}s "
            f"({int(news['news_data_available'].sum())}/{len(news)} returned data)")
        financials.to_csv(raw_path, index=False)
        log(f"  Updated raw financials with news -> {raw_path}")
    else:
        for col in NEWS_FIELDS:
            financials[col] = np.nan
        financials["news_data_available"] = False

    log("\nDetecting stale facts (550-day rule)...")
    dash = detect_stale_facts(financials.copy())

    log("Building ratios from stale-aware inputs...")
    dash = build_ratios_and_derived(dash)

    log("Deriving synthetic rating buckets for NR tickers...")
    n_pre_nr = int((dash["rating_bucket"] == "NR").sum())
    derived_count = 0
    for idx, r in dash.iterrows():
        if r["rating_bucket"] != "NR":
            continue
        new_bucket = derive_synthetic_rating(
            r.get("debt_to_assets"),
            r.get("leverage_proxy"),
            r.get("net_margin"),
            r.get("fcf_to_assets"),
        )
        if new_bucket != "NR":
            dash.at[idx, "rating_bucket"] = new_bucket
            existing_src = str(r.get("rating_source", ""))
            if "derive" not in existing_src.lower():
                dash.at[idx, "rating_source"] = (
                    f"synthetic (derived from leverage + margin)"
                )
            derived_count += 1
    log(f"  {derived_count}/{n_pre_nr} NR tickers received a synthetic rating bucket")

    log("Filling missing/stale ratios with sector assumptions...")
    dash = fill_assumptions(dash)

    log("Computing data quality score and note...")
    dash = compute_data_quality(dash)

    for col in DASHBOARD_COLUMNS:
        if col not in dash.columns:
            dash[col] = np.nan

    dash_path = OUTPUT_DIR / "surveillance_dashboard_input.csv"
    dash[DASHBOARD_COLUMNS].to_csv(dash_path, index=False)
    log(f"  -> {dash_path}")

    log("Building diagnostics (long format)...")
    diag = build_diagnostics(dash)
    diag_path = OUTPUT_DIR / "surveillance_data_diagnostics.csv"
    diag.to_csv(diag_path, index=False)
    log(f"  -> {diag_path}")

    log("\nDone.")
    log(f"  Total elapsed:             {time.time() - t_start:.1f}s")
    log(f"  Companies processed:       {len(dash)}")
    log(f"  SEC available:             {int(dash['sec_available'].sum())}")
    log(f"  Market data available:     {int(dash['market_data_available'].fillna(False).sum())}")
    log(f"  News data available:       {int(dash['news_data_available'].fillna(False).sum())}")
    log(f"  Avg data quality score:    {dash['data_quality_score'].mean():.1f}")
    log(f"  Avg assumptions per row:   {dash['assumption_metric_count'].mean():.1f}")
    log(f"  Avg stale facts per row:   {dash['stale_fact_count'].mean():.1f}")


if __name__ == "__main__":
    main()
