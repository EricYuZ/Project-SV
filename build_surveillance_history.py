"""
Surveillance Tool - Quarterly Time-Series Extractor
====================================================

Reads the existing SEC `companyfacts` JSON cache (already populated by
`build_surveillance_data.py`) and produces a long-format quarterly history
CSV covering the last N years (default 3) for the dashboard's time-series
view.

No network calls are made: this script is a pure transform over
`./cache_sec/CIK*_companyfacts.json`. Run after the main data prep script.

What it does
------------
For each company in the universe:

  1. Loads cached SEC facts.
  2. For BALANCE SHEET tags (instant): keeps every quarter-end snapshot.
     Dedupes per period_end keeping the LATEST `filed`.
  3. For INCOME STATEMENT / CASH FLOW tags (duration): builds two views:
       - QUARTERLY: discrete ~90-day records direct from SEC. Falls back
         to (FY annual) - (sum of Q1+Q2+Q3) for Q4 when discrete Q4 is
         not reported, and YTD subtraction (H1 - Q1, etc.) when only
         YTD facts exist.
       - TTM: rolling sum of the last 4 quarterly values at each quarter end.
  4. Computes credit ratios at every quarter end using point-in-time
     balance-sheet values + TTM income/cash-flow (leverage_proxy,
     debt_to_assets, cash_to_assets, net_margin, fcf_to_assets,
     accrual_proxy, current_ratio).

Output
------
./output/surveillance_quarterly_history.csv  (long format)

Columns: ticker, company_name, sector_group, industry_group, rating_bucket,
         period_end, metric, value, value_type, fy, fp, form, filed

  value_type is one of: instant | quarterly | ttm | ratio

Run
---
    python build_surveillance_history.py            # default 3 years
    HISTORY_YEARS=5 python build_surveillance_history.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


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

HISTORY_YEARS = int(os.getenv("HISTORY_YEARS", "3"))
TODAY = date.today()
EARLIEST_DATE = date(TODAY.year - HISTORY_YEARS - 1, 1, 1)  # buffer for TTM lookback

# The dashboard scoring path only reads these columns and metric/value_type
# combinations. Keeping the output compact avoids carrying source metadata that
# is useful for extraction forensics but not needed at runtime.
DASHBOARD_HISTORY_COLUMNS = ["ticker", "period_end", "metric", "value", "value_type"]
DASHBOARD_HISTORY_METRICS: Dict[str, set[str]] = {
    "instant": {
        "accounts_receivable",
        "ppe_net",
        "current_assets",
        "current_liabilities",
        "assets",
        "liabilities",
        "cash",
        "total_debt",
    },
    "ttm": {
        "revenue",
        "cogs",
        "depreciation",
        "sga",
        "net_income",
        "cfo",
        "capex",
        "fcf",
        "interest_expense",
    },
    "quarterly": {"revenue", "cogs", "depreciation", "sga", "net_income", "cfo", "capex"},
    "ratio": {
        "leverage_proxy",
        "debt_to_assets",
        "cash_to_assets",
        "net_margin",
        "fcf_to_assets",
        "accrual_proxy",
        "current_ratio",
    },
}


# ---------------------------------------------------------------------------
# Tag map - same as the main data prep script.
# Split here by financial statement type so we can apply the correct
# instant vs. duration logic.
# ---------------------------------------------------------------------------

INSTANT_TAGS: Dict[str, List[str]] = {
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
    # Added for Beneish M-Score AQI / DEPI components (EWIF v1.0 Section VI.B).
    "ppe_net": [
        "PropertyPlantAndEquipmentNet",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
    ],
}

DURATION_TAGS: Dict[str, List[str]] = {
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
    # GMI = Gross Margin Index (needs COGS), DEPI (Depreciation),
    # SGAI = SG&A Index.
    "cogs": [
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfRevenue",
        "CostOfServices",
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
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def _records_for_single_tag(
    facts: Dict[str, Any], tag: str
) -> List[Dict[str, Any]]:
    """Collect all USD records for ONE XBRL tag."""
    out: List[Dict[str, Any]] = []
    us = facts.get("facts", {}).get("us-gaap", {}) if facts else {}
    node = us.get(tag, {})
    for r in node.get("units", {}).get("USD", []):
        if r.get("val") is None:
            continue
        form = r.get("form", "")
        if form not in {"10-K", "10-Q", "20-F", "40-F"}:
            continue
        r2 = dict(r)
        r2["_tag"] = tag
        out.append(r2)
    return out


def _select_best_tag(
    facts: Dict[str, Any], tags: Iterable[str], require_duration: bool
) -> Optional[str]:
    """For one metric, pick the tag synonym with the most records in our date
    range. Mixing synonyms (e.g. JPM 'Revenues' vs 'NoninterestIncome') leads
    to apples-to-oranges arithmetic, so we lock in a single tag per company."""
    best_tag: Optional[str] = None
    best_count = 0
    for tag in tags:
        recs = _records_for_single_tag(facts, tag)
        relevant = 0
        for r in recs:
            end = _to_date(r.get("end"))
            if end is None or end < EARLIEST_DATE:
                continue
            if require_duration and not r.get("start"):
                continue
            relevant += 1
        if relevant > best_count:
            best_count = relevant
            best_tag = tag
    return best_tag


def _records_for_tags(
    facts: Dict[str, Any], tags: Iterable[str], require_duration: bool = False
) -> List[Dict[str, Any]]:
    """Pick the best single tag synonym for this company and return its records.
    Returns [] if no tag has data."""
    chosen = _select_best_tag(facts, tags, require_duration=require_duration)
    if chosen is None:
        return []
    return _records_for_single_tag(facts, chosen)


def _dedupe_by_period_end(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """For instant facts: one entry per period_end, keeping the latest filed."""
    by_end: Dict[str, Dict[str, Any]] = {}
    for r in records:
        end = r.get("end")
        if not end:
            continue
        existing = by_end.get(end)
        if existing is None or (r.get("filed", "") > existing.get("filed", "")):
            by_end[end] = r
    return sorted(by_end.values(), key=lambda x: x["end"])


def _dedupe_by_start_end(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """For duration facts: one entry per (start,end), latest filed wins."""
    by_period: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in records:
        s, e = r.get("start"), r.get("end")
        if not s or not e:
            continue
        key = (s, e)
        existing = by_period.get(key)
        if existing is None or (r.get("filed", "") > existing.get("filed", "")):
            by_period[key] = r
    return sorted(by_period.values(), key=lambda x: (x["end"], x["start"]))


# ---------------------------------------------------------------------------
# Quarterly series builder for duration tags
# ---------------------------------------------------------------------------


def _categorize(rec: Dict[str, Any]) -> str:
    """Tag a duration record by its length (q | h1 | 3q | fy | other)."""
    s = _to_date(rec.get("start"))
    e = _to_date(rec.get("end"))
    if s is None or e is None:
        return "other"
    days = (e - s).days
    if 80 <= days <= 100:
        return "q"
    if 170 <= days <= 200:
        return "h1"
    if 260 <= days <= 290:
        return "3q"
    if 350 <= days <= 380:
        return "fy"
    return "other"


def build_quarterly_series_duration(
    records: List[Dict[str, Any]],
) -> List[Tuple[date, float, str, Dict[str, Any]]]:
    """Convert mixed quarterly / YTD / annual records (from a SINGLE tag) into
    discrete quarterly values. Returns sorted list of
    (period_end, quarterly_value, source_form, representative_record).

    Indexes by calendar period_end (not by fiscal-year tags), which is robust
    to restated filings that re-tag prior periods with a different `fy`.

    Strategy:
      1. Take all discrete ~90-day records as-is.
      2. YTD subtraction (H1 - Q1, 3Q - H1) using the calendar-quarter chain.
      3. Q4 = FY annual - (Q1 + Q2 + Q3), when Q4 not reported discretely
         and the three preceding quarters are known at the standard
         calendar-quarter ends.
    """
    records = _dedupe_by_start_end(records)

    # Index records by (start, end) and by category.
    discrete: Dict[date, Dict[str, Any]] = {}
    h1_by_end: Dict[date, Dict[str, Any]] = {}
    third_q_by_end: Dict[date, Dict[str, Any]] = {}
    fy_by_end: Dict[date, Dict[str, Any]] = {}

    for r in records:
        cat = _categorize(r)
        end = _to_date(r["end"])
        if end is None:
            continue
        if cat == "q":
            discrete[end] = r
        elif cat == "h1":
            h1_by_end[end] = r
        elif cat == "3q":
            third_q_by_end[end] = r
        elif cat == "fy":
            fy_by_end[end] = r

    def _prev_q_end(end: date) -> Optional[date]:
        """Return the standard calendar quarter-end date 3 months earlier."""
        # Step back ~92 days, then snap to nearest quarter end.
        approx = end - timedelta(days=92)
        # Find nearest quarter end (Mar 31, Jun 30, Sep 30, Dec 31) close to approx
        candidates = [
            date(approx.year, 3, 31),
            date(approx.year, 6, 30),
            date(approx.year, 9, 30),
            date(approx.year, 12, 31),
            date(approx.year - 1, 12, 31),
        ]
        return min(candidates, key=lambda d: abs((d - approx).days))

    # Pass 2: YTD subtraction.
    # H1 ending date implies Q2 of that year. Q2 = H1 - Q1 (where Q1 = period 3 months earlier).
    for end, h in h1_by_end.items():
        if end in discrete:
            continue
        q1_end = _prev_q_end(end)
        if q1_end in discrete:
            val = float(h["val"]) - float(discrete[q1_end]["val"])
            d = dict(h)
            d["val"] = val
            d["_derived"] = "H1-Q1"
            discrete[end] = d

    # 3Q ending date = Q3. Q3 = 3Q_YTD - Q1 - Q2.
    for end, t in third_q_by_end.items():
        if end in discrete:
            continue
        q2_end = _prev_q_end(end)
        q1_end = _prev_q_end(q2_end) if q2_end else None
        if q1_end in discrete and q2_end in discrete:
            val = float(t["val"]) - float(discrete[q1_end]["val"]) - float(discrete[q2_end]["val"])
            d = dict(t)
            d["val"] = val
            d["_derived"] = "3Q-Q1-Q2"
            discrete[end] = d

    # Pass 3: Q4 = FY - (Q1+Q2+Q3) when Q4 not reported discretely.
    for end, annual in fy_by_end.items():
        if end in discrete:
            continue
        q3_end = _prev_q_end(end)
        q2_end = _prev_q_end(q3_end) if q3_end else None
        q1_end = _prev_q_end(q2_end) if q2_end else None
        if q1_end in discrete and q2_end in discrete and q3_end in discrete:
            val = (
                float(annual["val"])
                - float(discrete[q1_end]["val"])
                - float(discrete[q2_end]["val"])
                - float(discrete[q3_end]["val"])
            )
            d = dict(annual)
            d["val"] = val
            d["_derived"] = "FY-Q1-Q2-Q3"
            discrete[end] = d

    out: List[Tuple[date, float, str, Dict[str, Any]]] = []
    for end in sorted(discrete.keys()):
        r = discrete[end]
        out.append((end, float(r["val"]), r.get("form", ""), r))
    return out


# ---------------------------------------------------------------------------
# TTM series from quarterly
# ---------------------------------------------------------------------------


def build_ttm_series(
    quarterly: List[Tuple[date, float, str, Dict[str, Any]]],
) -> List[Tuple[date, float]]:
    """Rolling sum of last 4 quarters at each period_end. Requires at least
    4 sequential quarters; emits NaN otherwise."""
    out: List[Tuple[date, float]] = []
    for i in range(len(quarterly)):
        if i < 3:
            continue
        # Confirm consecutive quarters (within ~95 days each step)
        valid = True
        for j in range(i - 3, i):
            gap = (quarterly[j + 1][0] - quarterly[j][0]).days
            if gap < 70 or gap > 110:
                valid = False
                break
        if not valid:
            continue
        ttm = sum(q[1] for q in quarterly[i - 3 : i + 1])
        out.append((quarterly[i][0], ttm))
    return out


# ---------------------------------------------------------------------------
# Per-company extraction
# ---------------------------------------------------------------------------


@dataclass
class CompanyHistory:
    ticker: str
    company_name: str
    sector_group: str
    industry_group: str
    rating_bucket: str
    instant: Dict[str, List[Dict[str, Any]]]
    quarterly: Dict[str, List[Tuple[date, float, str, Dict[str, Any]]]]
    ttm: Dict[str, List[Tuple[date, float]]]
    source_tags: Dict[str, str]  # metric -> XBRL tag actually used


def extract_company_history(row: pd.Series, cik: Optional[str]) -> Optional[CompanyHistory]:
    if not cik:
        return None
    cache_path = CACHE_DIR / f"CIK{cik}_companyfacts.json"
    if not cache_path.exists():
        return None
    facts = json.loads(cache_path.read_text(encoding="utf-8"))

    source_tags: Dict[str, str] = {}

    instant: Dict[str, List[Dict[str, Any]]] = {}
    for metric, tags in INSTANT_TAGS.items():
        chosen = _select_best_tag(facts, tags, require_duration=False)
        if chosen is None:
            instant[metric] = []
            continue
        source_tags[metric] = chosen
        recs = _records_for_single_tag(facts, chosen)
        recs = _dedupe_by_period_end(recs)
        recs = [r for r in recs if (_to_date(r["end"]) or date(1900, 1, 1)) >= EARLIEST_DATE]
        instant[metric] = recs

    quarterly: Dict[str, List[Tuple[date, float, str, Dict[str, Any]]]] = {}
    ttm: Dict[str, List[Tuple[date, float]]] = {}
    for metric, tags in DURATION_TAGS.items():
        chosen = _select_best_tag(facts, tags, require_duration=True)
        if chosen is None:
            quarterly[metric] = []
            ttm[metric] = []
            continue
        source_tags[metric] = chosen
        recs = _records_for_single_tag(facts, chosen)
        q_series = build_quarterly_series_duration(recs)
        q_series = [q for q in q_series if q[0] >= EARLIEST_DATE]
        quarterly[metric] = q_series
        ttm[metric] = build_ttm_series(q_series)

    return CompanyHistory(
        ticker=row["ticker"],
        company_name=row["company_name"],
        sector_group=row["sector_group"],
        industry_group=row["industry_group"],
        rating_bucket=row.get("rating_bucket", "NR"),
        instant=instant,
        quarterly=quarterly,
        ttm=ttm,
        source_tags=source_tags,
    )


# ---------------------------------------------------------------------------
# Long-format flattener + ratio computation
# ---------------------------------------------------------------------------


RATIO_DEFS = [
    # (ratio_name, numerator_metric, numerator_kind, denominator_metric, denominator_kind)
    ("leverage_proxy", "liabilities", "instant", "assets", "instant"),
    ("debt_to_assets", "total_debt", "derived_instant", "assets", "instant"),
    ("cash_to_assets", "cash", "instant", "assets", "instant"),
    ("net_margin", "net_income", "ttm", "revenue", "ttm"),
    ("fcf_to_assets", "fcf", "derived_ttm", "assets", "instant"),
    ("accrual_proxy", "accrual_num", "derived_ttm", "assets", "instant"),
    ("current_ratio", "current_assets", "instant", "current_liabilities", "instant"),
]


def _instant_value_at(history_recs: List[Dict[str, Any]], qd: date) -> Optional[float]:
    for r in history_recs:
        if r.get("end") == qd.isoformat():
            return float(r["val"])
    return None


def _ttm_value_at(ttm_series: List[Tuple[date, float]], qd: date) -> Optional[float]:
    for d, v in ttm_series:
        if d == qd:
            return float(v)
    return None


def to_long_format(history: CompanyHistory) -> List[Dict[str, Any]]:
    """Flatten one company's history into long-format rows for the dashboard."""
    rows: List[Dict[str, Any]] = []
    common = {
        "ticker": history.ticker,
        "company_name": history.company_name,
        "sector_group": history.sector_group,
        "industry_group": history.industry_group,
        "rating_bucket": history.rating_bucket,
    }

    src = history.source_tags

    for metric, recs in history.instant.items():
        tag = src.get(metric, "")
        for r in recs:
            d = _to_date(r["end"])
            if d is None or d < EARLIEST_DATE:
                continue
            rows.append({**common,
                         "period_end": d.isoformat(),
                         "metric": metric,
                         "value": float(r["val"]),
                         "value_type": "instant",
                         "source_tag": tag,
                         "fy": r.get("fy"),
                         "fp": r.get("fp"),
                         "form": r.get("form", ""),
                         "filed": r.get("filed", "")})

    for metric, qlist in history.quarterly.items():
        tag = src.get(metric, "")
        for d, v, form, r in qlist:
            if d < EARLIEST_DATE:
                continue
            rows.append({**common,
                         "period_end": d.isoformat(),
                         "metric": metric,
                         "value": v,
                         "value_type": "quarterly",
                         "source_tag": tag,
                         "fy": r.get("fy"),
                         "fp": r.get("fp"),
                         "form": form,
                         "filed": r.get("filed", "")})

    cfo_ttm = dict(history.ttm.get("cfo", []))
    capex_ttm = dict(history.ttm.get("capex", []))
    ni_ttm = dict(history.ttm.get("net_income", []))

    for metric, tlist in history.ttm.items():
        tag = src.get(metric, "")
        for d, v in tlist:
            if d < EARLIEST_DATE:
                continue
            rows.append({**common,
                         "period_end": d.isoformat(),
                         "metric": metric,
                         "value": v,
                         "value_type": "ttm",
                         "source_tag": tag,
                         "fy": None, "fp": "TTM", "form": "", "filed": ""})

    fcf_ttm: Dict[date, float] = {}
    for d, v in cfo_ttm.items():
        c = capex_ttm.get(d)
        if c is not None:
            fcf = v - c
            fcf_ttm[d] = fcf
            rows.append({**common,
                         "period_end": d.isoformat(),
                         "metric": "fcf",
                         "value": fcf,
                         "value_type": "ttm",
                         "source_tag": "derived (cfo - capex)",
                         "fy": None, "fp": "TTM", "form": "", "filed": ""})
    accrual_num: Dict[date, float] = {}
    for d, ni in ni_ttm.items():
        c = cfo_ttm.get(d)
        if c is not None:
            accrual_num[d] = ni - c

    ltd_recs = history.instant.get("long_term_debt", [])
    std_recs = history.instant.get("short_term_debt", [])
    asset_dates = {_to_date(r["end"]) for r in history.instant.get("assets", [])}
    asset_dates.discard(None)
    total_debt_by_date: Dict[date, float] = {}
    for d in asset_dates:
        ltd = _instant_value_at(ltd_recs, d)
        std = _instant_value_at(std_recs, d)
        if ltd is None and std is None:
            continue
        total_debt_by_date[d] = (ltd or 0) + (std or 0)
        rows.append({**common,
                     "period_end": d.isoformat(),
                     "metric": "total_debt",
                     "value": total_debt_by_date[d],
                     "value_type": "instant",
                     "source_tag": "derived (long_term_debt + short_term_debt)",
                     "fy": None, "fp": "", "form": "", "filed": ""})

    for d in sorted(asset_dates):
        liab = _instant_value_at(history.instant.get("liabilities", []), d)
        assets = _instant_value_at(history.instant.get("assets", []), d)
        cash = _instant_value_at(history.instant.get("cash", []), d)
        ca = _instant_value_at(history.instant.get("current_assets", []), d)
        cl = _instant_value_at(history.instant.get("current_liabilities", []), d)
        td = total_debt_by_date.get(d)
        ni_t = ni_ttm.get(d)
        rev_t = dict(history.ttm.get("revenue", [])).get(d)
        fcf_t = fcf_ttm.get(d)
        acc_t = accrual_num.get(d)

        def _add_ratio(name: str, num: Optional[float], den: Optional[float]) -> None:
            if num is None or den is None or den == 0:
                return
            rows.append({**common,
                         "period_end": d.isoformat(),
                         "metric": name,
                         "value": num / den,
                         "value_type": "ratio",
                         "source_tag": "computed",
                         "fy": None, "fp": "", "form": "", "filed": ""})

        _add_ratio("leverage_proxy", liab, assets)
        _add_ratio("debt_to_assets", td, assets)
        _add_ratio("cash_to_assets", cash, assets)
        _add_ratio("net_margin", ni_t, rev_t)
        _add_ratio("fcf_to_assets", fcf_t, assets)
        _add_ratio("accrual_proxy", acc_t, assets)
        _add_ratio("current_ratio", ca, cl)

    return rows


# ---------------------------------------------------------------------------
# Universe + CIK reuse
# ---------------------------------------------------------------------------


def load_universe_with_cik() -> pd.DataFrame:
    universe = pd.read_csv(UNIVERSE_FILE)
    universe["ticker"] = universe["ticker"].astype(str).str.upper().str.strip()
    # Re-use CIKs by reading the cached SEC ticker map.
    map_path = CACHE_DIR / "company_tickers.json"
    if not map_path.exists():
        raise SystemExit(
            "Missing cache_sec/company_tickers.json. Run build_surveillance_data.py first."
        )
    data = json.loads(map_path.read_text(encoding="utf-8"))
    rows = [
        {"ticker": str(v["ticker"]).upper(), "cik": str(v["cik_str"]).zfill(10)}
        for _, v in data.items()
    ]
    sec_map = pd.DataFrame(rows)
    universe = universe.merge(sec_map, on="ticker", how="left")
    return universe


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Surveillance history extractor starting...")
    print(f"  HISTORY_YEARS = {HISTORY_YEARS}")
    print(f"  EARLIEST_DATE = {EARLIEST_DATE}")
    print(f"  Cache dir     = {CACHE_DIR}")
    print()

    universe = load_universe_with_cik()
    n_with_cik = int(universe["cik"].notna().sum())
    print(f"Loaded universe: {len(universe)} companies ({n_with_cik} with CIK)")

    all_rows: List[Dict[str, Any]] = []
    n_extracted = 0
    for _, row in universe.iterrows():
        cik = row.get("cik") if pd.notna(row.get("cik")) else None
        # Use the rating_bucket from dashboard input if available (it has the parsed version)
        history = extract_company_history(row, cik)
        if history is None:
            continue
        n_extracted += 1
        all_rows.extend(to_long_format(history))

    print(f"Extracted history for {n_extracted}/{len(universe)} companies")

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["ticker", "metric", "value_type", "period_end"])

    keep_mask = pd.Series(False, index=df.index)
    for value_type, metrics in DASHBOARD_HISTORY_METRICS.items():
        keep_mask = keep_mask | (
            (df["value_type"] == value_type) & df["metric"].isin(metrics)
        )
    df = df.loc[keep_mask, DASHBOARD_HISTORY_COLUMNS].copy()

    out_path = OUTPUT_DIR / "surveillance_quarterly_history.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df):,} rows -> {out_path}")

    # Quick summary
    print("\nRows per value_type:")
    print(df["value_type"].value_counts().to_string())

    print("\nRows per metric (top 25):")
    print(df["metric"].value_counts().head(25).to_string())

    if not df.empty:
        period_dates = pd.to_datetime(df["period_end"])
        print(f"\nPeriod end range: {period_dates.min().date()}  ->  {period_dates.max().date()}")
        rows_per_co = df.groupby("ticker").size().describe()
        print(f"\nRows per company: min={int(rows_per_co['min'])}, "
              f"median={int(rows_per_co['50%'])}, max={int(rows_per_co['max'])}")


if __name__ == "__main__":
    main()
