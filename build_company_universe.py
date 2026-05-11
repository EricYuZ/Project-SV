"""
Major-index company universe builder.

Builds the restart baseline requested for the surveillance project:

    S&P 500 + Nasdaq 100 + Dow Jones Industrial Average

The output keeps one row per unique ticker and stores index membership in a
structured field so future additions can be handled by editing or extending the
universe CSV rather than changing dashboard code.

Sources
-------
Wikipedia constituent tables:
    - List of S&P 500 companies
    - Nasdaq-100
    - Dow Jones Industrial Average

Run
---
    python build_company_universe.py

Optional env vars
-----------------
    UNIVERSE_OUT    Output CSV path
"""

from __future__ import annotations

import os
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
LEGACY_UNIVERSE = BASE_DIR / "ewif_company_universe_us_100.csv"
DEFAULT_OUT = BASE_DIR / "ewif_company_universe_major_indices.csv"
UNIVERSE_OUT = Path(os.getenv("UNIVERSE_OUT", str(DEFAULT_OUT)))

SOURCES = {
    "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "Nasdaq 100": "https://en.wikipedia.org/wiki/Nasdaq-100",
    "Dow": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
}

HTTP_HEADERS = {
    "User-Agent": "Surveillance demo research contact@example.com",
    "Accept": "text/html",
    "Accept-Language": "en-US,en;q=0.9",
}


INDUSTRIALS_SUB_TO_GROUP = {
    "airlines": "Transportation",
    "passenger airlines": "Transportation",
    "air freight": "Transportation",
    "logistics": "Transportation",
    "marine transportation": "Transportation",
    "trucking": "Transportation",
    "ground transportation": "Transportation",
    "rail": "Transportation",
    "aerospace": "Aerospace & Defense",
    "defense": "Aerospace & Defense",
}

CONSUMER_DISC_SUB_TO_GROUP = {
    "automobile": "Autos",
    "automotive": "Autos",
    "motorcycle": "Autos",
}


def map_sector(gics_sector: str, industry: str) -> str:
    """Map source sector/industry labels to the dashboard sector taxonomy."""
    s = (gics_sector or "").strip()
    sub = (industry or "").strip().lower()

    if s == "Industrials":
        for key, mapped in INDUSTRIALS_SUB_TO_GROUP.items():
            if key in sub:
                return mapped
        return "Industrials"

    if s == "Consumer Discretionary":
        for key, mapped in CONSUMER_DISC_SUB_TO_GROUP.items():
            if key in sub:
                return mapped
        return "Consumer Discretionary"

    if s == "Information Technology":
        return "Technology"

    if s == "Communication Services":
        if any(k in sub for k in ("telecom", "wireless", "cable", "satellite")):
            return "Technology & Communications"
        return "Communication Services"

    if s == "Health Care":
        return "Healthcare"

    if s in {
        "Energy",
        "Utilities",
        "Real Estate",
        "Financials",
        "Consumer Staples",
        "Materials",
        "Technology",
        "Healthcare",
    }:
        return s

    # Dow tables often expose an industry but not a GICS sector. Use conservative
    # keyword routing only when the sector field is unavailable.
    if not s:
        if any(k in sub for k in ("software", "semiconductor", "technology", "computer")):
            return "Technology"
        if any(k in sub for k in ("bank", "financial", "insurance", "payments")):
            return "Financials"
        if any(k in sub for k in ("pharma", "health", "biotech", "medical")):
            return "Healthcare"
        if any(k in sub for k in ("oil", "gas", "energy")):
            return "Energy"
        if any(k in sub for k in ("retail", "restaurant", "apparel", "consumer")):
            return "Consumer Discretionary"
        if any(k in sub for k in ("food", "beverage", "household")):
            return "Consumer Staples"
        if any(k in sub for k in ("telecom", "media", "entertainment")):
            return "Communication Services"
        if any(k in sub for k in ("chemical", "materials")):
            return "Materials"
    return s or "Industrials"


def _clean_ticker(value: object) -> str:
    return str(value).upper().strip().replace(".", "-")


def _read_tables(url: str) -> list[pd.DataFrame]:
    resp = requests.get(url, headers=HTTP_HEADERS, timeout=30)
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text))


def _pick_table(tables: Iterable[pd.DataFrame], required: set[str]) -> pd.DataFrame:
    for table in tables:
        normalized = {_normalize_col(c) for c in table.columns}
        if required.issubset(normalized):
            return table.copy()
    raise RuntimeError(f"No table contains required columns: {sorted(required)}")


def _normalize_col(col: object) -> str:
    return str(col).strip().lower().replace(" ", "_").replace("-", "_")


def _standardize(
    df: pd.DataFrame,
    index_name: str,
    ticker_candidates: Iterable[str],
    company_candidates: Iterable[str],
    sector_candidates: Iterable[str],
    industry_candidates: Iterable[str],
) -> pd.DataFrame:
    normalized = {_normalize_col(c): c for c in df.columns}

    def first(candidates: Iterable[str]) -> object | None:
        for candidate in candidates:
            if candidate in normalized:
                return normalized[candidate]
        return None

    ticker_col = first(ticker_candidates)
    company_col = first(company_candidates)
    sector_col = first(sector_candidates)
    industry_col = first(industry_candidates)
    if ticker_col is None or company_col is None:
        raise RuntimeError(f"{index_name} table missing ticker/company columns: {list(df.columns)}")

    out = pd.DataFrame(
        {
            "ticker": df[ticker_col].map(_clean_ticker),
            "company_name": df[company_col].astype(str).str.strip(),
            "source_sector": df[sector_col].astype(str).str.strip() if sector_col is not None else "",
            "industry_group": df[industry_col].astype(str).str.strip() if industry_col is not None else "",
            "index_membership": index_name,
        }
    )
    out = out[(out["ticker"].str.len() > 0) & (out["ticker"] != "NAN")]
    out = out.drop_duplicates("ticker").reset_index(drop=True)
    out["sector_group"] = out.apply(
        lambda r: map_sector(r["source_sector"], r["industry_group"]), axis=1
    )
    return out


def fetch_sp500() -> pd.DataFrame:
    tables = _read_tables(SOURCES["S&P 500"])
    table = _pick_table(tables, {"symbol", "security"})
    return _standardize(
        table,
        "S&P 500",
        ticker_candidates=["symbol"],
        company_candidates=["security"],
        sector_candidates=["gics_sector"],
        industry_candidates=["gics_sub_industry", "gics_sub_industry"],
    )


def fetch_nasdaq100() -> pd.DataFrame:
    tables = _read_tables(SOURCES["Nasdaq 100"])
    table = _pick_table(tables, {"ticker", "company"})
    return _standardize(
        table,
        "Nasdaq 100",
        ticker_candidates=["ticker", "symbol"],
        company_candidates=["company", "security"],
        sector_candidates=["gics_sector"],
        industry_candidates=["gics_sub_industry", "sub_industry"],
    )


def fetch_dow() -> pd.DataFrame:
    tables = _read_tables(SOURCES["Dow"])
    table = _pick_table(tables, {"symbol", "company"})
    return _standardize(
        table,
        "Dow",
        ticker_candidates=["symbol", "ticker"],
        company_candidates=["company"],
        sector_candidates=["gics_sector", "sector"],
        industry_candidates=["industry", "gics_sub_industry"],
    )


def load_legacy_ratings() -> pd.DataFrame:
    if not LEGACY_UNIVERSE.exists():
        return pd.DataFrame(columns=["ticker", "rating_seed", "rating_source"])
    df = pd.read_csv(LEGACY_UNIVERSE)
    keep = [c for c in ("ticker", "rating_seed", "rating_source") if c in df.columns]
    df = df[keep].copy()
    df["ticker"] = df["ticker"].map(_clean_ticker)
    return df.drop_duplicates("ticker")


def main() -> None:
    frames = []
    for name, func in [
        ("S&P 500", fetch_sp500),
        ("Nasdaq 100", fetch_nasdaq100),
        ("Dow", fetch_dow),
    ]:
        df = func()
        print(f"Pulled {len(df):>3} constituents from {name}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    membership = (
        combined.groupby("ticker")["index_membership"]
        .apply(lambda s: "; ".join(sorted(set(s))))
        .rename("index_memberships")
        .reset_index()
    )
    combined = combined.sort_values(
        ["ticker", "index_membership"], ascending=[True, True]
    ).drop_duplicates("ticker", keep="first")
    combined = combined.drop(columns=["index_membership"]).merge(membership, on="ticker", how="left")

    legacy = load_legacy_ratings()
    out = combined.merge(legacy, on="ticker", how="left")
    out["rating_seed"] = out["rating_seed"].fillna("NR (to derive from financials)")
    out["rating_source"] = out["rating_source"].fillna(
        "default - no public rating available; derive from leverage/margin post-SEC pull"
    )

    out = out.sort_values("ticker").reset_index(drop=True)
    out.insert(0, "id", range(1, len(out) + 1))
    out = out[
        [
            "id",
            "company_name",
            "ticker",
            "sector_group",
            "industry_group",
            "rating_seed",
            "rating_source",
            "index_memberships",
        ]
    ]

    print("\nIndex membership distribution:")
    print(out["index_memberships"].value_counts().to_string())
    print("\nSector_group distribution:")
    print(out["sector_group"].value_counts().to_string())

    out.to_csv(UNIVERSE_OUT, index=False)
    print(f"\nWrote {len(out)} unique companies -> {UNIVERSE_OUT}")


if __name__ == "__main__":
    main()
