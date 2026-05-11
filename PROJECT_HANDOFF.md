# Surveillance Tool — Project Handoff

> **Read this first.** This is the single source of truth for an LLM (or human) picking up the project. It captures the architecture, formulas, files, major decisions, dashboard behavior, and known gaps **as of 2026-05-10**.

---

## 1. What this project is

A public-data implementation of the "Surveillance Tool" credit risk framework. It pulls SEC EDGAR + market data for the **S&P 500 + Nasdaq 100 + Dow Jones Industrial Average** union (516 unique tickers in the latest dashboard input), computes 9 indicator scores + an absolute weighted **Surveillance Score** + an action **Tier** per company, and renders an interactive Streamlit dashboard.

Two reference specs drive the design:

1. `surveillance_tool_project_documentation.md` — original "Surveillance Tool" project spec (Sections 1–18).
2. **EWIF v1.0 framework** (provided by user as photos of a Word doc, not present in repo). Sections referenced throughout: I (philosophy), II (cross-factor examples), III (9 factors), V.A (weights), V.B (normalization), V.C (alert tiers), VI (fraud overlay incl. Beneish), VII (PD/LGD/EAD overlay), Appendix (data sources by factor).

When the two specs disagreed, we noted it in code comments and kept the 9-factor weight structure while aligning the dashboard score convention to **higher score = better credit quality / lower risk**. As of 2026-05-10, the displayed Surveillance Score is the weighted average of absolute indicator scores. Final portfolio-rank calibration was removed after the complementary redesign proposal advised separating absolute risk signal from peer-relative diagnostics.

---

## 2. Repo layout

```
SV/
├── PROJECT_HANDOFF.md                              ← this file
├── surveillance_tool_project_documentation.md     ← original v1 spec (~1200 lines)
│
├── ewif_public_data_prep_package/                 ← DATA PREP (no scoring lives here)
│   ├── build_company_universe.py                  ← scrapes S&P 500 + Nasdaq 100 + Dow lists from Wikipedia
│   ├── build_sp500_universe.py                    ← compatibility wrapper for build_company_universe.py
│   ├── build_surveillance_data.py                 ← main extractor: SEC + yfinance + (optional) GDELT
│   ├── build_surveillance_history.py              ← quarterly SEC history from cache
│   ├── build_surveillance_store.py                ← quarterly scores + SQLite structured store + assumption audit
│   ├── build_ewif_public_data.py                  ← LEGACY v1; do not modify (kept for reference)
│   ├── ewif_company_universe_major_indices.csv    ← active universe (S&P 500 + Nasdaq 100 + Dow)
│   ├── ewif_company_universe_sp500.csv            ← prior S&P 500 universe (kept for reference)
│   ├── ewif_company_universe_us_100.csv           ← legacy demo universe (kept for reference)
│   ├── cache_sec/                                 ← SEC companyfacts JSONs (one per CIK)
│   └── output/
│       ├── surveillance_dashboard_input.csv       ← 516 rows (one company per row, latest snapshot)
│       ├── surveillance_raw_financials.csv        ← raw SEC + market values
│       ├── surveillance_data_diagnostics.csv      ← long-format flags (stale / assumption / source tag)
│       ├── surveillance_quarterly_history.csv     ← compact quarterly SEC history
│       ├── surveillance_quarterly_scores.csv      ← 2-year quarterly company scores
│       └── surveillance_store.sqlite              ← structured SQLite store
│
├── requirements.txt                                ← minimal runtime deps for prep + dashboard
│
└── surveillance_dashboard/                         ← UI + SCORING
    ├── scoring.py                                  ← all formulas (9 indicators, Beneish, velocity, tiers)
    └── app.py                                      ← Streamlit UI, 5 tabs + drilldown quarterly/fraud + stress/memo/export tools
```

**Architectural rule we've been enforcing:** the prep scripts only pull / clean / fill assumptions. They do **not** score. All scoring lives in `scoring.py` so it can be unit-tested and reviewed independently.

---

## 3. End-to-end data flow

```
┌────────────────┐    ┌──────────────────────────┐    ┌──────────────────────┐
│ Wikipedia      │───▶│ build_company_universe.py│───▶│ universe_major_*.csv │
└────────────────┘    └──────────────────────────┘    └──────────────────────┘
                                                                 │
                                                                 ▼
┌──────────────┐  ┌──────────────────────────┐  ┌────────────────────────────────┐
│ SEC EDGAR    │─▶│                          │─▶│ surveillance_dashboard_input   │
│ (cached)     │  │                          │  │ surveillance_raw_financials    │
├──────────────┤  │ build_surveillance_data  │  │ surveillance_data_diagnostics  │
│ yfinance     │─▶│  (latest snapshot only)  │  └────────────────────────────────┘
├──────────────┤  │                          │
│ GDELT (off)  │─▶│                          │
└──────────────┘  └──────────────────────────┘
                       │
                       ▼ uses same SEC cache
                  ┌──────────────────────────────┐    ┌────────────────────────────┐
                  │ build_surveillance_history.py│───▶│ surveillance_quarterly_*   │
                  │  (quarterly long-format SEC) │    └────────────────────────────┘
                  └──────────────────────────────┘
                                                                 │
                                                                 ▼
                  ┌──────────────────────────────┐    ┌────────────────────────────┐
                  │ build_surveillance_store.py  │───▶│ quarterly_scores.csv       │
                  │  (2024Q1-2026Q1 grid + DB)  │    │ surveillance_store.sqlite  │
                  └──────────────────────────────┘    └────────────────────────────┘
                                                                 │
                                                                 ▼
                                            ┌─────────────────────────────────────────┐
                                            │ scoring.py                              │
                                            │  • compute_beneish_m_score(history)     │
                                            │  • compute_velocity_signals(history)    │
                                            │  • compute_scores(input, history)       │
                                            │    → 9 indicators + weighted score      │
                                            │      + action tier                      │
                                            │  • explanation_for_company(row)         │
                                            └─────────────────────────────────────────┘
                                                                 │
                                                                 ▼
                                                     ┌───────────────────────────┐
                                                     │ app.py  (Streamlit)       │
                                                     │  Page 1 Executive Overview│
                                                     │  Page 2 Drilldown         │
                                                     │  Page 3 Stress Testing    │
                                                     │  Page 4 Methodology       │
                                                     │  Page 5 Data Coverage     │
                                                     │  Fraud detail route       │
                                                     │  Scenario editor          │
                                                     │  Peer comparison          │
                                                     │  Memo + CSV/PDF export    │
                                                     └───────────────────────────┘
```

**Caching:**
- SEC `companyfacts` JSON is cached in `cache_sec/CIK*_companyfacts.json` and re-read on every run. To force re-fetch, delete the file.
- yfinance pulls fresh every run (batched, roughly 30-60s depending on universe size).
- Streamlit uses `@st.cache_data` on `load_data`; bouncing the server clears it.

---

## 4. How to run

```bash
# 0. install dependencies if needed
cd /Users/zeyangyu/Desktop/SV
python -m pip install -r requirements.txt

# 1. (optional) refresh the major-index universe
cd ewif_public_data_prep_package
python build_company_universe.py

# 2. main data prep — SEC cache hits, yfinance fresh.  ~35-60s for current universe.
RUN_GDELT=0 python build_surveillance_data.py

# 3. quarterly history — pure transform of cache, no network.
python build_surveillance_history.py

# 4. build quarterly scores + structured SQLite store.
# Defaults to the complete 2024Q1-2026Q1 scoring grid.
python build_surveillance_store.py

# 5. dashboard
cd ../surveillance_dashboard
python -m streamlit run app.py --server.port 8501 --server.headless true
# → http://localhost:8501
```

If port 8501 is occupied, run the dashboard on another port, e.g. `--server.port 8503`. During the latest work session, the verified live URL was `http://localhost:8503`.

**Environment variables you can set:**
- `RUN_GDELT=0` — disable news pull (default; GDELT API has been unreliable)
- `UNIVERSE_FILE=path/to/universe.csv` — override the universe (defaults to S&P 500 + Nasdaq 100 + Dow)
- `HISTORY_YEARS=3` — SEC cache history extraction window; leave at default for 2024Q1-2026Q1 scoring because TTM/Beneish needs lookback
- `SCORE_START_PERIOD=2024Q1`, `SCORE_END_PERIOD=2026Q1` — quarterly score grid bounds used by `build_surveillance_store.py`
- `RUN_QUARTERLY_MARKET=1` — pull quarter-end yfinance market metrics for quarterly scores
- `SEC_USER_AGENT="My Name myemail@example.com"` — required by SEC; defaults to a placeholder

---

## 5. The 9 indicators — actual formulas as implemented

All scores are 0–100 where **higher = better credit quality / lower risk**. This was changed on 2026-05-04 after manager feedback that scores and sub-scores should be directionally consistent and easier to explain.

Important 2026-05-10 scoring change: the displayed `surveillance_score` is no longer calibrated to the current cross-sectional portfolio range. It equals the weighted average of the 9 indicator scores, which are now wider absolute threshold scores for core financial and market measures.

```python
raw_surveillance_score = sum(score_i * weight_i for i in 9 indicators)
surveillance_score = raw_surveillance_score
```

Core financial and market sub-scores now use `threshold_score(series, breakpoints)`, a piecewise-linear absolute mapping from raw ratio to 0-100 quality score. Selected diagnostics still use cross-sectional percentiles where the proposal treats them as relative indicators. In the latest smoke test, AAPL scored 78.1, MSFT 78.7, NVDA 84.8, and the total latest-score range was 42.6-91.0 with 44 companies at or above 80.

`percentile_score(series, higher_is_risk=True)` always returns a quality score:
- if the raw input is riskier when higher, higher raw values receive lower scores.
- if the raw input is stronger when higher, higher raw values receive higher scores.
- missing values default to 50 before any higher-level assumptions.

### Composite weights (`INDICATOR_WEIGHTS` in `scoring.py`)

| Indicator | Weight | Source |
|---|---|---|
| Facility & Liquidity | 17% | EWIF v1.0 V.A |
| Financial Performance | 15% | EWIF v1.0 V.A (was 17% — bug fixed) |
| Behavioral & Payment | 8% | EWIF v1.0 V.A |
| Market-Implied Risk | 10% | EWIF v1.0 V.A |
| News & Sentiment | 10% | EWIF v1.0 V.A |
| Accounting Integrity | 15% | EWIF v1.0 V.A |
| Connectivity & Contagion | 10% | EWIF v1.0 V.A |
| Governance & Strategic Discipline | 10% | EWIF v1.0 V.A |
| Collateral & Recovery | 5% | EWIF v1.0 V.A |
| **Total** | **100%** | enforced by `assert` |

### 5.1 Facility & Liquidity (proxy)

```python
facility_liquidity_score = bounded(
    0.55 * financial_performance_score
  + 0.20 * score_cash
  + 0.15 * score_current_ratio
  + 0.10 * sector_quality_score
  + 10,
  0, 100
)
facility_liquidity_risk_proxy = 100 - facility_liquidity_score
```

Real spec wants utilization velocity, covenant headroom, deposit volatility — all internal-bank data. Public proxy uses repayment capacity, cash cushion, current liquidity, and sector quality. Rating bias was removed on 2026-05-10 because the available rating buckets were synthetic / inaccurate.

### 5.2 Financial Performance

```python
FP_score = mean(
    score_leverage,             # lower liabilities/assets = stronger
    score_debt,                 # lower debt/assets = stronger
    score_net_debt_ebitda,      # lower net debt / EBITDA = stronger
    score_interest_coverage,    # higher EBITDA / interest = stronger
    score_cash,                 # higher cash/assets = stronger
    score_margin,               # higher net margin = stronger
    score_fcf,                  # higher FCF/assets = stronger
    score_fcf_sustainability,   # see sub-index below
    score_current_ratio,        # higher current ratio = stronger
)
```

Additional repayment-capacity fields now derived in `compute_scores`:

```python
ebitda_proxy = net_income + interest_expense.fillna(0) + depreciation.fillna(0)
net_debt = total_debt - cash
net_debt_to_ebitda = net_debt / ebitda_proxy        # if EBITDA > 0; net debt with nonpositive EBITDA scores 0
ebitda_interest_coverage = ebitda_proxy / interest_expense
```

FCF Sustainability sub-index from `compute_fcf_sustainability(history_df)`:

```python
score_fcf_sustainability = mean(
    threshold_score(negative_fcf_quarters_l4, [(0,100),(1,75),(2,50),(3,25),(4,0)]),
    threshold_score(fcf_ebitda_conversion, [(-0.5,0),(0,40),(0.4,65),(0.8,85),(1.2,100)]),
    threshold_score(cash_burn_to_cash, [(0,100),(0.25,80),(0.5,60),(1,35),(2,10)]),
)

negative_fcf_quarters_l4 = count of quarters where (CFO - capex) < 0 in latest 4 quarters
fcf_ebitda_conversion = TTM FCF / EBITDA proxy
cash_burn_to_cash = abs(TTM FCF) / cash when TTM FCF < 0, else 0
```

**Remaining spec gaps:** EBIT/(Interest+Rent), ROIC−WACC, Growth Quality sub-index, sector-relative scoring.

### 5.3 Behavioral & Payment (proxy)

```python
behavioral_payment_score = bounded(
    0.55 * financial_performance_score
  + 0.20 * accounting_integrity_score
  + 0.15 * data_quality_score
  + 0.10 * market_implied_risk_score
  + 8,
  0, 100
)
behavioral_payment_risk_proxy = 100 - behavioral_payment_score
```

Real spec wants payment delay days, waivers, RM contact frequency — internal data only.

### 5.4 Market-Implied Risk

```python
score = mean(
    threshold_score(ret_3m,      [(-0.30,0),(-0.15,25),(0,50),(0.10,75),(0.25,100)]),
    threshold_score(vol_3m,      [(0,100),(0.15,85),(0.25,65),(0.40,40),(0.60,15),(1,0)]),
    threshold_score(drawdown_6m, [(-0.60,0),(-0.40,20),(-0.25,45),(-0.10,70),(0,100)]),
)
```

Inputs from yfinance close-price series. **Spec gap:** missing CDS spread, bond OAS, short interest, options skew, analyst revisions. CDS/OAS need paid feeds; short interest is in `yf.Ticker(...).info["shortPercentOfFloat"]` and can be added cheaply.

### 5.5 News & Sentiment

No longer held flat at 50. GDELT remains disabled (`RUN_GDELT=0`) because it was slow/unreliable, but the factor now uses a documented public-data assumption when real news data is unavailable.

```python
if has_news_variation:
    news_sentiment_score = (
        0.4 * score_news_mentions_30d
        + 0.6 * score_risk_news_mentions_30d
    )
else:
    news_sentiment_score = (
        0.50 * market_implied_risk_score
        + 0.30 * financial_performance_score
        + 0.20 * sector_quality_score
    )
    news_sentiment_assumption_used = True
```

### 5.6 Accounting Integrity

```python
score_sloan = percentile_score(accrual_proxy, higher_is_risk=True)
score_beneish = percentile_score(beneish_m, higher_is_risk=True)

if beneish_m available:
    accounting_integrity_score = mean(score_sloan, score_beneish)
else:
    accounting_integrity_score = (
        0.50 * financial_performance_score
        + 0.30 * data_quality_score
        + 0.20 * score_sloan
    )
    accounting_integrity_assumption_used = True
```

This replaced the previous fixed-neutral Accounting score on 2026-05-04. Sloan accruals and Beneish are still shown in drilldown; when Beneish is unavailable, the fallback assumption uses Financial Performance, Data Quality, and Sloan accruals so the factor remains directional without overreacting to Sloan alone.

#### Beneish M-Score (8-variable, EWIF v1.0 VI.B)

```
M = -4.84
    + 0.92  · DSRI    Days Sales in Receivables Index
    + 0.528 · GMI     Gross Margin Index            (prior margin / current margin)
    + 0.404 · AQI     Asset Quality Index           ((1-(CA+PPE)/TA) at t / at t-1)
    + 0.892 · SGI     Sales Growth Index            (Sales_t / Sales_t-1)
    + 0.115 · DEPI    Depreciation Index            (prior dep-rate / current dep-rate)
    - 0.172 · SGAI    SG&A Index                    ((SGA/Sales)_t / (SGA/Sales)_{t-1})
    + 4.679 · TATA    Total Accruals to Total Assets ((NI - CFO) / TA)
    - 0.327 · LVGI    Leverage Index                ((Liab/Assets)_t / (Liab/Assets)_{t-1})
```

- Period t = latest available TTM/instant snapshot; t−1 = closest snapshot ~365 days earlier (±60 days).
- Missing components are neutralized (DSRI/GMI/AQI/SGI/DEPI/SGAI/LVGI → 1.0; TATA → 0.0) so the M is still meaningful for partial-data companies.
- M > **−1.78** historically signals elevated earnings-management risk (Beneish 1999).
- **Coverage note:** Beneish coverage varies by universe and history availability. In the 2026-05-04 run, 166/503 tickers got a Beneish M-Score; re-run `compute_scores(...)` for current coverage.

#### Why Beneish is still retained
The framework names Beneish explicitly. It remains valuable as a diagnostic and in the memo/drilldown, and now affects Accounting Integrity when available.

### 5.7 Connectivity & Contagion (proxy)

```python
sector_density = count of tickers in same sector_group
connectivity_contagion_score = bounded(
    0.45 * market_implied_risk_score
  + 0.25 * sector_quality_score
  + 0.20 * news_sentiment_score
  + 0.10 * data_quality_score
  + 5,
  0, 100
)
```

Real spec wants ownership graph, supply-chain links, IAFSI — needs Orbis/GLEIF/Panjiva. The current public proxy blends market stress, sector context, news, and data confidence while still exposing `sector_density_count` for context.

### 5.8 Governance & Strategic Discipline (proxy)

```python
governance_discipline_score = bounded(
    0.45 * accounting_integrity_score
  + 0.25 * news_sentiment_score
  + 0.20 * financial_performance_score
  + 0.10 * data_quality_score
  + 5,
  0, 100
)
governance_discipline_risk_proxy = 100 - governance_discipline_score
```

Real spec wants insider trading (SEC Form 4 — public!), M&A premium, goodwill creation. Insider trading is the obvious next add.

### 5.9 Collateral & Recovery (proxy)

```python
collateral_recovery_score = bounded(
    0.40 * score_debt
  + 0.25 * score_net_debt_ebitda
  + 0.20 * score_leverage
  + 0.15 * financial_performance_score
  + 5,
  0, 100
)
collateral_recovery_risk_proxy = 100 - collateral_recovery_score
```

Real spec needs LTV, lien position, intercreditor — all internal.

### 5.10 Composite

```python
raw_surveillance_score = sum(score_i * weight_i for i in 9 indicators)
surveillance_score     = raw_surveillance_score
ewif_score             = surveillance_score
```

---

## 6. Tier classification

```python
TIER1_LEVEL = 40.0
TIER2_LEVEL = 55.0
TIER1_INDICATOR_SEVERE = 25.0   # score <= 25 on any indicator
TIER1_MIN_SEVERE_COUNT = 2

if score < TIER1_LEVEL or count(indicators <= 25) >= 2:
    tier = "Tier 1"   # high-conviction deterioration  → mandatory review
elif score < TIER2_LEVEL and velocity_trigger:
    tier = "Tier 2"   # accelerating stress           → targeted perturbation
elif velocity_trigger:
    tier = "Tier 3"   # monitoring queue              → confirmatory evidence
else:
    tier = "None"
```

The dashboard now uses the framework-friendly convention: **higher score = better credit quality / lower risk**. No risk-up threshold inversion is used anymore.

### Velocity trigger (count >= 2)

Computed YoY from `surveillance_quarterly_history.csv`:

1. TTM revenue contracting (Δ < 0)
2. Net margin compressing > 2pp
3. Debt/Assets rising > 5pp

Tier counts changed after the 2026-05-10 absolute-threshold scoring update and rating removal. Re-run `compute_scores(...)` or inspect `latest_scores` in `surveillance_store.sqlite` for the current counts.

### Continuous alert color (kept from original spec, runs in parallel)

| Surveillance Score | Alert |
|---|---|
| 55–100 | Green |
| 45–55 | Yellow |
| 35–45 | Orange |
| 0–35 | Red |

The dashboard shows **both** the color (continuous risk level) and the tier badge (action urgency).

---

## 7. Data prep specifics

### 7.1 SEC EDGAR — XBRL tag map

`TAG_MAP` in `build_surveillance_data.py` maps each business field to a list of XBRL synonyms; `select_latest_fact` picks the latest filed fact across them, preferring 10-K. Critical convention: per-metric we lock in **one** synonym per company (`_select_best_tag` in the history extractor) to avoid mixing e.g. `Revenues` + `NoninterestIncome` for JPM.

**Fields extracted (TAG_MAP keys):**
revenue, net_income, assets, liabilities, equity, cash, cfo, capex, interest_expense, current_assets, current_liabilities, accounts_receivable, inventory, accounts_payable, long_term_debt, short_term_debt, **cogs, ppe_net, depreciation, sga** (last four added 2026-05-02 for Beneish).

### 7.2 Quarterly history derivation

`build_surveillance_history.py` distinguishes:

- **Instant facts** (balance sheet) → keep every quarter-end snapshot, dedup to latest `filed`.
- **Duration facts** (P&L, cash flow) → derive discrete quarterly series:
  - Use ~90-day records directly when present.
  - When only YTD records exist (common for 10-Q): subtract H1−Q1, 9M−H1, etc.
  - **Q4 = FY annual − (Q1 + Q2 + Q3)** indexed by **calendar `period_end`** (NOT fiscal year tag — those are unreliable across companies).
- **TTM** = rolling 4-quarter sum of the discrete quarterlies at each quarter end.
- **Ratios** = point-in-time balance sheet ÷ TTM income flow at each quarter end.

Runtime output is compact long format: `ticker, period_end, metric, value, value_type ∈ {instant, quarterly, ttm, ratio}`.

The extraction code still tracks source tags internally; compact runtime CSV omits source metadata to keep file size manageable.

### 7.3 Stale data flag (550-day rule)

`detect_stale_facts` in `build_surveillance_data.py`:
- For each company, find the most-recent fact period_end across all SEC fields → `latest_period_end`.
- Any field with period_end older than `latest_period_end - 550 days` is flagged stale.
- `stale_fact_count` and per-field `_is_stale` columns added to the dashboard input.

### 7.4 Sector-level assumption fill

`SECTOR_ASSUMPTIONS` in `build_surveillance_data.py` — one dict per sector with reasonable defaults for the 7 ratios (`leverage_proxy`, `debt_to_assets`, `cash_to_assets`, `net_margin`, `fcf_to_assets`, `accrual_proxy`, `current_ratio`).

When a ratio cannot be computed from clean SEC data, `fill_assumptions` fills it from the sector dict and sets `<ratio>_is_assumption = True`. The dashboard surfaces this on Page 5 Data Coverage.

### 7.5 Data quality score

```python
data_quality_score = 0.7 * (1 - assumption_count / num_ratios) * 100
                   + 0.3 * (1 - stale_count / num_sec_fields) * 100
data_quality_note  = human-readable summary based on count buckets
```

### 7.6 Ratings removed from scoring and UI

Earlier builds carried `rating_seed`, `rating_bucket`, and `rating_source`, and derived synthetic ratings for unrated companies. After review, these were judged inaccurate and removed from the dashboard experience and scoring path on 2026-05-10.

Current behavior:
- `scoring.py` does not use rating buckets or rating bias.
- `compute_scores` drops `rating_seed`, `rating_bucket`, and `rating_source` from the scored dataframe.
- The Streamlit UI no longer shows rating filters, rating columns, rating methodology tables, or rating in the company header / memo.
- The raw input CSV may still carry old rating columns for backward compatibility with data prep, but they should be ignored unless a validated internal/vendor rating source is connected.

---

## 8. Output CSV reference

### `surveillance_dashboard_input.csv` (516 rows)

Per-company latest snapshot; one row per ticker. Key column groups:

| Group | Columns |
|---|---|
| Identity | `ticker`, `company_name`, `sector_group`, `industry_group`, `cik`, `sec_available` plus legacy rating columns that are ignored by scoring |
| Raw SEC | `revenue`, `net_income`, `assets`, `liabilities`, `equity`, `cash`, `cfo`, `capex`, `interest_expense`, `current_assets`, `current_liabilities`, `accounts_receivable`, `inventory`, `accounts_payable`, `long_term_debt`, `short_term_debt`, `cogs`, `ppe_net`, `depreciation`, `sga`, `total_debt`, `fcf` |
| SEC provenance | for each of the above: `<field>_tag`, `<field>_form`, `<field>_period_end`, `<field>_filed`, `<field>_is_stale` |
| Ratios | `leverage_proxy`, `debt_to_assets`, `cash_to_assets`, `net_margin`, `fcf_to_assets`, `accrual_proxy`, `current_ratio`, plus `<ratio>_is_assumption` flags |
| Market | `last_close`, `ret_1m`, `ret_3m`, `vol_3m`, `drawdown_6m` |
| News | `news_mentions_30d`, `risk_news_mentions_30d`, `news_data_available` |
| QA | `stale_fact_count`, `data_quality_score`, `data_quality_note` |

The dashboard layer (`compute_scores`) appends:
- `score_*` per-component fields
- `financial_performance_score` … `collateral_recovery_score` (the 9 indicators)
- `ebitda_proxy`, `net_debt`, `net_debt_to_ebitda`, `ebitda_interest_coverage`
- `negative_fcf_quarters_l4`, `fcf_ebitda_conversion`, `cash_burn_to_cash`, `score_fcf_sustainability`
- `sector_base_risk`, `sector_quality_score`, `sector_density_count`, `sector_density_score` for formula transparency
- `beneish_m`, `beneish_dsri` … `beneish_lvgi`, `beneish_period_t`, `beneish_period_t_minus_1`
- `velocity_signal_count`, `velocity_trigger`
- `raw_surveillance_score`, `surveillance_score`, `ewif_score`, `alert_level`, `alert_tier`
- `top_risk_drivers`, `top_driver_objects`

### `surveillance_quarterly_history.csv` (181,009 rows in latest build)

`ticker, period_end, metric, value, value_type`

Used by: Beneish diagnostics, FCF sustainability, velocity signals, and historical SEC-derived ratio work. As of 2026-05-10 it also feeds quarterly scoring.

The file is compacted to runtime columns and selected metrics. In the latest build it used `HISTORY_YEARS=3` and extracted history for 516/516 companies from local SEC cache.

### `surveillance_quarterly_scores.csv` (4,644 rows in latest build)

Built by `build_surveillance_store.py`. One row per company per calendar-quarter bucket for the complete 2024Q1-2026Q1 grid: 516 companies x 9 quarters. It uses SEC quarterly / TTM financial data from `surveillance_quarterly_history.csv`, quarter-end yfinance market metrics, and documented assumptions where values are unavailable.

Important fields:
- `ticker`, `company_name`, `sector_group`, `industry_group`
- `score_period` (calendar quarter bucket such as `2025Q4`)
- `period_end` (company-specific SEC fiscal quarter end)
- `raw_surveillance_score`
- `surveillance_score`
- `alert_level`, `alert_tier`
- the 9 indicator scores and component `score_*` columns

Quarterly scores use the same absolute scoring curves as the latest snapshot. `score_period` groups off-calendar fiscal filers like Apple into calendar-quarter buckets for charting and latest-quarter tables. Missing company-quarter financial inputs are filled from reported values, same-company latest values, sector/period medians, or deterministic ratio-based assumptions, and every filled field is marked with `<field>_is_assumption`.

### `surveillance_store.sqlite`

Structured SQLite store created by `build_surveillance_store.py`.

Tables:
- `companies`
- `latest_inputs`
- `latest_scores`
- `quarterly_history`
- `quarterly_inputs`
- `quarterly_scores`
- `data_diagnostics`
- `assumption_audit`

`assumption_audit` stores one row per company/field assumption with:
`period_type`, `score_period`, `ticker`, `field_name`, `reported_value`, `final_value_used`, `assumption_used`, `assumption_method`, `assumption_source`, `peer_group_used`, `peer_count`, `assumption_confidence`, `assumption_formula`, `affected_factor`, and `affected_ratio`.

Latest smoke test:
- `companies`: 516 rows
- `latest_scores`: 516 rows
- `quarterly_inputs`: 4,644 rows
- `quarterly_scores`: 4,644 rows
- quarterly companies: 516
- quarterly period buckets: 9 (`2024Q1` through `2026Q1`)
- quarterly score range: 39.7-91.3, average 69.4, no null scores
- `assumption_audit`: 49,262 rows

### `surveillance_data_diagnostics.csv` (~2k rows, long format)

Per-company per-field flags: stale, assumption-filled, source tag chosen. Drives Page 5 Data Coverage.

### `surveillance_raw_financials.csv`

Pre-assumption-fill snapshot of the same columns as `surveillance_dashboard_input.csv` minus the score / scoring-derived columns. Useful for forensics ("what did SEC actually report before we filled?").

---

## 9. Dashboard structure and interaction model

| Page | Function |
|---|---|
| **1. Executive Overview** | KPI cards (count, avg score, Tier 1/2/3 counts, avg DQ), action tier distribution, sector × action tier heatmap focused on Tier 1/2/3, top-10 lowest-scoring table, sector average bar. |
| **2. Company Drilldown** | Header (score, alert chip, tier chip, velocity signals, DQ, stale facts), scenario editor, executive memo/export, gauge, 9-indicator bar, peer/industry comparison, score calculation audit, plain-English explanation, key financial metrics, market metrics, company quarterly score history, and compact Fraud & Accounting card with a detail-page button. |
| **3. Stress Testing** | Session-only stress templates (Credit downturn, Liquidity stress, Market shock, Sector contagion stress, Accounting/fraud stress), editable stress sliders, score/tier movement, indicator delta chart, PD/LGD/EAD overlay rationale, stress CSV export. |
| **4. Indicator Methodology** | Composite weights table, continuous alert thresholds, Tier 1/2/3 trigger table, per-indicator methodology cards (purpose, inputs, formula, data source, production roadmap). |
| **5. Data Coverage** | Per-field assumption / stale rates, per-company DQ score, sector-level coverage stats. |

Separate route: `?view=fraud&ticker=AAPL` opens the detailed Fraud & Accounting page for a ticker. The Company Drilldown card links to this route with `Open fraud detail`.

### 9.1 Sidebar filters

The left sidebar is now a management-facing filter panel:

- company search by ticker or company name
- sector and industry multi-selects
- alert level and action tier multi-selects
- surveillance score range
- data quality range
- max stale facts
- max assumption fills
- advanced filters:
  - velocity trigger: all / triggered only / not triggered
  - SEC available only
  - market data available only
  - scenario-edited names only
- reset filters button
- current-view count (`X of 516 companies selected` in the current build)

### 9.2 Company Drilldown additions

**Scenario editor**

- Lets the user override latest-snapshot raw values for the selected ticker. The raw fields are grouped under the 9 indicator headings and are initialized from the original/base latest-snapshot values. Proxy-only indicators show their original proxy score in an editable 0-100 box.
- Edits are stored in `st.session_state["scenario_overrides"]`; source CSVs are unchanged.
- After edits, `apply_scenario_overrides(raw_df, overrides)` updates the selected row, `_recompute_latest_snapshot_ratios` recalculates derived latest-snapshot ratios, and `compute_scores(...)` reruns on the copied dataframe.
- Editable raw fields include sector group, revenue, net income, assets, liabilities, cash, CFO, capex, interest expense, depreciation, current assets, current liabilities, accounts receivable, total debt, 3M return, 3M volatility, and 6M drawdown.
- Rating inputs are not editable and no longer feed scoring.

### 9.3 Stress Testing page

- Uses `STRESS_TEMPLATES` in `app.py` to apply session-only shocks to one selected ticker.
- Templates modify revenue, margin, cash, total debt, current liabilities, 3M return, volatility, and drawdown. Some templates also apply direct factor-score overlays for News, Connectivity, Accounting, and Governance.
- `_apply_stress_template(...)` recomputes the selected stressed company against the loaded universe; source CSVs and database tables are unchanged.
- `_overlay_rationale(...)` converts score and indicator deltas into management-facing PD/LGD/EAD review rationale. This is a qualitative stress-testing bridge, not an official regulatory capital calculator.

### 9.4 Fraud & Accounting detail route

- `compute_scores(...)` now emits:
  - `fraud_quality_subscore`
  - `fraud_risk_subscore` (100 - fraud quality; higher = more fraud/accounting risk)
  - `fraud_watch_flag`
- Fraud watch flag turns on when Beneish M-Score is above -1.78, Sloan accruals score is <=25, or Accounting Integrity score is <=35.
- The Company Drilldown shows a compact Fraud & Accounting card. `Open fraud detail` links to `?view=fraud&ticker=<ticker>`, which renders the detailed route with selected-company Sloan/Beneish/fallback details plus the top portfolio fraud watchlist.
- Scenario mode can be cleared per company or globally from the sidebar.

**Peer and industry comparison**

- Lives inside Company Drilldown because it answers “is this company strong or weak relative to peers?” in context.
- Peer group is `industry_group` excluding the selected ticker if at least 3 peers exist; otherwise it falls back to `sector_group`.
- Shows company score, peer average, quality rank, peer percentile, radar chart across 9 indicators, company-minus-peer delta chart, key ratio benchmark, and lowest-scoring peers table.

**Score calculation audit**

- Shows exact formula, origin data, raw input value, component sub-score, and role in formula.
- Includes an `Audit period` selector so the user can inspect `Current` or saved quarterly score periods such as `2026Q1`, `2025Q4`, etc. Historical periods come from `surveillance_quarterly_scores.csv`, and the audit shows a period-specific 9-indicator snapshot before the formula details.
- Uses a selection box for `Composite Score`, each of the 9 indicators, and the `FCF Sustainability Sub-score`.
- Formula text was simplified on 2026-05-04 to avoid showing raw `clip(...)`; dashboard copy says scores are bounded 0–100.
- Uses intermediate fields exposed from `scoring.py` (`sector_base_risk`, `sector_quality_score`, `sector_density_score`, etc.) so the displayed formula matches the actual score.

**Executive memo and scenario export**

- Generates a management-ready memo for the selected company.
- Memo includes score/tier, peer context, top risk drivers, key metrics, scenario impact, recommended management action, and public-data caveat.
- Export buttons:
  - `Download memo PDF` (implemented using matplotlib/PdfPages; no extra dependency)
  - `Download scenario CSV` (base value, scenario/current value, and change across summary fields, metrics, and indicators)
- Exports reflect the current selected row, including active scenario edits.

---

## 10. Decisions made and tradeoffs

| Decision | Reason |
|---|---|
| Remove final portfolio-rank calibration | The complementary redesign proposal says the Surveillance Score should be an absolute credit-quality signal, not a portfolio percentile. Compression was addressed by widening the underlying absolute threshold/proxy scores instead. |
| Add quarterly scoring + SQLite store | Manager review asked for company scores by quarter and a structured database for future company additions. Implemented `build_surveillance_store.py`, `surveillance_quarterly_scores.csv`, and `surveillance_store.sqlite`. |
| Remove rating from UI and scoring | Synthetic / stale rating buckets were not accurate enough. Ratings may remain in raw CSVs but are ignored by `compute_scores` and hidden from Streamlit. |
| Disable GDELT (`RUN_GDELT=0`) | API consistently returned NaN with 17–21s per-company latency. News now uses an explicit assumption from market, financial, and sector-quality scores when real news is unavailable. |
| Use Wikipedia for major-index lists | Free, current, includes usable constituent tables for S&P 500, Nasdaq 100, and Dow. S&P/Nasdaq provide GICS-style fields; Dow industry labels are keyword-mapped if no GICS sector is available. |
| Lock per-metric XBRL tag per company | Prevented data mixing (JPM `Revenues` vs `NoninterestIncome` was producing nonsense quarterly revenue). |
| Index quarterlies by calendar `period_end` not `fy` | SEC `fy` tags are inconsistent across companies; `period_end` is reliable. |
| Beneish neutralizes missing components to 1.0 (or 0.0 for TATA) | Avoids dropping 200+ companies that lack one of COGS / PP&E / Dep / SGA. Result is still a meaningful M dominated by available signals. |
| Higher score = better credit quality | Changed on 2026-05-04 after manager feedback. All sub-scores and composite scores now use the same quality-up direction. |
| Continuous color + Tier badge shown together | Color answers "how bad?", Tier answers "what action?". Both fit on a credit dashboard. |
| Scoring lives in `scoring.py`, not the raw data prep script | Easier to A/B different scoring methodologies without re-pulling data. `build_surveillance_store.py` imports `scoring.py` only to persist scored quarterly/latest tables. |
| Accounting Integrity no longer fixed at 50 | Changed on 2026-05-04. Beneish + Sloan drive the factor where available; otherwise a fallback assumption uses Financial Performance, Data Quality, and Sloan. |
| Removed standalone time-series page | User said Page 3 did not make sense for the management-facing tool. Historical data still supports scoring diagnostics but is not a top-level UI page. |
| Put peer comparison in Company Drilldown | Better management flow: score, drivers, peer context, and scenario impacts are all visible for the selected name without switching pages. |
| Scenario edits are session-only | Prevents accidental mutation of source CSVs. Users override raw latest-snapshot inputs, with editable proxy-score boxes for indicators that have no direct raw input in the editor. Formula scoring then recalculates the 9 indicators and composite score. Exports are explicit via CSV/PDF buttons. |
| Compact quarterly history CSV | User needed `surveillance_quarterly_history.csv` under 25 MB. It is now ~5.6 MB and future history builds write the compact schema. |

---

## 11. Open spec gaps (prioritized work backlog)

### Tier-1 implementable from data on hand

| Gap | Effort | Where to add |
|---|---|---|
| EBIT/Interest and EBIT/(Interest+Rent) | Easy/Medium | `scoring.py` Financial Performance; rent not currently pulled |
| Growth Quality sub-index (revenue growth z vs sector, EBITDA-vs-Revenue margin test) | Medium | needs sector grouping in history |
| ROIC-WACC | Medium | needs invested capital and market/WACC assumptions |
| Sector-relative scoring | Medium | replace broad S&P 500 percentiles for selected ratios with sector/industry percentiles |
| AR/Sales & Inventory/Sales divergence in Accounting Integrity | Easy | data already present |
| Cross-factor PD interaction (`PD_adj = PD_base × Growth_Excess × Burn × Leverage_Slope`) | Medium | depends on Growth + FCF Sustainability |

### Tier-2 needs additional public pulls

| Gap | Source |
|---|---|
| Insider trading (Form 4) for Governance | EDGAR Form 4 filings — public, free |
| Short interest % for Market-Implied Risk | yfinance `Ticker(...).info["shortPercentOfFloat"]` |
| Analyst revisions for Market-Implied Risk | yfinance `recommendationKey`, earnings estimates |
| Equity downside skew | yfinance options chain |

### Tier-3 needs paid feeds (out of scope for public POC)

CDS spreads, bond OAS, ownership graph (Orbis/GLEIF), supply chain (Panjiva), M&A premium (PitchBook), real S&P/Moody's/Fitch ratings.

---

## 12. Known limitations

1. **News indicator is assumption-based for everyone in the current run** because GDELT is disabled/unavailable. It now varies by company using market, financial, and sector-quality scores, but it is still not real news sentiment.
2. **Ratings are removed from scoring and UI.** Raw files may still carry legacy rating columns, but they should be ignored until a validated internal/vendor rating feed is connected.
3. **Accounting Integrity is assumption-heavy where Beneish is unavailable.** Coverage varies by history availability and XBRL fields; unavailable names use the fallback assumption.
4. **Proxies dominate 5 of 9 indicators** (Facility, Behavioral, Connectivity, Governance, Collateral). They lean on financial strength, market stress, accounting, news, sector context, and data confidence. In production these become real internal data.
5. **Some diagnostics are still cross-sectional percentiles.** Accounting, sector-density, and news-count diagnostics rank companies against the loaded universe where absolute public thresholds are not yet defined. Sector-relative diagnostics remain a reasonable next step.
6. **Velocity is a 3-signal heuristic**, not the rich "velocity overlay" described in spec III. Good enough for a tier trigger, not a velocity factor in itself.
7. **Scenario edits are session-only.** Raw-input edits affect latest-snapshot inputs only and do not rewrite quarterly history; proxy-score boxes affect only the current session's derived proxy indicator values. Formula scoring recalculates the composite, alert, tier, and weakest drivers for that session.
8. **The score is not an agency rating or absolute PD estimate.** It is an internal surveillance signal built from public data and documented assumptions.

---

## 13. How to talk to next-time LLM

When you start a new chat, paste this whole file plus the EWIF v1.0 doc images. Then a useful kickoff prompt is:

> "I'm continuing the Surveillance Tool project. Read PROJECT_HANDOFF.md to get state. The dashboard is at `surveillance_dashboard/`, data prep at `ewif_public_data_prep_package/`. The framework images are the formal spec; `surveillance_tool_project_documentation.md` is the older v1 spec we cherry-pick from. **The most important rule: scoring lives in `scoring.py`, never in the prep scripts. Current score convention is higher score = better credit quality / lower risk.** Today I want to [X]."

Common follow-ups that worked well in this session:
- "Audit current methods against [doc] — show alignment table, then propose a fix list."
- "Add [factor X] from [data Y]" — e.g. "add short interest from yfinance to Market-Implied Risk"
- "Sanity-check the latest run — show top-10 by [metric] and flag anything that looks wrong"
- "Re-run the full pipeline" — runs in <1 min total since SEC is cached

Anti-patterns to avoid:
- Don't ask the LLM to compute scores in `build_surveillance_data.py` — by design they live in `scoring.py`.
- Don't put scoring in `build_surveillance_data.py`; keep scoring in `scoring.py`.
- Don't assume all displayed formulas are production-grade; many indicators are public-data proxies until internal bank/vendor data is connected.
- Don't add new XBRL fields without also extending `build_surveillance_history.py` (otherwise time series will be missing them).
- Don't delete the `cache_sec/` folder unless you mean to — re-fetching all 511 CIKs from SEC takes ~3 minutes and they rate-limit.

---

## 14. Quick formula crib sheet

```
percentile_score(s, higher_is_risk=True)
    = 100 - rank_pct(s) * 100  if higher_is_risk
      else rank_pct(s) * 100
    # NaN → 50 (neutral)
    # Output convention: higher = better credit quality / lower risk

leverage_proxy   = liabilities / assets
debt_to_assets   = (long_term_debt + short_term_debt) / assets
cash_to_assets   = cash / assets
net_margin       = net_income / revenue
fcf              = cfo - capex
fcf_to_assets    = fcf / assets
accrual_proxy    = (net_income - cfo) / assets             # Sloan accruals
current_ratio    = current_assets / current_liabilities

ebitda_proxy = net_income + interest_expense.fillna(0) + depreciation.fillna(0)
net_debt = total_debt - cash
net_debt_to_ebitda = net_debt / ebitda_proxy
ebitda_interest_coverage = ebitda_proxy / interest_expense

score_fcf_sustainability = mean(
    percentile_score(negative_fcf_quarters_l4, higher_is_risk=True),
    percentile_score(fcf_ebitda_conversion, higher_is_risk=False),
    percentile_score(cash_burn_to_cash, higher_is_risk=True),
)

ret_3m, vol_3m, drawdown_6m  = standard yfinance close-price derivations

Beneish M = -4.84 + 0.92·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI
                  + 0.115·DEPI - 0.172·SGAI + 4.679·TATA - 0.327·LVGI
    DSRI = (AR/Sales)_t / (AR/Sales)_{t-1}
    GMI  = ((Rev-COGS)/Rev)_{t-1} / ((Rev-COGS)/Rev)_t
    AQI  = (1-(CA+PPE)/TA)_t / (1-(CA+PPE)/TA)_{t-1}
    SGI  = Sales_t / Sales_{t-1}
    DEPI = (Dep/(Dep+PPE))_{t-1} / (Dep/(Dep+PPE))_t
    SGAI = (SGA/Sales)_t / (SGA/Sales)_{t-1}
    TATA = (NI - CFO) / TA   at t
    LVGI = (Liab/TA)_t / (Liab/TA)_{t-1}
    Threshold: M > -1.78 = elevated manipulation risk

accounting_integrity_score =
    if Beneish available:
        mean(score_sloan_accruals, score_beneish)
    else:
        0.50 * financial_performance_score
      + 0.30 * data_quality_score
      + 0.20 * score_sloan_accruals

news_sentiment_score =
    if real news variation available:
        0.40 * score_news_mentions_30d + 0.60 * score_risk_news_mentions_30d
    else:
        0.50 * market_implied_risk_score
      + 0.30 * financial_performance_score
      + 0.20 * sector_quality_score

velocity_signal_count = sum of:
    1[ rev_t < rev_{t-4Q} ]
    1[ net_margin_t - net_margin_{t-4Q} < -0.02 ]
    1[ debt_to_assets_t - debt_to_assets_{t-4Q} > 0.05 ]
velocity_trigger = (velocity_signal_count >= 2)

raw_surveillance_score = sum_i(score_i * weight_i),  weights sum = 1.0
surveillance_score = raw_surveillance_score

alert_level (continuous color):
    [55,100] Green | [45,55) Yellow | [35,45) Orange | [0,35) Red

alert_tier (action urgency):
    score < 40                                       → Tier 1
    OR  count(indicator_score <= 25) >= 2            → Tier 1
    40 <= score < 55  AND velocity_trigger           → Tier 2
    score >= 55       AND velocity_trigger           → Tier 3
    else                                             → None
```

---

## 15. Recent change log

| Date | Change |
|---|---|
| 2026-05-10 | Dashboard default filters now show the full 516-company S&P 500 + Nasdaq 100 + Dow universe. Added an Index filter and made full-default filters non-restrictive so stale Streamlit session state does not collapse the view to S&P-only names. |
| 2026-05-10 | Removed broad cross-sectional percentile conversion from scoring sub-components. Sloan accruals, Beneish, sector quality, sector density, and news-count scoring now use absolute threshold curves; peer percentiles remain diagnostics only. |
| 2026-05-10 | Restarted data baseline around S&P 500 + Nasdaq 100 + Dow. Added `build_company_universe.py` and regenerated active universe with 516 unique companies and structured `index_memberships`. |
| 2026-05-10 | Rebuilt full data pipeline for 516 companies. Latest outputs: 516 latest-score rows, 181,009 quarterly history rows, complete 4,644-row quarterly score grid for `2024Q1`-`2026Q1`, and 49,262 assumption-audit rows. Quarterly score range is 39.7-91.3 with no null scores. |
| 2026-05-10 | Consolidated navigation: removed top-level Quarterly Scores and Fraud & Accounting tabs. Quarterly score history now appears inside Company Drilldown. Fraud & Accounting is a compact drilldown card with an `Open fraud detail` button to `?view=fraud&ticker=<ticker>`. |
| 2026-05-10 | Added dashboard Page 3 - Stress Testing with scenario templates, factor shock sliders, score/tier movement, indicator deltas, PD/LGD/EAD overlay rationale, and stress CSV export. |
| 2026-05-11 | Reworked Company Drilldown scenario editor: raw input overrides are grouped under the 9 indicator headings, fields initialize from original/base values, proxy-only indicators expose editable proxy-score boxes, and the old separate direct 9-indicator override section was removed. |
| 2026-05-11 | Added an `Audit period` selector to Company Drilldown score calculation audit, letting users switch between `Current` and saved quarterly periods like `2026Q1` or `2025Q4`. |
| 2026-05-10 | Added Fraud & Accounting diagnostics. `compute_scores` now emits `fraud_quality_subscore`, `fraud_risk_subscore`, and `fraud_watch_flag`; latest smoke test showed 148 fraud-watch names and 164 Beneish-covered names. |
| 2026-05-10 | Removed final portfolio-rank score calibration based on the complementary redesign proposal. Core financial/market sub-scores now use absolute threshold curves and public-data proxy indicators were widened. Latest smoke test after the major-index rebuild: AAPL 78.1, MSFT 78.7, NVDA 84.8, range 42.6-91.0, 44 names >=80. |
| 2026-05-10 | Added structured `assumption_audit` table to `surveillance_store.sqlite`, covering data-prep fallback flags, quarterly financial/market assumptions, and scoring fallbacks for News and Accounting Integrity. |
| 2026-05-10 | Regenerated `surveillance_quarterly_history.csv` with expanded compact history metric set so quarterly scoring has the needed ratios and raw values. |
| 2026-05-10 | Added `build_surveillance_store.py`, `surveillance_quarterly_scores.csv`, and `surveillance_store.sqlite`. The current build uses a complete 516-company x 9-quarter grid. |
| 2026-05-10 | Added quarterly score history from `surveillance_quarterly_scores.csv`; this now renders inside Company Drilldown for the selected ticker. |
| 2026-05-10 | Removed rating from scoring and UI because the available rating buckets were synthetic/inaccurate. News fallback now uses sector quality instead of rating quality; Facility, Behavioral, Governance, and Collateral no longer include rating bias. |
| 2026-05-10 | Added manual non-S&P company rows requested by user: JetBlue (`JBLU`), Lumen (`LUMN`), Braskem (`BAK`), Macy's (`M`). Paramount Skydance (`PSKY`) was already present in the active universe. |
| 2026-05-10 | Executive overview changed to remove alert distribution and focus sector heatmap on action tiers `Tier 1`, `Tier 2`, and `Tier 3`; `None` is excluded from the sector heatmap. |
| 2026-05-10 | Company Drilldown score calculation audit now includes a `Value -> sub-score` column explaining how raw values convert into sub-scores. |
| 2026-05-04 | Renamed dashboard/title language from "Public Credit Surveillance Tool" to "Surveillance Tool" and removed visible EWIF v1.0 section references from dashboard copy. |
| 2026-05-04 | Changed scoring convention to **higher score = better credit quality / lower risk** across component sub-scores, 9 indicators, composite score, alert color, and tier logic. Apple now defaults to a Green / None result in smoke testing. |
| 2026-05-04 | Replaced fixed-neutral News & Sentiment with an assumption formula when real news data is unavailable: 50% Market + 30% Financial + 20% Rating Quality. Superseded on 2026-05-10 by Sector Quality instead of Rating Quality. |
| 2026-05-04 | Replaced fixed-neutral Accounting Integrity with Beneish/Sloan scoring where available and a fallback assumption where Beneish is unavailable: 50% Financial + 30% Data Quality + 20% Sloan. |
| 2026-05-04 | Reworked Company Drilldown score calculation audit into a selectbox-driven explainer for Composite, each of the 9 indicators, and FCF Sustainability. Formula text avoids raw `clip(...)` and says scores are bounded 0-100. |
| 2026-05-04 | Expanded scenario editor: removed rating bucket selector, kept raw-input overrides, and added direct 0-100 override fields for all 9 indicator scores. Indicator overrides recalculate composite, alert, tier, and weakest drivers without changing source CSVs. |
| 2026-05-04 | Compacted `surveillance_quarterly_history.csv` from ~30 MB / 211k rows to ~5.6 MB / 131k rows by keeping only dashboard-runtime columns and metrics. Updated `build_surveillance_history.py` so future builds stay compact. |
| 2026-05-03 | Added Financial Performance enhancement: EBITDA proxy, Net Debt/EBITDA, EBITDA/Interest coverage, and FCF Sustainability sub-index from quarterly history. |
| 2026-05-03 | User requested Accounting Integrity be set to neutral 50 for all companies. This was later superseded on 2026-05-04 by directional Accounting assumptions. |
| 2026-05-03 | Added Company Drilldown score calculation audit showing exact formula, origin data, raw values, and component sub-scores for each indicator. Later updated to remove rating fields and add value-to-sub-score conversion copy. |
| 2026-05-03 | Added session-only scenario editor in Company Drilldown. Users can override selected latest-snapshot values and rerun scoring without changing source CSVs. Added reset per company and clear-all scenario controls. |
| 2026-05-03 | Removed standalone 3-year Time Series dashboard page. Added Peer and Industry Comparison inside Company Drilldown with peer average, rank, percentile, radar chart, delta chart, key ratio benchmark, and peer table. The table now shows lowest-scoring peers under the quality-up convention. |
| 2026-05-03 | Added Executive Memo generator and scenario export buttons. Memo can be downloaded as PDF; scenario/current vs base values can be downloaded as CSV. |
| 2026-05-03 | Upgraded sidebar filters: company search, sector, industry, rating, alert, tier, score range, data quality, stale facts, assumption fills, velocity trigger, SEC availability, market availability, scenario-only filter, reset filters, and current-view count. Rating filter was removed on 2026-05-10. |
| 2026-05-03 | Added `requirements.txt` with pandas, numpy, requests, yfinance, plotly, and streamlit for reproducible setup. |
| 2026-05-02 | EWIF v1.0 alignment pass: weights bug (FP 17→15%), added cogs/ppe/depreciation/sga to TAG_MAP + history extractor, implemented Beneish M-Score, blended into Accounting Integrity 50/50 with Sloan, added velocity-signal computation, replaced 4-color alerts with Tier 1/2/3 scheme + velocity trigger, dashboard now shows tier chip + Beneish panel + tier filter + tier KPI cards. |
| 2026-05-02 | Switched universe from 100-company demo to S&P 500 (503 tickers via Wikipedia). Added synthetic rating derivation for NR tickers. Rating use was later removed on 2026-05-10. |
| 2026-05-02 | Built `build_surveillance_history.py` for 3-year quarterly time series (single-tag-per-metric selection, calendar-`period_end` indexing, derived Q4 from FY−Q1−Q2−Q3, source_tag transparency). |
| 2026-05-02 | Built original Streamlit dashboard; current management-facing dashboard has 5 top-level tabs after re-adding quarterly scores on 2026-05-10. |
| 2026-05-02 | Built `build_surveillance_data.py` (data prep only — no scoring) and original dashboard. Disabled GDELT after timeouts. |
