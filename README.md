# Breakout Intelligence Lab

A Streamlit app focused on 3 steps:

1. Ingest market articles from across the web.
2. Ingest institutional trade news from large hedge funds, banks, and major trading desks.
3. Run an in-house ML breakout engine and output market outlook + top 5 breakout watchlist stocks.
4. Run a whole-market weekly tactical module that recommends 5 stocks for the current week using geopolitical + macro factors.

## What It Does

- Pulls global market articles via Google News RSS queries.
- Pulls institutional trade-flow headlines (Goldman Sachs, JPMorgan, Morgan Stanley, BofA, etc.).
- Pulls symbol-specific stock news for your watchlist.
- Pulls live quote snapshots and historical OHLCV data from Yahoo Finance web APIs.
- Trains an ensemble breakout model (Logistic + RandomForest + GradientBoosting) when `scikit-learn` is available.
- Falls back to deterministic heuristic scoring if ML dependencies are unavailable.
- Scans a broader market universe for weekly setups and adjusts ranking with a geopolitical risk regime.
- Outputs:
  - Market outlook probability
  - Ranked full watchlist
  - Top 5 breakout watchout stocks
  - Top 5 weekly tactical entries (whole-market scan)
  - Model diagnostics
  - Optional GPT in-house desk note

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## NJ Real Estate Weekly Monitor

This repo now also includes a dedicated NJ property monitor that tracks:

- New and upcoming communities/listings from NewHomeSource.
- Optional Zillow API listings (when API credentials are configured).
- Hot deals based on area-level price deviations.
- Weekly change tracking using local snapshots.

### Run Interactive Monitor

```bash
streamlit run nj_real_estate_monitor.py
```

### Deploy To Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. In Streamlit Community Cloud, click **New app**.
3. Pick your GitHub repo and branch.
4. Set **Main file path** to `nj_real_estate_monitor.py`.
5. In app settings, add your secrets from `.streamlit/secrets.toml.example` into the **Secrets** panel.
6. Click **Deploy**.

### Run Weekly Digest (CLI)

```bash
python weekly_nj_digest.py \
  --areas "Monmouth County,Middlesex County,Bergen County" \
  --sources "newhomesource,zillow" \
  --max-results 40 \
  --hot-deal-threshold 15
```

This writes:

- Snapshots to `data/real_estate_monitor/snapshots/`
- Markdown reports to `data/real_estate_monitor/reports/`

### Property Detail Page

- In **Top Hot Deals** and **Listings**, click a property name to open its detail page.
- The detail page includes:
  - Source listing details + description
  - Builder offers/incentives extraction (when available)
  - HOA, estimated tax, and insurance estimates
  - Mortgage calculator based on your down payment + rate inputs
  - GenAI + ML deal rating: **Stealer / Good Deal / Bad Deal** (with confidence, strengths, and risks)

### Zillow Configuration

Zillow blocks direct scraping requests; use API access via environment variables:

- `ZILLOW_API_KEY`
- `ZILLOW_API_URL` (optional override, default uses a RapidAPI Zillow-compatible endpoint)
- `ZILLOW_API_HOST` (optional host header for RapidAPI-style routing)
- `ZILLOW_EXTRA_QUERY_JSON` (optional JSON object for extra query params)

Example:

```bash
export ZILLOW_API_KEY="YOUR_KEY"
export ZILLOW_API_URL="https://zillow-com1.p.rapidapi.com/propertyExtendedSearch"
export ZILLOW_API_HOST="zillow-com1.p.rapidapi.com"
```

HasData Zillow example:

```bash
export ZILLOW_PROVIDER="hasdata"
export ZILLOW_API_KEY="YOUR_KEY"
export ZILLOW_API_URL="https://api.hasdata.com/scrape/zillow/listing"
export ZILLOW_API_KEY_HEADER="x-api-key"
export ZILLOW_AREA_PARAM="keyword"
export ZILLOW_AREA_FORMAT="{area}, NJ"
export ZILLOW_HASDATA_TYPE="forSale"
```

Validate your Zillow setup:

```bash
python test_zillow_api.py --area "Monmouth County" --max-results 5
```

## In App Workflow

1. Enter your symbol universe (at least 5 symbols).
2. Configure article/news ingestion settings.
3. Configure breakout model settings.
4. Configure capital objective (`$25k` to `$50k`, horizon).
5. Click **Run 3-Step Intelligence**.

## Optional Environment Variables

- `OPENAI_API_KEY`
- `OPENAI_MODEL` (default: `gpt-4.1-mini`)
- `OPENAI_WEB_MODEL` (default: `gpt-4o-search-preview`)

## Notes

- Output is research tooling, not guaranteed financial outcomes.
- Performance depends on live web/news availability at run time.
