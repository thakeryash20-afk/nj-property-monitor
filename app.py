import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import quote, quote_plus
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

try:
    from openai import OpenAI

    HAS_OPENAI = True
except Exception:
    OpenAI = None
    HAS_OPENAI = False

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except Exception:
    GradientBoostingClassifier = None
    RandomForestClassifier = None
    LogisticRegression = None
    accuracy_score = None
    roc_auc_score = None
    Pipeline = None
    StandardScaler = None
    HAS_SKLEARN = False

try:
    import yfinance as yf

    HAS_YFINANCE = True
except Exception:
    yf = None
    HAS_YFINANCE = False

APP_TITLE = "Breakout Intelligence Lab"
APP_SUBTITLE = "Web articles + institutional trade news + ML breakout watchlist"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_WEB_MODEL = os.getenv("OPENAI_WEB_MODEL", "gpt-4o-search-preview")

GLOBAL_WEB_QUERIES = [
    "stock market outlook",
    "federal reserve policy rates inflation",
    "us jobs report market impact",
    "global recession probability",
    "oil prices stock market",
    "AI semiconductor demand",
    "consumer spending trend stocks",
    "earnings season expectations",
    "geopolitical risk market",
    "china economy market impact",
]

INSTITUTIONAL_TRADE_QUERIES = [
    "Goldman Sachs stock call buy sell",
    "JPMorgan stock recommendation",
    "Morgan Stanley equity strategy",
    "Bank of America stock upgrade downgrade",
    "Citi equity research stock",
    "UBS stock outlook",
    "Bridgewater Associates market view",
    "Citadel portfolio positioning",
    "Point72 trading idea",
    "Pershing Square investment thesis",
    "Renaissance Technologies position",
    "ARK Invest buy sell stocks",
]

DEFAULT_WATCHLIST = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AMD",
    "AVGO",
    "JPM",
    "BAC",
    "XOM",
    "LLY",
    "NFLX",
    "PLTR",
]

BENCHMARK_SYMBOL = "SPY"
WEEKLY_SCAN_SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AMD",
    "AVGO",
    "NFLX",
    "PLTR",
    "JPM",
    "BAC",
    "GS",
    "MS",
    "XOM",
    "CVX",
    "LLY",
    "UNH",
    "COST",
    "WMT",
    "CAT",
    "GE",
    "LMT",
    "ORCL",
    "QCOM",
    "CRM",
    "ADBE",
    "MU",
    "PANW",
]
SYMBOL_SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "AMZN": "Consumer Cyclical",
    "GOOGL": "Communication",
    "META": "Communication",
    "TSLA": "Consumer Cyclical",
    "AMD": "Technology",
    "AVGO": "Technology",
    "NFLX": "Communication",
    "PLTR": "Technology",
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "XOM": "Energy",
    "CVX": "Energy",
    "LLY": "Healthcare",
    "UNH": "Healthcare",
    "COST": "Consumer Defensive",
    "WMT": "Consumer Defensive",
    "CAT": "Industrials",
    "GE": "Industrials",
    "LMT": "Defense",
    "ORCL": "Technology",
    "QCOM": "Technology",
    "CRM": "Technology",
    "ADBE": "Technology",
    "MU": "Technology",
    "PANW": "Technology",
}
DEFENSIVE_SECTORS = {"Healthcare", "Consumer Defensive", "Energy", "Defense"}
CYCLICAL_SECTORS = {"Technology", "Consumer Cyclical", "Financials", "Industrials", "Communication"}
GEO_KEYWORDS = [
    "war",
    "ceasefire",
    "attack",
    "missile",
    "sanction",
    "tariff",
    "trade war",
    "nato",
    "china",
    "taiwan",
    "middle east",
    "russia",
    "ukraine",
    "iran",
    "israel",
    "shipping route",
    "red sea",
    "oil shock",
]

POSITIVE_WORDS = {
    "beat",
    "beats",
    "bullish",
    "breakout",
    "upside",
    "upgrade",
    "growth",
    "strong",
    "surge",
    "rally",
    "record",
    "outperform",
    "expansion",
    "acceleration",
    "improving",
    "momentum",
}

NEGATIVE_WORDS = {
    "miss",
    "misses",
    "bearish",
    "downgrade",
    "downside",
    "weak",
    "drop",
    "selloff",
    "slowdown",
    "contraction",
    "recession",
    "warning",
    "risk",
    "volatility",
    "inflation",
    "geopolitical",
}


def normalize_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_text(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text or "")
    cleaned = unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_symbols(raw_value: str) -> list[str]:
    if not raw_value.strip():
        return []
    parts = re.split(r"[,\s]+", raw_value.strip().upper())
    symbols: list[str] = []
    for part in parts:
        sym = part.replace("$", "").strip()
        if not sym:
            continue
        if not re.fullmatch(r"[A-Z.\-]{1,10}", sym):
            continue
        if sym not in symbols:
            symbols.append(sym)
    return symbols


def safe_json_object(text: str) -> dict[str, Any]:
    content = (text or "").strip()
    if not content:
        return {}
    try:
        payload = json.loads(content)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    snippet = content[start : end + 1]
    try:
        payload = json.loads(snippet)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def sentiment_score(text: str) -> float:
    tokens = re.findall(r"[a-zA-Z']+", (text or "").lower())
    if not tokens:
        return 0.0
    pos = sum(1 for token in tokens if token in POSITIVE_WORDS)
    neg = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    raw = (pos - neg) / max(len(tokens), 1)
    return float(np.clip(raw * 8.0, -1.0, 1.0))


def weighted_sentiment(df: pd.DataFrame, text_col: str = "Text") -> float:
    if df.empty or text_col not in df.columns:
        return 0.0
    scores = df[text_col].fillna("").map(sentiment_score).astype(float)
    if "Published" not in df.columns:
        return float(scores.mean()) if len(scores) else 0.0

    published = pd.to_datetime(df["Published"], errors="coerce")
    now = pd.Timestamp.utcnow().tz_localize(None)
    if hasattr(published.dt, "tz") and published.dt.tz is not None:
        published = published.dt.tz_localize(None)
    age_days = (now - published).dt.total_seconds() / (24 * 3600)
    age_days = age_days.fillna(7.0).clip(lower=0.0, upper=30.0)
    weights = np.exp(-age_days / 6.0)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return float(scores.mean()) if len(scores) else 0.0
    return float((scores * weights).sum() / weight_sum)


def get_openai_client(runtime_key: str | None = None) -> Any | None:
    if not HAS_OPENAI:
        return None

    api_key = (runtime_key or "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        try:
            api_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
        except Exception:
            api_key = ""
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def safe_json_url(url: str, timeout: int = 12) -> dict[str, Any] | None:
    candidates = [url]
    if "query1.finance.yahoo.com" in url:
        candidates.append(url.replace("query1.finance.yahoo.com", "query2.finance.yahoo.com"))

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
    }
    for candidate in candidates:
        for _ in range(2):
            try:
                response = requests.get(candidate, headers=headers, timeout=timeout)
                if response.status_code == 429:
                    continue
                if response.status_code >= 400:
                    break
                data = response.json()
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
    return None


def fetch_yfinance_quote_row(symbol: str) -> dict[str, Any] | None:
    if not HAS_YFINANCE:
        return None

    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="5d", interval="1d", auto_adjust=False)
        if history.empty or "Close" not in history.columns:
            return None

        closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
        if closes.empty:
            return None

        price = float(closes.iloc[-1])
        prev = float(closes.iloc[-2]) if len(closes) > 1 else np.nan
        change = price - prev if not np.isnan(prev) else np.nan
        change_pct = (change / prev * 100.0) if not np.isnan(prev) and prev != 0 else np.nan

        volume = np.nan
        if "Volume" in history.columns:
            volumes = pd.to_numeric(history["Volume"], errors="coerce").dropna()
            if not volumes.empty:
                volume = float(volumes.iloc[-1])

        exchange = ""
        currency = ""
        market_cap = np.nan
        try:
            fast = dict(getattr(ticker, "fast_info", {}) or {})
            exchange = str(fast.get("exchange", ""))
            currency = str(fast.get("currency", ""))
            market_cap = normalize_float(fast.get("marketCap"), default=np.nan)
        except Exception:
            pass

        return {
            "Symbol": symbol,
            "Name": symbol,
            "Price": price,
            "Change": change,
            "Change %": change_pct,
            "Volume": volume,
            "Market Cap": market_cap,
            "Exchange": exchange,
            "Currency": currency,
        }
    except Exception:
        return None


def fetch_yfinance_chart_history(symbol: str, years: int = 5) -> pd.DataFrame:
    if not HAS_YFINANCE:
        return pd.DataFrame()

    years = int(max(1, years))
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period=f"{years}y", interval="1d", auto_adjust=False)
        if history.empty:
            return pd.DataFrame()

        frame = history.reset_index().copy()
        date_col = "Date" if "Date" in frame.columns else frame.columns[0]
        frame["Date"] = pd.to_datetime(frame[date_col], errors="coerce", utc=True).dt.tz_convert(None)
        out = pd.DataFrame(
            {
                "Date": frame["Date"],
                "Open": pd.to_numeric(frame.get("Open"), errors="coerce"),
                "High": pd.to_numeric(frame.get("High"), errors="coerce"),
                "Low": pd.to_numeric(frame.get("Low"), errors="coerce"),
                "Close": pd.to_numeric(frame.get("Close"), errors="coerce"),
                "Volume": pd.to_numeric(frame.get("Volume"), errors="coerce"),
            }
        )
        out = out.dropna(subset=["Date", "Close"]).copy()
        if out.empty:
            return pd.DataFrame()
        return out.set_index("Date").sort_index()
    except Exception:
        return pd.DataFrame()


def parse_rss(url: str, limit: int = 40) -> list[dict[str, Any]]:
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        if response.status_code >= 400:
            return []
        payload = response.content
        root = ET.fromstring(payload)
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for item in root.findall(".//item")[:limit]:
        title = clean_text(item.findtext("title", default=""))
        summary = clean_text(item.findtext("description", default=""))
        link = str(item.findtext("link", default="")).strip()
        source = clean_text(item.findtext("source", default="Google News"))

        published = None
        raw_date = str(item.findtext("pubDate", default="")).strip()
        if raw_date:
            try:
                published = pd.to_datetime(raw_date, utc=True).tz_convert(None).to_pydatetime()
            except Exception:
                published = None

        text = f"{title}. {summary}".strip()
        rows.append(
            {
                "Title": title,
                "Summary": summary,
                "Link": link,
                "Source": source or "Google News",
                "Published": published,
                "Text": text,
            }
        )
    return rows


def google_news_rss(query: str, days: int = 7) -> str:
    encoded = quote_plus(f"{query} when:{days}d")
    return f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"


def fetch_articles(queries: list[str], category: str, max_per_query: int, days: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query in queries:
        url = google_news_rss(query=query, days=days)
        for row in parse_rss(url, limit=max_per_query):
            row["Category"] = category
            row["Query"] = query
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df[df["Title"].astype(str).str.len() > 0].copy()
    df = df.drop_duplicates(subset=["Title", "Link"]).copy()
    df["Sentiment"] = df["Text"].map(sentiment_score)
    if "Published" in df.columns:
        df["Published"] = pd.to_datetime(df["Published"], errors="coerce")
        df = df.sort_values("Published", ascending=False, na_position="last")
    return df.reset_index(drop=True)


def fetch_symbol_articles(symbols: list[str], days: int, max_per_symbol: int) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        query = f"{symbol} stock breakout earnings guidance"
        df = fetch_articles([query], category="symbol", max_per_query=max_per_symbol, days=days)
        if not df.empty:
            df["Symbol"] = symbol
        out[symbol] = df
    return out


@st.cache_data(ttl=120, show_spinner=False)
def fetch_yahoo_quote_snapshot(symbols: tuple[str, ...]) -> tuple[pd.DataFrame, list[str]]:
    if not symbols:
        return pd.DataFrame(), []

    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for raw_symbol in symbols:
        symbol = str(raw_symbol).upper().strip()
        if not symbol:
            continue

        row: dict[str, Any] | None = None

        search_url = (
            "https://query1.finance.yahoo.com/v1/finance/search"
            f"?q={quote_plus(symbol)}&quotesCount=10&newsCount=0"
        )
        payload = safe_json_url(search_url)
        if payload:
            quote_items = payload.get("quotes", [])
            candidate = None
            for item in quote_items:
                item_symbol = str(item.get("symbol", "")).upper().strip()
                if item_symbol == symbol:
                    candidate = item
                    break
            if candidate is None and quote_items:
                candidate = quote_items[0]

            if isinstance(candidate, dict):
                price = normalize_float(candidate.get("regularMarketPrice"), default=np.nan)
                if not np.isnan(price):
                    row = {
                        "Symbol": symbol,
                        "Name": str(candidate.get("shortname", candidate.get("longname", symbol))),
                        "Price": price,
                        "Change": normalize_float(candidate.get("regularMarketChange"), default=np.nan),
                        "Change %": normalize_float(candidate.get("regularMarketChangePercent"), default=np.nan),
                        "Volume": normalize_float(candidate.get("regularMarketVolume"), default=np.nan),
                        "Market Cap": normalize_float(candidate.get("marketCap"), default=np.nan),
                        "Exchange": str(candidate.get("exchangeDisp", candidate.get("exchange", ""))),
                        "Currency": str(candidate.get("currency", "")),
                    }

        if row is None:
            chart_url = (
                f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}"
                "?range=5d&interval=1d&events=div,splits"
            )
            chart_payload = safe_json_url(chart_url)
            result_list = chart_payload.get("chart", {}).get("result", []) if chart_payload else []
            if result_list:
                result = result_list[0]
                meta = result.get("meta", {})
                quote_blocks = result.get("indicators", {}).get("quote", [])
                quote_block = quote_blocks[0] if quote_blocks else {}
                closes = quote_block.get("close", [])
                volumes = quote_block.get("volume", [])
                valid_closes = [normalize_float(v, default=np.nan) for v in closes if v is not None]
                valid_closes = [v for v in valid_closes if not np.isnan(v)]
                if valid_closes:
                    price = valid_closes[-1]
                    prev = valid_closes[-2] if len(valid_closes) > 1 else np.nan
                    change = price - prev if not np.isnan(prev) else np.nan
                    change_pct = (change / prev * 100.0) if not np.isnan(prev) and prev != 0 else np.nan
                    last_vol = np.nan
                    if isinstance(volumes, list) and volumes:
                        for v in reversed(volumes):
                            if v is not None:
                                last_vol = normalize_float(v, default=np.nan)
                                break
                    row = {
                        "Symbol": symbol,
                        "Name": str(meta.get("symbol", symbol)),
                        "Price": price,
                        "Change": change,
                        "Change %": change_pct,
                        "Volume": last_vol,
                        "Market Cap": np.nan,
                        "Exchange": str(meta.get("exchangeName", "")),
                        "Currency": str(meta.get("currency", "")),
                    }

        if row is None:
            row = fetch_yfinance_quote_row(symbol)

        if row is None:
            errors.append(f"{symbol}: quote unavailable")
        else:
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        if not errors:
            errors = ["No quote data returned."]
        return df, errors

    missing = [symbol for symbol in symbols if symbol not in set(df["Symbol"].tolist())]
    if missing:
        errors.append("Missing quotes: " + ", ".join(missing))
    return df.sort_values("Symbol").reset_index(drop=True), errors


@st.cache_data(ttl=900, show_spinner=False)
def fetch_yahoo_chart_history(symbol: str, years: int = 5) -> pd.DataFrame:
    encoded = quote(symbol, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}"
        f"?range={years}y&interval=1d&events=div,splits"
    )
    payload = safe_json_url(url, timeout=15)
    if not payload:
        return fetch_yfinance_chart_history(symbol=symbol, years=years)

    results = payload.get("chart", {}).get("result", [])
    if not results:
        return fetch_yfinance_chart_history(symbol=symbol, years=years)

    result = results[0]
    timestamps = result.get("timestamp", [])
    quote_list = result.get("indicators", {}).get("quote", [])
    quote_block = quote_list[0] if quote_list else {}
    if not timestamps or not quote_block:
        return fetch_yfinance_chart_history(symbol=symbol, years=years)

    close_values = quote_block.get("close", [])
    adjclose_list = result.get("indicators", {}).get("adjclose", [])
    if adjclose_list and isinstance(adjclose_list[0], dict):
        close_values = adjclose_list[0].get("adjclose", close_values)

    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
            "Open": quote_block.get("open", []),
            "High": quote_block.get("high", []),
            "Low": quote_block.get("low", []),
            "Close": close_values,
            "Volume": quote_block.get("volume", []),
        }
    )
    if frame.empty:
        return fetch_yfinance_chart_history(symbol=symbol, years=years)

    frame = frame.dropna(subset=["Close"]).copy()
    if frame.empty:
        return fetch_yfinance_chart_history(symbol=symbol, years=years)

    return frame.set_index("Date").sort_index()


def aggregate_daily_news(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Date", f"{prefix}_sentiment", f"{prefix}_count"])

    work = df.copy()
    work["Published"] = pd.to_datetime(work["Published"], errors="coerce")
    if hasattr(work["Published"].dt, "tz") and work["Published"].dt.tz is not None:
        work["Published"] = work["Published"].dt.tz_localize(None)
    work["Date"] = work["Published"].dt.normalize()
    work = work.dropna(subset=["Date"])
    if work.empty:
        return pd.DataFrame(columns=["Date", f"{prefix}_sentiment", f"{prefix}_count"])

    daily = (
        work.groupby("Date", as_index=False)
        .agg(
            sentiment=("Sentiment", "mean"),
            count=("Title", "count"),
        )
        .rename(
            columns={
                "sentiment": f"{prefix}_sentiment",
                "count": f"{prefix}_count",
            }
        )
    )
    return daily


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = down.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def build_feature_dataset(
    history_df: pd.DataFrame,
    global_daily: pd.DataFrame,
    inst_daily: pd.DataFrame,
    symbol_daily: pd.DataFrame,
    horizon_days: int,
    breakout_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.Series | None]:
    if history_df.empty or "Close" not in history_df.columns:
        return pd.DataFrame(), pd.DataFrame(), [], None

    df = history_df.copy()
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df["ret_1d"] = df["Close"].pct_change(1)
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_20d"] = df["Close"].pct_change(20)
    df["vol_10d"] = df["ret_1d"].rolling(10).std()
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["sma20_ratio"] = df["Close"] / df["sma_20"]
    df["sma50_ratio"] = df["Close"] / df["sma_50"]
    df["rsi_14"] = compute_rsi(df["Close"], 14)
    df["breakout_dist"] = df["Close"] / df["Close"].rolling(20).max() - 1.0
    volume_mean = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / volume_mean.replace(0, np.nan)
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)

    df["future_return"] = df["Close"].shift(-horizon_days) / df["Close"] - 1.0
    df["target_breakout"] = (df["future_return"] >= breakout_threshold).astype(int)

    df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()

    for daily in [global_daily, inst_daily, symbol_daily]:
        if daily.empty:
            continue
        merge_cols = [c for c in daily.columns if c != "Date"]
        df = df.merge(daily[["Date"] + merge_cols], on="Date", how="left")

    for col in [
        "global_sentiment",
        "global_count",
        "inst_sentiment",
        "inst_count",
        "symbol_sentiment",
        "symbol_count",
    ]:
        if col in df.columns:
            fill = 0.0 if col.endswith("sentiment") else 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill)

    feature_cols = [
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "vol_10d",
        "vol_20d",
        "sma20_ratio",
        "sma50_ratio",
        "rsi_14",
        "breakout_dist",
        "volume_ratio",
        "intraday_range",
    ]
    for optional_col in [
        "global_sentiment",
        "global_count",
        "inst_sentiment",
        "inst_count",
        "symbol_sentiment",
        "symbol_count",
    ]:
        if optional_col in df.columns:
            feature_cols.append(optional_col)

    dataset = (
        df[feature_cols + ["target_breakout", "future_return", "Date", "Close"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    latest_features = (
        df[feature_cols + ["Date", "Close"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .tail(1)
        .copy()
    )
    latest_row = latest_features.iloc[0] if not latest_features.empty else None
    return dataset, latest_features, feature_cols, latest_row


def train_breakout_ensemble(
    dataset: pd.DataFrame,
    latest_features: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    if dataset.empty or latest_features.empty or not feature_cols:
        return {"error": "insufficient feature dataset"}

    if dataset["target_breakout"].nunique() < 2 or len(dataset) < 180:
        return {"error": "insufficient class diversity"}

    if not HAS_SKLEARN:
        return {"error": "sklearn unavailable"}

    split_idx = int(len(dataset) * 0.8)
    split_idx = max(split_idx, 120)
    if split_idx >= len(dataset) - 20:
        split_idx = len(dataset) - 20
    if split_idx < 80:
        return {"error": "insufficient train-test split"}

    X = dataset[feature_cols]
    y = dataset["target_breakout"].astype(int)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    models: list[tuple[str, Any]] = [
        (
            "Logistic",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(max_iter=4000)),
                ]
            ),
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=4,
                random_state=42,
            ),
        ),
        (
            "GradientBoost",
            GradientBoostingClassifier(random_state=42),
        ),
    ]

    test_probs: list[np.ndarray] = []
    latest_probs: list[float] = []
    diagnostics: list[dict[str, Any]] = []

    for model_name, model in models:
        try:
            model.fit(X_train, y_train)
            prob_test = model.predict_proba(X_test)[:, 1]
            pred_test = (prob_test >= 0.5).astype(int)
            prob_latest = float(model.predict_proba(latest_features[feature_cols])[:, 1][0])

            acc = float(accuracy_score(y_test, pred_test))
            auc = None
            if y_test.nunique() >= 2:
                auc = float(roc_auc_score(y_test, prob_test))

            diagnostics.append(
                {
                    "model": model_name,
                    "test_accuracy": round(acc, 4),
                    "test_roc_auc": round(auc, 4) if auc is not None else None,
                    "latest_breakout_prob": round(prob_latest, 4),
                }
            )
            latest_probs.append(prob_latest)
            test_probs.append(prob_test)
        except Exception:
            continue

    if not latest_probs:
        return {"error": "all models failed"}

    ensemble_prob = float(np.mean(latest_probs))
    ensemble_test = np.mean(np.column_stack(test_probs), axis=1)
    test_acc = float(accuracy_score(y_test, (ensemble_test >= 0.5).astype(int)))
    test_auc = None
    if y_test.nunique() >= 2:
        test_auc = float(roc_auc_score(y_test, ensemble_test))

    return {
        "error": None,
        "ensemble_prob": ensemble_prob,
        "ensemble_test_accuracy": test_acc,
        "ensemble_test_auc": test_auc,
        "model_diagnostics": diagnostics,
        "rows_used": int(len(dataset)),
    }


def fallback_breakout_probability(latest_row: pd.Series | None) -> float:
    if latest_row is None:
        return 0.5

    ret_20d = normalize_float(latest_row.get("ret_20d"), default=0.0)
    breakout_dist = normalize_float(latest_row.get("breakout_dist"), default=0.0)
    volume_ratio = normalize_float(latest_row.get("volume_ratio"), default=1.0)
    rsi = normalize_float(latest_row.get("rsi_14"), default=50.0)
    global_sent = normalize_float(latest_row.get("global_sentiment"), default=0.0)
    inst_sent = normalize_float(latest_row.get("inst_sentiment"), default=0.0)
    symbol_sent = normalize_float(latest_row.get("symbol_sentiment"), default=0.0)

    score = 0.5
    score += 0.18 * np.tanh(ret_20d * 8.0)
    score += 0.14 * np.tanh(breakout_dist * 18.0)
    score += 0.08 * np.tanh((volume_ratio - 1.0) * 2.0)
    score += 0.08 * np.tanh((rsi - 50.0) / 12.0)
    score += 0.22 * symbol_sent
    score += 0.18 * global_sent
    score += 0.12 * inst_sent
    return float(np.clip(score, 0.05, 0.95))


def market_outlook_from_probability(prob_up: float) -> str:
    if prob_up >= 0.6:
        return "Bullish"
    if prob_up <= 0.4:
        return "Bearish"
    return "Neutral"


def required_annualized_return(budget: float, target_amount: float, horizon_months: int) -> float:
    if budget <= 0 or target_amount <= budget or horizon_months <= 0:
        return 0.0
    years = horizon_months / 12.0
    return float((target_amount / budget) ** (1.0 / years) - 1.0)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_yahoo_symbol_news(symbol: str, max_items: int = 10) -> pd.DataFrame:
    url = (
        "https://query1.finance.yahoo.com/v1/finance/search"
        f"?q={quote_plus(symbol + ' stock')}&quotesCount=0&newsCount={max_items}"
    )
    payload = safe_json_url(url, timeout=12)
    news_items = payload.get("news", []) if isinstance(payload, dict) else []

    rows: list[dict[str, Any]] = []
    for item in news_items[:max_items]:
        if not isinstance(item, dict):
            continue
        title = clean_text(str(item.get("title", "")))
        if not title:
            continue
        publisher = clean_text(str(item.get("publisher", "Yahoo Finance")))
        link = str(item.get("link", item.get("url", ""))).strip()
        summary = clean_text(str(item.get("summary", "")))

        published = None
        ts = item.get("providerPublishTime")
        if isinstance(ts, (int, float)):
            try:
                published = pd.to_datetime(int(ts), unit="s", utc=True).tz_convert(None).to_pydatetime()
            except Exception:
                published = None
        if published is None:
            raw_pub = item.get("pubDate")
            if raw_pub:
                try:
                    published = pd.to_datetime(raw_pub, utc=True).tz_convert(None).to_pydatetime()
                except Exception:
                    published = None

        rows.append(
            {
                "Title": title,
                "Summary": summary,
                "Link": link,
                "Source": publisher or "Yahoo Finance",
                "Published": published,
                "Text": f"{title}. {summary}".strip(),
                "Symbol": symbol,
            }
        )

    if not rows:
        # RSS fallback keeps module alive if Yahoo news endpoint is sparse.
        rss_query = f"{symbol} stock news trading setup"
        rss_rows = parse_rss(google_news_rss(rss_query, days=7), limit=max_items)
        for row in rss_rows:
            row["Symbol"] = symbol
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Sentiment"] = df["Text"].map(sentiment_score)
    if "Published" in df.columns:
        df["Published"] = pd.to_datetime(df["Published"], errors="coerce")
        df = df.sort_values("Published", ascending=False, na_position="last")
    return df.reset_index(drop=True)


def compute_geopolitical_regime(global_df: pd.DataFrame, institutional_df: pd.DataFrame) -> dict[str, Any]:
    frames = [df for df in [global_df, institutional_df] if isinstance(df, pd.DataFrame) and not df.empty]
    if not frames:
        return {
            "regime": "Normal",
            "geo_sentiment": 0.0,
            "geo_pressure": 0.0,
            "mention_ratio": 0.0,
            "headline_count": 0,
            "geo_headlines": pd.DataFrame(),
        }

    merged = pd.concat(frames, ignore_index=True)
    if "Text" not in merged.columns:
        merged["Text"] = merged.get("Title", pd.Series([""] * len(merged))).astype(str)
    text_lower = merged["Text"].fillna("").astype(str).str.lower()
    geo_regex = "|".join(re.escape(k) for k in GEO_KEYWORDS)
    geo_mask = text_lower.str.contains(geo_regex, regex=True, na=False)
    geo_df = merged[geo_mask].copy()

    base_sentiment = weighted_sentiment(merged, text_col="Text")
    geo_sentiment = weighted_sentiment(geo_df, text_col="Text") if not geo_df.empty else base_sentiment * 0.5
    mention_ratio = float(len(geo_df) / max(len(merged), 1))
    geo_pressure = float(max(0.0, -geo_sentiment) * (1.0 + mention_ratio * 1.8))

    if geo_pressure >= 0.25:
        regime = "High Risk"
    elif geo_pressure >= 0.12:
        regime = "Elevated"
    else:
        regime = "Normal"

    geo_headlines = pd.DataFrame()
    if not geo_df.empty:
        view_cols = [c for c in ["Published", "Source", "Title", "Link", "Sentiment"] if c in geo_df.columns]
        geo_headlines = geo_df[view_cols].sort_values("Published", ascending=False, na_position="last").head(30)

    return {
        "regime": regime,
        "geo_sentiment": geo_sentiment,
        "geo_pressure": geo_pressure,
        "mention_ratio": mention_ratio,
        "headline_count": int(len(geo_df)),
        "geo_headlines": geo_headlines,
    }


def sector_geopolitical_adjustment(
    sector: str,
    geo_regime: str,
    global_sentiment: float,
    institutional_sentiment: float,
) -> float:
    sector_name = (sector or "Other").strip()
    adjustment = 0.0

    if geo_regime == "High Risk":
        if sector_name in DEFENSIVE_SECTORS:
            adjustment += 0.09
        if sector_name in CYCLICAL_SECTORS:
            adjustment -= 0.09
    elif geo_regime == "Elevated":
        if sector_name in DEFENSIVE_SECTORS:
            adjustment += 0.05
        if sector_name in CYCLICAL_SECTORS:
            adjustment -= 0.05
    else:
        risk_on = global_sentiment + institutional_sentiment
        if risk_on > 0.08 and sector_name in CYCLICAL_SECTORS:
            adjustment += 0.03
        if risk_on < -0.08 and sector_name in DEFENSIVE_SECTORS:
            adjustment += 0.02

    return float(adjustment)


def weekly_probability_from_score(score: float) -> float:
    return float(np.clip(1.0 / (1.0 + np.exp(-3.6 * score)), 0.03, 0.97))


def build_gpt_weekly_note(
    client: Any,
    geo_summary: dict[str, Any],
    top5_df: pd.DataFrame,
    scan_df: pd.DataFrame,
    weekly_horizon_days: int,
) -> tuple[str, str | None]:
    if client is None:
        return "", "OpenAI client unavailable"

    top_payload = top5_df.head(5).to_dict(orient="records") if not top5_df.empty else []
    scan_payload = scan_df.head(12).to_dict(orient="records") if not scan_df.empty else []
    prompt = (
        "You are a tactical equity strategist.\n"
        "Write a concise weekly trading note from this quant + geopolitics dataset.\n"
        f"Weekly horizon days: {weekly_horizon_days}\n"
        f"Geopolitical regime: {geo_summary.get('regime')} | "
        f"Geo pressure: {normalize_float(geo_summary.get('geo_pressure')):.3f}\n"
        f"Top 5 weekly picks: {json.dumps(top_payload, default=str)}\n"
        f"Top scan rows: {json.dumps(scan_payload, default=str)}\n"
        "Output markdown with sections: Weekly Market View, 5 Weekly Entries, Risk Events To Watch."
    )
    try:
        response = client.responses.create(
            model=os.getenv("OPENAI_WEB_MODEL", DEFAULT_WEB_MODEL),
            input=prompt,
            tools=[{"type": "web_search_preview"}],
        )
        return response.output_text or "", None
    except Exception as exc:
        try:
            fallback = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You are a tactical equity strategist. Respond in markdown."},
                    {"role": "user", "content": prompt},
                ],
            )
            return fallback.choices[0].message.content or "", None
        except Exception as second_exc:
            return "", f"Weekly GPT note failed: {exc}; fallback failed: {second_exc}"


def build_weekly_market_module(
    global_df: pd.DataFrame,
    institutional_df: pd.DataFrame,
    weekly_horizon_days: int,
    max_symbol_news: int,
    investment_budget: float,
    runtime_openai_key: str | None,
    use_gpt_overlay: bool,
) -> dict[str, Any]:
    global_sentiment = weighted_sentiment(global_df, text_col="Text") if not global_df.empty else 0.0
    institutional_sentiment = weighted_sentiment(institutional_df, text_col="Text") if not institutional_df.empty else 0.0
    geo_summary = compute_geopolitical_regime(global_df, institutional_df)

    technical_rows: list[dict[str, Any]] = []
    for symbol in WEEKLY_SCAN_SYMBOLS:
        history_df = fetch_yahoo_chart_history(symbol=symbol, years=2)
        if history_df.empty or len(history_df) < 80:
            continue

        close = history_df["Close"].dropna()
        if close.empty:
            continue
        volume = history_df["Volume"].fillna(0.0)
        latest_price = float(close.iloc[-1])
        ret_5d = float(close.pct_change(5).dropna().iloc[-1]) if len(close) > 8 else 0.0
        ret_20d = float(close.pct_change(20).dropna().iloc[-1]) if len(close) > 25 else 0.0
        high_30d = float(close.rolling(30).max().dropna().iloc[-1]) if len(close) > 35 else latest_price
        breakout_dist = latest_price / high_30d - 1.0 if high_30d > 0 else 0.0
        volume_ma20 = float(volume.rolling(20).mean().dropna().iloc[-1]) if len(volume) > 25 else 0.0
        volume_ratio = float(volume.iloc[-1] / volume_ma20) if volume_ma20 > 0 else 1.0
        rsi = float(compute_rsi(close, 14).iloc[-1]) if len(close) > 20 else 50.0
        vol_20d = float(close.pct_change().rolling(20).std().dropna().iloc[-1]) if len(close) > 25 else 0.02

        rsi_center_score = 1.0 - min(1.0, abs(rsi - 58.0) / 42.0)
        technical_score = (
            0.34 * np.tanh(ret_5d * 16.0)
            + 0.26 * np.tanh(ret_20d * 8.0)
            + 0.18 * np.tanh((volume_ratio - 1.0) * 2.0)
            + 0.14 * np.tanh((breakout_dist + 0.03) * 20.0)
            + 0.08 * (rsi_center_score * 2.0 - 1.0)
        )

        technical_rows.append(
            {
                "Symbol": symbol,
                "Sector": SYMBOL_SECTOR_MAP.get(symbol, "Other"),
                "Last Price": latest_price,
                "Ret 5D %": ret_5d * 100.0,
                "Ret 20D %": ret_20d * 100.0,
                "Breakout Dist %": breakout_dist * 100.0,
                "Volume Ratio": volume_ratio,
                "RSI(14)": rsi,
                "Vol 20D %": vol_20d * 100.0,
                "Technical Score": float(technical_score),
            }
        )

    scan_df = pd.DataFrame(technical_rows)
    if scan_df.empty:
        return {
            "weekly_scan": pd.DataFrame(),
            "weekly_top5": pd.DataFrame(),
            "geo_summary": geo_summary,
            "weekly_note": "",
            "weekly_note_error": "Weekly scan could not fetch enough market history.",
            "weekly_engine": "unavailable",
        }

    scan_df = scan_df.sort_values("Technical Score", ascending=False).reset_index(drop=True)
    shortlist = scan_df["Symbol"].head(15).tolist()

    symbol_sentiment_map: dict[str, float] = {symbol: 0.0 for symbol in scan_df["Symbol"].tolist()}
    symbol_news_count_map: dict[str, int] = {symbol: 0 for symbol in scan_df["Symbol"].tolist()}
    for symbol in shortlist:
        symbol_news = fetch_yahoo_symbol_news(symbol, max_items=max_symbol_news)
        if symbol_news.empty:
            continue
        symbol_sentiment_map[symbol] = weighted_sentiment(symbol_news, text_col="Text")
        symbol_news_count_map[symbol] = int(len(symbol_news))

    final_rows: list[dict[str, Any]] = []
    for _, row in scan_df.iterrows():
        symbol = str(row["Symbol"])
        sector = str(row.get("Sector", "Other"))
        symbol_sentiment = float(symbol_sentiment_map.get(symbol, 0.0))
        sector_adj = sector_geopolitical_adjustment(
            sector=sector,
            geo_regime=str(geo_summary.get("regime", "Normal")),
            global_sentiment=global_sentiment,
            institutional_sentiment=institutional_sentiment,
        )

        final_score = (
            normalize_float(row.get("Technical Score"), default=0.0)
            + 0.22 * global_sentiment
            + 0.20 * institutional_sentiment
            + 0.16 * symbol_sentiment
            + sector_adj
        )
        breakout_prob = weekly_probability_from_score(final_score)
        confidence = float(np.clip(42.0 + abs(breakout_prob - 0.5) * 140.0, 35.0, 97.0))
        vol_20d_pct = normalize_float(row.get("Vol 20D %"), default=2.0)
        stop_loss_pct = float(np.clip(3.5 + vol_20d_pct * 1.6, 4.0, 12.0))
        target_pct = float(np.clip(stop_loss_pct * 1.9, 7.0, 24.0))
        price = normalize_float(row.get("Last Price"), default=0.0)
        entry_low = price * 0.985 if price > 0 else np.nan
        entry_high = price * 1.005 if price > 0 else np.nan

        rationale = []
        if normalize_float(row.get("Ret 5D %")) > 2.5:
            rationale.append("strong weekly momentum")
        if normalize_float(row.get("Breakout Dist %")) > -1.5:
            rationale.append("pressing against local highs")
        if symbol_sentiment > 0.08:
            rationale.append("positive stock news flow")
        if sector_adj > 0:
            rationale.append("sector favored under current geopolitical regime")
        if not rationale:
            rationale.append("balanced technical setup")

        recommendation = "WATCH"
        if breakout_prob >= 0.72:
            recommendation = "ENTER THIS WEEK"
        elif breakout_prob >= 0.62:
            recommendation = "ACCUMULATE"
        elif breakout_prob < 0.45:
            recommendation = "AVOID THIS WEEK"

        final_rows.append(
            {
                "Symbol": symbol,
                "Sector": sector,
                "Weekly Recommendation": recommendation,
                "Weekly Breakout Prob %": round(breakout_prob * 100.0, 2),
                "Confidence %": round(confidence, 1),
                "Last Price": round(price, 4) if price > 0 else None,
                "Entry Zone Low": round(entry_low, 4) if not np.isnan(entry_low) else None,
                "Entry Zone High": round(entry_high, 4) if not np.isnan(entry_high) else None,
                "Stop Loss %": round(stop_loss_pct, 2),
                "Take Profit %": round(target_pct, 2),
                "Ret 5D %": round(normalize_float(row.get("Ret 5D %")), 2),
                "Ret 20D %": round(normalize_float(row.get("Ret 20D %")), 2),
                "Volume Ratio": round(normalize_float(row.get("Volume Ratio"), default=1.0), 2),
                "RSI(14)": round(normalize_float(row.get("RSI(14)"), default=50.0), 2),
                "Symbol Sentiment": round(symbol_sentiment, 3),
                "Global Sentiment": round(global_sentiment, 3),
                "Institutional Sentiment": round(institutional_sentiment, 3),
                "Geo Regime": str(geo_summary.get("regime", "Normal")),
                "Why": "; ".join(rationale),
            }
        )

    weekly_scan_df = pd.DataFrame(final_rows).sort_values(
        ["Weekly Breakout Prob %", "Confidence %"],
        ascending=[False, False],
    ).reset_index(drop=True)
    weekly_top5 = weekly_scan_df.head(5).copy()
    if not weekly_top5.empty and investment_budget > 0:
        default_alloc = [24, 22, 20, 18, 16]
        alloc = default_alloc[: len(weekly_top5)]
        alloc_sum = float(sum(alloc)) if alloc else 1.0
        alloc_pct = [round(a / alloc_sum * 100.0, 2) for a in alloc]
        weekly_top5["Suggested Allocation %"] = alloc_pct
        weekly_top5["Suggested Allocation $"] = [
            round(investment_budget * pct / 100.0, 2) for pct in alloc_pct
        ]

    weekly_note = ""
    weekly_note_error = None
    weekly_engine = "quant"
    if use_gpt_overlay:
        client = get_openai_client(runtime_key=runtime_openai_key)
        weekly_note, weekly_note_error = build_gpt_weekly_note(
            client=client,
            geo_summary=geo_summary,
            top5_df=weekly_top5,
            scan_df=weekly_scan_df,
            weekly_horizon_days=weekly_horizon_days,
        )
        if weekly_note:
            weekly_engine = "quant+gpt"
    if not weekly_note:
        weekly_note = build_quant_weekly_note(
            top5_df=weekly_top5,
            geo_summary=geo_summary,
            weekly_horizon_days=weekly_horizon_days,
        )
        weekly_note_error = None

    return {
        "weekly_scan": weekly_scan_df,
        "weekly_top5": weekly_top5,
        "geo_summary": geo_summary,
        "weekly_note": weekly_note,
        "weekly_note_error": weekly_note_error,
        "weekly_engine": weekly_engine,
    }


def build_gpt_overlay(
    client: Any,
    symbols: list[str],
    market_prob_up: float,
    top_watch_df: pd.DataFrame,
    global_df: pd.DataFrame,
    inst_df: pd.DataFrame,
    investment_budget: float,
    target_amount: float,
    horizon_months: int,
) -> tuple[str, str | None]:
    if client is None:
        return "", "OpenAI client unavailable"

    market_outlook = market_outlook_from_probability(market_prob_up)
    top_payload = top_watch_df.head(5).to_dict(orient="records") if not top_watch_df.empty else []

    global_payload = []
    if not global_df.empty:
        global_payload = global_df[["Title", "Source", "Sentiment"]].head(15).to_dict(orient="records")

    inst_payload = []
    if not inst_df.empty:
        inst_payload = inst_df[["Title", "Source", "Sentiment"]].head(15).to_dict(orient="records")

    prompt = (
        "You are a buy-side desk strategist.\n"
        "Build a concise in-house note with practical action steps.\n"
        f"Budget: {investment_budget:.2f}, Target: {target_amount:.2f}, Horizon months: {horizon_months}.\n"
        f"Current market outlook probability-up: {market_prob_up:.3f} ({market_outlook}).\n"
        f"Watchlist universe: {symbols}.\n"
        f"Top breakout candidates: {json.dumps(top_payload, default=str)}\n"
        f"Global web headlines sample: {json.dumps(global_payload, default=str)}\n"
        f"Institutional trade headlines sample: {json.dumps(inst_payload, default=str)}\n"
        "Output markdown with sections: Market Outlook, What To Do This Week, Risk Controls."
    )
    try:
        response = client.responses.create(
            model=os.getenv("OPENAI_WEB_MODEL", DEFAULT_WEB_MODEL),
            input=prompt,
            tools=[{"type": "web_search_preview"}],
        )
        return response.output_text or "", None
    except Exception as exc:
        try:
            fallback = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a buy-side strategist. Provide concise markdown.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return fallback.choices[0].message.content or "", None
        except Exception as second_exc:
            return "", f"GPT overlay failed: {exc}; fallback failed: {second_exc}"


def build_quant_weekly_note(
    top5_df: pd.DataFrame,
    geo_summary: dict[str, Any],
    weekly_horizon_days: int,
) -> str:
    regime = str(geo_summary.get("regime", "Normal"))
    geo_pressure = normalize_float(geo_summary.get("geo_pressure"), default=0.0)
    headline_count = int(normalize_float(geo_summary.get("headline_count"), default=0.0))
    lines = [
        "### Weekly Market View",
        f"- Geopolitical regime: **{regime}** (pressure score **{geo_pressure:.3f}**).",
        f"- Geopolitical headline count: **{headline_count}**.",
        f"- Trading horizon: **{int(max(1, weekly_horizon_days))} days**.",
        "",
        "### 5 Weekly Entries",
    ]

    if isinstance(top5_df, pd.DataFrame) and not top5_df.empty:
        for _, row in top5_df.head(5).iterrows():
            symbol = str(row.get("Symbol", "")).strip()
            rec = str(row.get("Weekly Recommendation", "WATCH"))
            prob = normalize_float(row.get("Weekly Breakout Prob %"), default=0.0)
            stop_loss = normalize_float(row.get("Stop Loss %"), default=np.nan)
            take_profit = normalize_float(row.get("Take Profit %"), default=np.nan)
            why = str(row.get("Why", "quant setup")).strip() or "quant setup"
            stop_txt = f"{stop_loss:.1f}%" if not np.isnan(stop_loss) else "n/a"
            target_txt = f"{take_profit:.1f}%" if not np.isnan(take_profit) else "n/a"
            lines.append(
                f"- **{symbol}**: {rec}; breakout probability {prob:.1f}%; stop {stop_txt}; target {target_txt}; reason: {why}."
            )
    else:
        lines.append("- No weekly candidates met the model quality filters this run.")

    lines.extend(
        [
            "",
            "### Risk Events To Watch",
            "- Watch macro prints (inflation, payrolls, central-bank commentary) for volatility regime shifts.",
            "- Cut exposure quickly if index breadth deteriorates and top picks break below planned stop levels.",
        ]
    )
    return "\n".join(lines)


def build_quant_desk_note(
    market_prob_up: float,
    top_watch_df: pd.DataFrame,
    investment_budget: float,
    target_amount: float,
    horizon_months: int,
) -> str:
    market_outlook = market_outlook_from_probability(market_prob_up)
    req_cagr = required_annualized_return(investment_budget, target_amount, horizon_months)
    lines = [
        "### Market Outlook",
        f"- Model market outlook: **{market_outlook}**.",
        f"- Probability of upside regime: **{market_prob_up * 100:.2f}%**.",
        f"- Required annualized return to reach target: **{req_cagr * 100:.2f}%**.",
        "",
        "### What To Do This Week",
    ]

    if isinstance(top_watch_df, pd.DataFrame) and not top_watch_df.empty:
        alloc_template = [30, 25, 20, 15, 10]
        top = top_watch_df.head(5).reset_index(drop=True)
        alloc = alloc_template[: len(top)]
        alloc_sum = float(sum(alloc)) if alloc else 1.0
        for idx, (_, row) in enumerate(top.iterrows()):
            symbol = str(row.get("Symbol", "")).strip()
            prob = normalize_float(row.get("Breakout Prob %"), default=0.0)
            rec = str(row.get("Recommendation", "WATCHOUT")).strip()
            allocation_pct = alloc[idx] / alloc_sum * 100.0
            allocation_usd = investment_budget * allocation_pct / 100.0
            lines.append(
                f"- **{symbol}**: {rec}; breakout probability {prob:.1f}%; suggested allocation {allocation_pct:.1f}% (~${allocation_usd:,.0f})."
            )
    else:
        lines.append("- No symbols qualified for high-conviction setup in this run.")

    lines.extend(
        [
            "",
            "### Risk Controls",
            "- Use position-level stops and reduce exposure if aggregate sentiment turns negative.",
            "- Avoid adding size immediately ahead of major earnings or macro-event windows.",
        ]
    )
    return "\n".join(lines)


def run_pipeline(
    symbols: list[str],
    article_days: int,
    max_articles_per_query: int,
    max_symbol_articles: int,
    horizon_days: int,
    breakout_threshold: float,
    investment_budget: float,
    target_amount: float,
    horizon_months: int,
    runtime_openai_key: str | None,
    use_gpt_overlay: bool,
    weekly_module_enabled: bool,
    weekly_horizon_days: int,
    weekly_symbol_news_items: int,
) -> dict[str, Any]:
    global_df = fetch_articles(
        queries=GLOBAL_WEB_QUERIES,
        category="global",
        max_per_query=max_articles_per_query,
        days=article_days,
    )
    institutional_df = fetch_articles(
        queries=INSTITUTIONAL_TRADE_QUERIES,
        category="institutional",
        max_per_query=max_articles_per_query,
        days=article_days,
    )

    symbol_news_map = fetch_symbol_articles(
        symbols=symbols,
        days=article_days,
        max_per_symbol=max_symbol_articles,
    )

    quote_df, quote_errors = fetch_yahoo_quote_snapshot(tuple(sorted(set(symbols + [BENCHMARK_SYMBOL]))))
    quote_lookup = {
        str(row["Symbol"]).upper(): row
        for _, row in quote_df.iterrows()
    } if not quote_df.empty else {}

    global_daily = aggregate_daily_news(global_df, prefix="global")
    inst_daily = aggregate_daily_news(institutional_df, prefix="inst")

    recommendation_rows: list[dict[str, Any]] = []
    model_diagnostics_rows: list[dict[str, Any]] = []

    for symbol in symbols:
        history_df = fetch_yahoo_chart_history(symbol=symbol, years=5)
        symbol_df = symbol_news_map.get(symbol, pd.DataFrame())
        symbol_daily = aggregate_daily_news(symbol_df, prefix="symbol")

        dataset, latest_features, feature_cols, latest_row = build_feature_dataset(
            history_df=history_df,
            global_daily=global_daily,
            inst_daily=inst_daily,
            symbol_daily=symbol_daily,
            horizon_days=horizon_days,
            breakout_threshold=breakout_threshold,
        )

        train_result = train_breakout_ensemble(dataset, latest_features, feature_cols)

        if train_result.get("error"):
            breakout_prob = fallback_breakout_probability(latest_row)
            model_type = f"heuristic ({train_result.get('error')})"
            model_quality = None
        else:
            breakout_prob = normalize_float(train_result.get("ensemble_prob"), default=0.5)
            model_type = "ensemble_ml"
            model_quality = train_result.get("ensemble_test_auc")
            for diag in train_result.get("model_diagnostics", []):
                model_diagnostics_rows.append(
                    {
                        "Symbol": symbol,
                        "Model": diag.get("model"),
                        "Test Accuracy": diag.get("test_accuracy"),
                        "Test ROC AUC": diag.get("test_roc_auc"),
                        "Latest Breakout Prob": diag.get("latest_breakout_prob"),
                    }
                )

        action = "WATCHOUT"
        if breakout_prob >= 0.70:
            action = "STRONG WATCHOUT"
        elif breakout_prob < 0.52:
            action = "LOW PRIORITY"

        latest_price = normalize_float(quote_lookup.get(symbol, {}).get("Price"), default=np.nan)
        latest_chg = normalize_float(quote_lookup.get(symbol, {}).get("Change %"), default=np.nan)
        latest_news_sent = weighted_sentiment(symbol_df, text_col="Text") if not symbol_df.empty else 0.0
        global_sent = weighted_sentiment(global_df, text_col="Text") if not global_df.empty else 0.0
        inst_sent = weighted_sentiment(institutional_df, text_col="Text") if not institutional_df.empty else 0.0

        ret_20d = normalize_float(latest_row.get("ret_20d"), default=0.0) if latest_row is not None else 0.0
        rsi = normalize_float(latest_row.get("rsi_14"), default=50.0) if latest_row is not None else 50.0
        breakout_dist = normalize_float(latest_row.get("breakout_dist"), default=0.0) if latest_row is not None else 0.0

        confidence = float(np.clip(40 + abs(breakout_prob - 0.5) * 140, 35, 97))
        rationale = []
        if ret_20d > 0.05:
            rationale.append("positive 20D momentum")
        if breakout_dist > -0.02:
            rationale.append("near local highs")
        if latest_news_sent > 0.08:
            rationale.append("supportive symbol news")
        if inst_sent > 0.05:
            rationale.append("institutional tone constructive")
        if not rationale:
            rationale.append("mixed setup")

        recommendation_rows.append(
            {
                "Symbol": symbol,
                "Recommendation": action,
                "Breakout Prob %": round(breakout_prob * 100.0, 2),
                "Confidence %": round(confidence, 1),
                "Model Type": model_type,
                "Model ROC AUC": round(float(model_quality), 3) if model_quality is not None else None,
                "Last Price": round(latest_price, 4) if not np.isnan(latest_price) else None,
                "1D Change %": round(latest_chg, 3) if not np.isnan(latest_chg) else None,
                "20D Return %": round(ret_20d * 100.0, 2),
                "RSI(14)": round(rsi, 2),
                "Symbol Sentiment": round(latest_news_sent, 3),
                "Global Sentiment": round(global_sent, 3),
                "Institutional Sentiment": round(inst_sent, 3),
                "Why": "; ".join(rationale),
            }
        )

    rec_df = pd.DataFrame(recommendation_rows)
    if not rec_df.empty:
        rec_df = rec_df.sort_values(["Breakout Prob %", "Confidence %"], ascending=[False, False]).reset_index(drop=True)

    top5_df = rec_df.head(5).copy() if not rec_df.empty else pd.DataFrame()

    # Market outlook via benchmark model
    benchmark_history = fetch_yahoo_chart_history(symbol=BENCHMARK_SYMBOL, years=6)
    benchmark_symbol_news = symbol_news_map.get(BENCHMARK_SYMBOL, pd.DataFrame())
    benchmark_daily = aggregate_daily_news(benchmark_symbol_news, prefix="symbol")
    bench_dataset, bench_latest, bench_features, bench_latest_row = build_feature_dataset(
        history_df=benchmark_history,
        global_daily=global_daily,
        inst_daily=inst_daily,
        symbol_daily=benchmark_daily,
        horizon_days=horizon_days,
        breakout_threshold=0.0,
    )
    bench_result = train_breakout_ensemble(bench_dataset, bench_latest, bench_features)
    if bench_result.get("error"):
        market_prob_up = fallback_breakout_probability(bench_latest_row)
        market_model_type = f"heuristic ({bench_result.get('error')})"
    else:
        market_prob_up = normalize_float(bench_result.get("ensemble_prob"), default=0.5)
        market_model_type = "ensemble_ml"

    market_outlook = market_outlook_from_probability(market_prob_up)
    req_cagr = required_annualized_return(investment_budget, target_amount, horizon_months)

    gpt_note = ""
    gpt_error = None
    if use_gpt_overlay:
        client = get_openai_client(runtime_key=runtime_openai_key)
        gpt_note, gpt_error = build_gpt_overlay(
            client=client,
            symbols=symbols,
            market_prob_up=market_prob_up,
            top_watch_df=top5_df,
            global_df=global_df,
            inst_df=institutional_df,
            investment_budget=investment_budget,
            target_amount=target_amount,
            horizon_months=horizon_months,
        )
    if not gpt_note:
        gpt_note = build_quant_desk_note(
            market_prob_up=market_prob_up,
            top_watch_df=top5_df,
            investment_budget=investment_budget,
            target_amount=target_amount,
            horizon_months=horizon_months,
        )
        gpt_error = None

    weekly_module = {
        "weekly_scan": pd.DataFrame(),
        "weekly_top5": pd.DataFrame(),
        "geo_summary": {},
        "weekly_note": "",
        "weekly_note_error": None,
        "weekly_engine": "disabled",
    }
    if weekly_module_enabled:
        weekly_module = build_weekly_market_module(
            global_df=global_df,
            institutional_df=institutional_df,
            weekly_horizon_days=weekly_horizon_days,
            max_symbol_news=max(3, int(weekly_symbol_news_items)),
            investment_budget=investment_budget,
            runtime_openai_key=runtime_openai_key,
            use_gpt_overlay=use_gpt_overlay,
        )

    return {
        "run_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "global_articles": global_df,
        "institutional_articles": institutional_df,
        "symbol_news_map": symbol_news_map,
        "quotes": quote_df,
        "quote_errors": quote_errors,
        "recommendations": rec_df,
        "top5": top5_df,
        "model_diagnostics": pd.DataFrame(model_diagnostics_rows),
        "market_prob_up": market_prob_up,
        "market_outlook": market_outlook,
        "market_model_type": market_model_type,
        "required_cagr": req_cagr,
        "gpt_note": gpt_note,
        "gpt_error": gpt_error,
        "use_gpt_overlay": use_gpt_overlay,
        "weekly_module": weekly_module,
    }


def render_result(result: dict[str, Any], investment_budget: float, target_amount: float, horizon_months: int) -> None:
    global_df = result.get("global_articles", pd.DataFrame())
    institutional_df = result.get("institutional_articles", pd.DataFrame())
    top5_df = result.get("top5", pd.DataFrame())
    rec_df = result.get("recommendations", pd.DataFrame())
    diagnostics_df = result.get("model_diagnostics", pd.DataFrame())
    quote_df = result.get("quotes", pd.DataFrame())
    quote_errors = result.get("quote_errors", [])
    weekly_module = result.get("weekly_module", {})
    if not isinstance(weekly_module, dict):
        weekly_module = {}
    weekly_scan_df = weekly_module.get("weekly_scan", pd.DataFrame())
    weekly_top5_df = weekly_module.get("weekly_top5", pd.DataFrame())
    geo_summary = weekly_module.get("geo_summary", {})
    if not isinstance(geo_summary, dict):
        geo_summary = {}
    geo_headlines_df = geo_summary.get("geo_headlines", pd.DataFrame())
    weekly_note = str(weekly_module.get("weekly_note", "")).strip()
    weekly_note_error = weekly_module.get("weekly_note_error")
    weekly_engine = str(weekly_module.get("weekly_engine", "disabled"))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pipeline Run", result.get("run_time", "n/a"))
    m2.metric("Market Outlook", str(result.get("market_outlook", "n/a")))
    m3.metric("Market Prob Up", f"{normalize_float(result.get('market_prob_up')) * 100:.2f}%")
    m4.metric("Req. Annualized Return", f"{normalize_float(result.get('required_cagr')) * 100:.2f}%")

    st.caption(
        f"Budget ${investment_budget:,.0f} -> Target ${target_amount:,.0f} in {horizon_months} months. "
        f"Outlook engine: {result.get('market_model_type', 'n/a')}"
    )

    if quote_errors:
        st.warning("Live quote issues: " + " | ".join(str(err) for err in quote_errors[:5]))

    st.subheader("Step 1: Global Web Articles")
    if global_df.empty:
        st.warning("No global web articles were fetched.")
    else:
        st.write(f"Fetched {len(global_df)} articles from global web themes.")
        view_cols = [c for c in ["Published", "Source", "Query", "Sentiment", "Title", "Link"] if c in global_df.columns]
        out = global_df[view_cols].copy()
        if "Sentiment" in out.columns:
            out["Sentiment"] = out["Sentiment"].map(lambda v: round(float(v), 3))
        st.dataframe(out.head(120), use_container_width=True)

    st.subheader("Step 2: Institutional Trade News (Hedge Funds / Banks / Traders)")
    if institutional_df.empty:
        st.warning("No institutional trade news was fetched.")
    else:
        st.write(f"Fetched {len(institutional_df)} institutional/newsflow articles.")
        view_cols = [c for c in ["Published", "Source", "Query", "Sentiment", "Title", "Link"] if c in institutional_df.columns]
        out = institutional_df[view_cols].copy()
        if "Sentiment" in out.columns:
            out["Sentiment"] = out["Sentiment"].map(lambda v: round(float(v), 3))
        st.dataframe(out.head(120), use_container_width=True)

    st.subheader("Step 3: In-House ML Recommendations")
    if rec_df.empty:
        st.error("Could not produce recommendations. Check web connectivity and symbol universe.")
    else:
        st.markdown("**Top 5 Breakout Watchout Stocks**")
        st.dataframe(top5_df, use_container_width=True)

        st.markdown("**Full Ranked Watchlist**")
        st.dataframe(rec_df, use_container_width=True)

        prob_chart = px.bar(
            top5_df,
            x="Symbol",
            y="Breakout Prob %",
            color="Breakout Prob %",
            title="Top 5 Breakout Probability",
        )
        st.plotly_chart(prob_chart, use_container_width=True)

    st.subheader("Weekly Whole-Market Tactical Module")
    if isinstance(weekly_top5_df, pd.DataFrame) and not weekly_top5_df.empty:
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Weekly Engine", weekly_engine)
        w2.metric("Geo Regime", str(geo_summary.get("regime", "n/a")))
        w3.metric("Geo Pressure", f"{normalize_float(geo_summary.get('geo_pressure')):.3f}")
        w4.metric("Geo Headlines", str(int(normalize_float(geo_summary.get('headline_count')))))

        st.markdown("**Top 5 Weekly Stocks To Enter**")
        st.dataframe(weekly_top5_df, use_container_width=True)

        if isinstance(weekly_scan_df, pd.DataFrame) and not weekly_scan_df.empty:
            st.markdown("**Full Weekly Market Scan**")
            st.dataframe(weekly_scan_df.head(25), use_container_width=True)

            weekly_chart = px.bar(
                weekly_top5_df,
                x="Symbol",
                y="Weekly Breakout Prob %",
                color="Weekly Breakout Prob %",
                title="Weekly Top 5 Breakout Probability",
            )
            st.plotly_chart(weekly_chart, use_container_width=True)
    else:
        st.info("Weekly module is disabled or no weekly candidates were produced.")

    if isinstance(geo_headlines_df, pd.DataFrame) and not geo_headlines_df.empty:
        st.markdown("**Geopolitical Headlines Driving Weekly Risk**")
        geo_view = geo_headlines_df.copy()
        if "Sentiment" in geo_view.columns:
            geo_view["Sentiment"] = pd.to_numeric(geo_view["Sentiment"], errors="coerce").round(3)
        st.dataframe(geo_view.head(30), use_container_width=True)

    st.markdown("**Weekly Strategy Note**")
    if weekly_note:
        st.markdown(weekly_note)
    else:
        st.info("Weekly GPT strategy note unavailable for this run.")
    if weekly_note_error:
        st.caption(f"Weekly note diagnostic: {weekly_note_error}")

    st.subheader("Live Yahoo Snapshot")
    if quote_df.empty:
        st.info("No live Yahoo quote snapshot returned.")
    else:
        quote_view = quote_df.copy()
        for col in ["Price", "Change", "Change %"]:
            if col in quote_view.columns:
                quote_view[col] = pd.to_numeric(quote_view[col], errors="coerce").round(4)
        st.dataframe(quote_view, use_container_width=True)

    st.subheader("Model Diagnostics")
    if diagnostics_df.empty:
        st.info("No ML diagnostics available (likely fallback mode for some symbols).")
    else:
        st.dataframe(diagnostics_df, use_container_width=True)

    st.subheader("GPT In-House Desk Note")
    gpt_note = str(result.get("gpt_note", "")).strip()
    gpt_error = result.get("gpt_error")
    if gpt_note:
        st.markdown(gpt_note)
    else:
        st.info("GPT note unavailable for this run.")
    if gpt_error:
        st.caption(f"GPT diagnostic: {gpt_error}")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    if not HAS_SKLEARN:
        st.warning("scikit-learn not available. The app will run with heuristic fallback scoring.")
    if not HAS_OPENAI:
        st.info("OpenAI SDK not available. GPT desk note will be disabled.")

    if "pipeline_result" not in st.session_state:
        st.session_state["pipeline_result"] = None

    if "watchlist_input" not in st.session_state:
        st.session_state["watchlist_input"] = ", ".join(DEFAULT_WATCHLIST)
    if "runtime_openai_key" not in st.session_state:
        st.session_state["runtime_openai_key"] = ""

    _, center, _ = st.columns([1, 2.2, 1])
    with center:
        st.subheader("Portfolio Universe")
        watchlist_input = st.text_area(
            "Symbols (comma or space separated)",
            key="watchlist_input",
            height=120,
            help="You can edit this list directly or add one symbol at a time below.",
        )
        symbols = parse_symbols(watchlist_input)
        st.caption(f"Add/remove tickers in the symbol box above. Active symbols: {len(symbols)}")

        with st.expander("Model And Portfolio Settings", expanded=True):
            article_days = st.slider(
                "News lookback days",
                min_value=3,
                max_value=21,
                value=st.session_state.get("article_days", 7),
                key="article_days",
            )
            max_articles_per_query = st.slider(
                "Max articles per global/institutional query",
                min_value=10,
                max_value=60,
                value=st.session_state.get("max_articles_per_query", 28),
                key="max_articles_per_query",
            )
            max_symbol_articles = st.slider(
                "Max articles per symbol",
                min_value=8,
                max_value=40,
                value=st.session_state.get("max_symbol_articles", 18),
                key="max_symbol_articles",
            )
            horizon_days = st.slider(
                "Prediction horizon (days)",
                min_value=3,
                max_value=15,
                value=st.session_state.get("horizon_days", 5),
                key="horizon_days",
            )
            breakout_threshold = st.slider(
                "Breakout threshold (future return)",
                min_value=0.03,
                max_value=0.15,
                value=st.session_state.get("breakout_threshold", 0.06),
                step=0.005,
                key="breakout_threshold",
            )
            investment_budget = st.number_input(
                "Starting capital (USD)",
                min_value=1000.0,
                value=float(st.session_state.get("investment_budget", 25000.0)),
                step=500.0,
                key="investment_budget",
            )
            target_amount = st.number_input(
                "Target capital (USD)",
                min_value=1000.0,
                value=float(st.session_state.get("target_amount", 50000.0)),
                step=500.0,
                key="target_amount",
            )
            horizon_months = st.slider(
                "Target horizon (months)",
                min_value=3,
                max_value=36,
                value=st.session_state.get("horizon_months", 18),
                key="horizon_months",
            )
            use_gpt_overlay = st.checkbox(
                "Use GPT desk note",
                value=bool(st.session_state.get("use_gpt_overlay", True)),
                key="use_gpt_overlay",
            )
            runtime_openai_key = st.text_input(
                "OpenAI API key (optional)",
                value=st.session_state.get("runtime_openai_key", ""),
                type="password",
                key="runtime_openai_key",
                help="If provided, this key is used for GPT desk analysis in this session.",
            )
            weekly_module_enabled = st.checkbox(
                "Enable weekly whole-market recommendations",
                value=bool(st.session_state.get("weekly_module_enabled", True)),
                key="weekly_module_enabled",
                help="Scans a broad market universe and recommends 5 tactical weekly entries.",
            )
            weekly_horizon_days = st.slider(
                "Weekly trade horizon (days)",
                min_value=3,
                max_value=10,
                value=st.session_state.get("weekly_horizon_days", 5),
                key="weekly_horizon_days",
            )
            weekly_symbol_news_items = st.slider(
                "Weekly news items per scanned symbol",
                min_value=4,
                max_value=14,
                value=st.session_state.get("weekly_symbol_news_items", 8),
                key="weekly_symbol_news_items",
            )

        run_button = st.button("Refresh Intelligence", type="primary", use_container_width=True)

    first_load = st.session_state.get("pipeline_result") is None
    refresh_requested = bool(run_button or first_load)
    if refresh_requested:
        if len(symbols) < 5:
            st.error("Provide at least 5 symbols in the watchlist.")
        else:
            with st.spinner("Running web/news ingestion and ML pipeline..."):
                st.session_state["pipeline_result"] = run_pipeline(
                    symbols=symbols,
                    article_days=int(article_days),
                    max_articles_per_query=int(max_articles_per_query),
                    max_symbol_articles=int(max_symbol_articles),
                    horizon_days=int(horizon_days),
                    breakout_threshold=float(breakout_threshold),
                    investment_budget=float(investment_budget),
                    target_amount=float(target_amount),
                    horizon_months=int(horizon_months),
                    runtime_openai_key=str(runtime_openai_key or "").strip() or None,
                    use_gpt_overlay=bool(use_gpt_overlay),
                    weekly_module_enabled=bool(weekly_module_enabled),
                    weekly_horizon_days=int(weekly_horizon_days),
                    weekly_symbol_news_items=int(weekly_symbol_news_items),
                )

    result = st.session_state.get("pipeline_result")
    if result:
        render_result(
            result=result,
            investment_budget=float(investment_budget),
            target_amount=float(target_amount),
            horizon_months=int(horizon_months),
        )
    else:
        st.info("Configure watchlist and click 'Run 3-Step Intelligence'.")


if __name__ == "__main__":
    main()
