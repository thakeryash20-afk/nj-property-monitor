from __future__ import annotations

import base64
import json
import math
import os
import re
from pathlib import Path
from urllib.parse import quote, unquote

import pandas as pd
import streamlit as st

from nj_property_monitor_core import (
    DEFAULT_NJ_AREAS,
    annotate_changes,
    apply_deal_scoring,
    area_summary,
    collect_listings,
    fetch_property_profile,
    fetch_nhs_nj_areas,
    format_price,
    listing_key,
    load_latest_snapshot,
    price_midpoint,
    save_snapshot,
)

try:
    from sklearn.ensemble import RandomForestRegressor

    HAS_SKLEARN = True
except Exception:
    RandomForestRegressor = None
    HAS_SKLEARN = False

try:
    from openai import OpenAI

    HAS_OPENAI = True
except Exception:
    OpenAI = None
    HAS_OPENAI = False

st.set_page_config(page_title="NJ Property Monitor", page_icon="🏠", layout="wide")
DEFAULT_MAX_RESULTS = 40
DEFAULT_TIMEOUT_SECONDS = 25
DEFAULT_HOT_DEAL_THRESHOLD = 15.0
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
SECRET_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_WEB_MODEL",
    "ZILLOW_API_KEY",
    "ZILLOW_PROVIDER",
    "ZILLOW_API_URL",
    "ZILLOW_API_HOST",
    "ZILLOW_API_KEY_HEADER",
    "ZILLOW_API_HOST_HEADER",
    "ZILLOW_AREA_PARAM",
    "ZILLOW_AREA_FORMAT",
    "ZILLOW_HASDATA_TYPE",
    "ZILLOW_EXTRA_QUERY_JSON",
)


@st.cache_data(ttl=60 * 60 * 6)
def load_area_map() -> dict[str, str]:
    try:
        return fetch_nhs_nj_areas()
    except Exception:
        return DEFAULT_NJ_AREAS.copy()


def init_state() -> None:
    defaults = {
        "report_df": None,
        "warnings": [],
        "snapshot_path": "",
        "summary_row": {},
        "raw_listings": [],
        "autoload_done": False,
        "autoload_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def hydrate_env_from_streamlit_secrets() -> None:
    try:
        secrets_obj = st.secrets
    except Exception:
        return
    for key in SECRET_ENV_KEYS:
        if os.getenv(key, "").strip():
            continue
        try:
            value = secrets_obj.get(key)  # type: ignore[assignment]
        except Exception:
            value = None
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            text = str(value).strip()
            if text:
                os.environ[key] = text


def run_monitor(
    selected_areas: list[str],
    area_map: dict[str, str],
    include_newhomesource: bool,
    include_zillow: bool,
    max_results: int,
    timeout_seconds: int,
    hot_deal_threshold: float,
) -> None:
    previous_snapshot = load_latest_snapshot()
    listings, warnings = collect_listings(
        area_map=area_map,
        selected_areas=selected_areas,
        include_newhomesource=include_newhomesource,
        include_zillow=include_zillow,
        max_results_per_area=max_results,
        timeout_seconds=timeout_seconds,
    )
    if not listings and isinstance(previous_snapshot, dict):
        previous_listings = previous_snapshot.get("listings")
        if isinstance(previous_listings, list) and previous_listings:
            listings = [dict(item) for item in previous_listings if isinstance(item, dict)]
            warnings.append("Live sources returned no rows. Showing latest published snapshot data.")
    annotated, delta = annotate_changes(listings, previous_snapshot)
    scored_df = apply_deal_scoring(annotated, hot_deal_threshold_pct=hot_deal_threshold)
    snapshot_path = save_snapshot(
        listings=annotated,
        selected_areas=selected_areas,
        enabled_sources=[source for source, enabled in [("NewHomeSource", include_newhomesource), ("Zillow", include_zillow)] if enabled],
        extra_metadata={
            "hot_deal_threshold_pct": hot_deal_threshold,
            "max_results_per_area": max_results,
            "timeout_seconds": timeout_seconds,
        },
    )

    st.session_state["report_df"] = scored_df
    st.session_state["warnings"] = warnings
    st.session_state["snapshot_path"] = str(snapshot_path)
    st.session_state["summary_row"] = {
        "new_count": delta["new_count"],
        "dropped_count": delta["dropped_count"],
        "total_count": int(len(scored_df)),
        "upcoming_count": int(scored_df["is_upcoming"].fillna(False).sum()) if not scored_df.empty else 0,
        "hot_deal_count": int(scored_df["is_hot_deal"].fillna(False).sum()) if not scored_df.empty else 0,
    }
    st.session_state["raw_listings"] = annotated


def autoload_hot_deals(area_map: dict[str, str]) -> None:
    if st.session_state.get("autoload_done"):
        return
    st.session_state["autoload_done"] = True
    st.session_state["autoload_error"] = ""

    selected_areas = list(area_map.keys())
    include_zillow = bool(os.getenv("ZILLOW_API_KEY", "").strip())
    previous_snapshot = load_latest_snapshot()

    listings, warnings = collect_listings(
        area_map=area_map,
        selected_areas=selected_areas,
        include_newhomesource=True,
        include_zillow=include_zillow,
        max_results_per_area=30,
        timeout_seconds=25,
    )
    if not listings and isinstance(previous_snapshot, dict):
        previous_listings = previous_snapshot.get("listings")
        if isinstance(previous_listings, list) and previous_listings:
            listings = [dict(item) for item in previous_listings if isinstance(item, dict)]
            warnings.append("Live sources returned no rows at startup. Showing latest published snapshot data.")
    annotated, delta = annotate_changes(listings, previous_snapshot)
    scored_df = apply_deal_scoring(annotated, hot_deal_threshold_pct=15.0)
    snapshot_path = save_snapshot(
        listings=annotated,
        selected_areas=selected_areas,
        enabled_sources=[name for name, enabled in [("NewHomeSource", True), ("Zillow", include_zillow)] if enabled],
        extra_metadata={
            "autoload": True,
            "hot_deal_threshold_pct": 15.0,
            "max_results_per_area": 30,
            "timeout_seconds": 25,
        },
    )

    st.session_state["report_df"] = scored_df
    st.session_state["warnings"] = warnings
    st.session_state["snapshot_path"] = f"{snapshot_path} (startup auto-snapshot)"
    st.session_state["summary_row"] = {
        "new_count": delta["new_count"],
        "dropped_count": delta["dropped_count"],
        "total_count": int(len(scored_df)),
        "upcoming_count": int(scored_df["is_upcoming"].fillna(False).sum()) if not scored_df.empty else 0,
        "hot_deal_count": int(scored_df["is_hot_deal"].fillna(False).sum()) if not scored_df.empty else 0,
    }
    st.session_state["raw_listings"] = annotated


def header_block() -> None:
    st.title("NJ Weekly Property Intelligence")
    st.caption(
        "Tracks new/upcoming listings and hot deals across selected NJ areas using "
        "NewHomeSource plus optional Zillow API ingestion."
    )


def listing_filter_options(df: pd.DataFrame | None) -> tuple[list[str], list[str], list[str]]:
    if df is None or df.empty:
        return [], [], []
    areas = sorted([value for value in df["area"].dropna().unique().tolist() if str(value).strip()])
    sources = sorted([value for value in df["source"].dropna().unique().tolist() if str(value).strip()])
    cities = sorted([value for value in df["city"].dropna().unique().tolist() if str(value).strip()])
    return areas, sources, cities


def city_options_for_areas(df: pd.DataFrame | None, selected_areas: list[str]) -> list[str]:
    if df is None or df.empty:
        return []
    scoped_df = df
    if selected_areas:
        scoped_df = df[df["area"].isin(selected_areas)]
    return sorted([value for value in scoped_df["city"].dropna().unique().tolist() if str(value).strip()])


def sidebar(
    report_df: pd.DataFrame | None,
) -> dict[str, list[str] | str]:
    with st.sidebar:
        st.subheader("Listing Filters")
        st.caption("Leave filters empty to show all listings.")

        area_options, source_options, _ = listing_filter_options(report_df)
        selected_filter_areas = st.multiselect(
            "Areas",
            options=area_options,
            default=[],
            key="listing_filter_areas",
            placeholder="All areas",
        )
        city_options = city_options_for_areas(report_df, selected_filter_areas)
        if "listing_filter_cities" in st.session_state:
            valid_cities = set(city_options)
            st.session_state["listing_filter_cities"] = [
                city for city in st.session_state["listing_filter_cities"] if city in valid_cities
            ]
        selected_filter_sources = st.multiselect(
            "Sources",
            options=source_options,
            default=[],
            key="listing_filter_sources",
            placeholder="All sources",
        )
        selected_filter_cities = st.multiselect(
            "Cities",
            options=city_options,
            default=[],
            key="listing_filter_cities",
            placeholder="All cities",
        )
        selected_home_types = st.multiselect(
            "Home Type",
            options=["New Homes", "Old/Resale Homes"],
            default=[],
            key="listing_filter_home_type",
            placeholder="All home types",
        )
        selected_signals = st.multiselect(
            "Signals",
            options=["New This Run", "Upcoming", "Hot Deals"],
            default=[],
            key="listing_filter_signals",
            placeholder="All signals",
        )
        keyword = st.text_input(
            "Keyword",
            key="listing_filter_keyword",
            placeholder="Name, city, builder",
        )

    listing_filters: dict[str, list[str] | str] = {
        "areas": selected_filter_areas,
        "sources": selected_filter_sources,
        "cities": selected_filter_cities,
        "home_types": selected_home_types,
        "signals": selected_signals,
        "keyword": keyword,
    }

    return listing_filters


def clean_property_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 .'\-]", "", str(label or "")).strip()
    return cleaned or "Property"


def property_token_from_row(row: pd.Series | dict) -> str:
    source = str(row.get("source", ""))
    listing_id = str(row.get("listing_id", ""))
    return f"{source}::{listing_id}"


def property_page_link(row: pd.Series | dict) -> str:
    token = quote(property_token_from_row(row), safe="")
    label = clean_property_label(str(row.get("name", "Property")))
    payload_obj = {
        "source": row.get("source"),
        "listing_id": row.get("listing_id"),
        "name": row.get("name"),
        "area": row.get("area"),
        "city": row.get("city"),
        "status": row.get("status"),
        "price_low": row.get("price_low"),
        "price_high": row.get("price_high"),
        "beds": row.get("beds"),
        "baths": row.get("baths"),
        "sqft": row.get("sqft"),
        "is_new_construction": row.get("is_new_construction"),
        "deal_discount_pct": row.get("deal_discount_pct"),
        "url": row.get("url"),
    }
    payload_raw = json.dumps(payload_obj, separators=(",", ":"), default=str)
    payload_b64 = base64.urlsafe_b64encode(payload_raw.encode("utf-8")).decode("utf-8").rstrip("=")
    return f"?property={token}&label={label}&payload={payload_b64}"


def add_property_links(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["property_token"] = out.apply(property_token_from_row, axis=1)
    out["name_link"] = out.apply(property_page_link, axis=1)
    return out


def query_property_token() -> str:
    raw = st.query_params.get("property")
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    return unquote(str(raw or "")).strip()


def query_property_payload() -> dict | None:
    raw = st.query_params.get("payload")
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    raw = str(raw or "").strip()
    if not raw:
        return None
    try:
        padded = raw + "=" * (-len(raw) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8")
        payload = json.loads(decoded)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def clear_property_query() -> None:
    st.query_params.clear()


def _listing_from_report_df(property_token: str, report_df: pd.DataFrame | None) -> dict | None:
    if report_df is None or report_df.empty:
        return None
    keys = report_df["source"].astype(str) + "::" + report_df["listing_id"].astype(str)
    matched = report_df[keys == property_token]
    if matched.empty:
        return None
    return matched.iloc[0].to_dict()


def _listing_from_snapshot(property_token: str) -> dict | None:
    snapshot = load_latest_snapshot()
    if not snapshot or not isinstance(snapshot, dict):
        return None
    listings = snapshot.get("listings")
    if not isinstance(listings, list):
        return None
    for listing in listings:
        if isinstance(listing, dict) and listing_key(listing) == property_token:
            return dict(listing)
    return None


def resolve_listing(
    property_token: str,
    report_df: pd.DataFrame | None,
    payload_listing: dict | None = None,
) -> dict | None:
    listing = _listing_from_report_df(property_token, report_df)
    if listing is not None:
        return listing
    listing = _listing_from_snapshot(property_token)
    if listing is not None:
        return listing
    if isinstance(payload_listing, dict):
        return payload_listing
    return None


@st.cache_data(ttl=60 * 60)
def load_property_profile_cached(listing_json: str) -> tuple[dict, list[str]]:
    listing = json.loads(listing_json)
    return fetch_property_profile(listing, timeout_seconds=35)


def mortgage_payment(principal: float, annual_rate_pct: float, years: int) -> float:
    if principal <= 0:
        return 0.0
    months = years * 12
    monthly_rate = annual_rate_pct / 100.0 / 12.0
    if monthly_rate <= 0:
        return principal / months
    factor = (1 + monthly_rate) ** months
    return principal * (monthly_rate * factor) / (factor - 1)


def financial_projection(
    purchase_price: float,
    down_payment_pct: float,
    annual_rate_pct: float,
    loan_years: int,
    tax_annual: float,
    insurance_annual: float,
    hoa_monthly: float,
) -> dict[str, float]:
    down_payment = purchase_price * down_payment_pct / 100.0
    loan_amount = max(purchase_price - down_payment, 0.0)
    monthly_pi = mortgage_payment(loan_amount, annual_rate_pct, loan_years)
    monthly_tax = max(tax_annual, 0.0) / 12.0
    monthly_insurance = max(insurance_annual, 0.0) / 12.0
    monthly_hoa = max(hoa_monthly, 0.0)
    monthly_total = monthly_pi + monthly_tax + monthly_insurance + monthly_hoa
    return {
        "down_payment": down_payment,
        "loan_amount": loan_amount,
        "monthly_pi": monthly_pi,
        "monthly_tax": monthly_tax,
        "monthly_insurance": monthly_insurance,
        "monthly_hoa": monthly_hoa,
        "monthly_total": monthly_total,
    }


def _ml_predicted_price(listing: dict, report_df: pd.DataFrame | None) -> float | None:
    if not HAS_SKLEARN or report_df is None or report_df.empty:
        return None
    train = report_df.copy()
    numeric_cols = ["reference_price", "beds", "baths", "sqft"]
    for col in numeric_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce")
    train = train.dropna(subset=["reference_price"])
    if len(train) < 25:
        return None

    train["beds"] = train["beds"].fillna(train["beds"].median())
    train["baths"] = train["baths"].fillna(train["baths"].median())
    train["sqft"] = train["sqft"].fillna(train["sqft"].median())
    train["area"] = train["area"].fillna("Unknown").astype(str)
    area_dummies = pd.get_dummies(train["area"], prefix="area")
    X_train = pd.concat([train[["beds", "baths", "sqft"]], area_dummies], axis=1)
    y_train = train["reference_price"]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    row_area = str(listing.get("area") or "Unknown")
    row_beds = pd.to_numeric(pd.Series([listing.get("beds")]), errors="coerce").fillna(train["beds"].median()).iloc[0]
    row_baths = pd.to_numeric(pd.Series([listing.get("baths")]), errors="coerce").fillna(train["baths"].median()).iloc[0]
    row_sqft = pd.to_numeric(pd.Series([listing.get("sqft")]), errors="coerce").fillna(train["sqft"].median()).iloc[0]

    X_row = pd.DataFrame([{"beds": row_beds, "baths": row_baths, "sqft": row_sqft}])
    row_area_dummies = pd.get_dummies(pd.Series([row_area]), prefix="area")
    for col in row_area_dummies.columns:
        X_row[col] = row_area_dummies.iloc[0][col]
    X_row = X_row.reindex(columns=X_train.columns, fill_value=0)
    prediction = model.predict(X_row)
    if len(prediction) == 0:
        return None
    return float(prediction[0])


def _area_feature_average_price(listing: dict, report_df: pd.DataFrame | None) -> tuple[float | None, int]:
    if report_df is None or report_df.empty:
        return None, 0

    df = report_df.copy()
    for col in ("reference_price", "beds", "baths", "sqft"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["reference_price"]).copy()
    if df.empty:
        return None, 0

    listing_area = str(listing.get("area") or "").strip()
    listing_is_new = listing.get("is_new_construction")
    listing_beds = pd.to_numeric(pd.Series([listing.get("beds")]), errors="coerce").iloc[0]
    listing_baths = pd.to_numeric(pd.Series([listing.get("baths")]), errors="coerce").iloc[0]
    listing_sqft = pd.to_numeric(pd.Series([listing.get("sqft")]), errors="coerce").iloc[0]

    if listing_area:
        area_df = df[df["area"].astype(str) == listing_area].copy()
        if len(area_df) >= 3:
            df = area_df

    if listing_is_new in (True, False):
        same_type = df[df["is_new_construction"] == listing_is_new]
        if len(same_type) >= 4:
            df = same_type.copy()

    if df.empty:
        return None, 0

    fill_beds = float(df["beds"].median()) if df["beds"].notna().any() else 3.0
    fill_baths = float(df["baths"].median()) if df["baths"].notna().any() else 2.0
    fill_sqft = float(df["sqft"].median()) if df["sqft"].notna().any() else 1800.0

    listing_beds = float(listing_beds) if pd.notna(listing_beds) else fill_beds
    listing_baths = float(listing_baths) if pd.notna(listing_baths) else fill_baths
    listing_sqft = float(listing_sqft) if pd.notna(listing_sqft) and listing_sqft > 0 else fill_sqft

    tiers = [
        {"bed_delta": 0.5, "bath_delta": 0.5, "sqft_low": 0.82, "sqft_high": 1.18, "min_comps": 5},
        {"bed_delta": 1.0, "bath_delta": 1.0, "sqft_low": 0.72, "sqft_high": 1.28, "min_comps": 5},
        {"bed_delta": 1.5, "bath_delta": 1.5, "sqft_low": 0.62, "sqft_high": 1.38, "min_comps": 4},
        {"bed_delta": 2.0, "bath_delta": 2.0, "sqft_low": 0.52, "sqft_high": 1.48, "min_comps": 3},
    ]
    selected: pd.DataFrame | None = None
    fallback: pd.DataFrame | None = None

    for tier in tiers:
        candidate = df.copy()
        candidate = candidate[
            candidate["beds"].fillna(fill_beds).between(
                listing_beds - tier["bed_delta"],
                listing_beds + tier["bed_delta"],
                inclusive="both",
            )
        ]
        candidate = candidate[
            candidate["baths"].fillna(fill_baths).between(
                listing_baths - tier["bath_delta"],
                listing_baths + tier["bath_delta"],
                inclusive="both",
            )
        ]
        if listing_sqft > 0:
            sqft_low = listing_sqft * tier["sqft_low"]
            sqft_high = listing_sqft * tier["sqft_high"]
            candidate = candidate[candidate["sqft"].fillna(fill_sqft).between(sqft_low, sqft_high, inclusive="both")]

        if fallback is None or len(candidate) > len(fallback):
            fallback = candidate.copy()
        if len(candidate) >= tier["min_comps"]:
            selected = candidate.copy()
            break

    scoped = selected if selected is not None else fallback
    if scoped is None or scoped.empty:
        scoped = df.copy()

    scoped = scoped.dropna(subset=["reference_price"]).copy()
    if len(scoped) < 3:
        return None, int(len(scoped))

    comp_beds = scoped["beds"].fillna(fill_beds)
    comp_baths = scoped["baths"].fillna(fill_baths)
    comp_sqft = scoped["sqft"].fillna(fill_sqft)

    bed_dist = (comp_beds - listing_beds).abs() / max(1.0, listing_beds)
    bath_dist = (comp_baths - listing_baths).abs() / max(1.0, listing_baths)
    sqft_dist = (comp_sqft - listing_sqft).abs() / max(500.0, listing_sqft)
    distance = 0.50 * sqft_dist + 0.28 * bed_dist + 0.22 * bath_dist
    similarity = 1.0 / (1.0 + distance)
    sim_total = float(similarity.sum())
    if sim_total <= 0:
        return None, int(len(scoped))

    fair_value = float((similarity * scoped["reference_price"]).sum() / sim_total)
    return fair_value, int(len(scoped))


def _feature_similarity_fair_value(listing: dict, report_df: pd.DataFrame | None) -> tuple[float | None, int]:
    if report_df is None or report_df.empty:
        return None, 0

    df = report_df.copy()
    for col in ("reference_price", "beds", "baths", "sqft"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["reference_price"]).copy()
    if df.empty:
        return None, 0

    listing_area = str(listing.get("area") or "")
    listing_is_new = listing.get("is_new_construction")
    listing_beds = pd.to_numeric(pd.Series([listing.get("beds")]), errors="coerce").iloc[0]
    listing_baths = pd.to_numeric(pd.Series([listing.get("baths")]), errors="coerce").iloc[0]
    listing_sqft = pd.to_numeric(pd.Series([listing.get("sqft")]), errors="coerce").iloc[0]

    if listing_area:
        area_only = df[df["area"] == listing_area]
        if len(area_only) >= 8:
            df = area_only.copy()

    if listing_is_new in (True, False):
        same_type = df[df["is_new_construction"] == listing_is_new]
        if len(same_type) >= 8:
            df = same_type.copy()

    if len(df) < 6:
        return None, int(len(df))

    fill_beds = float(df["beds"].median()) if df["beds"].notna().any() else 3.0
    fill_baths = float(df["baths"].median()) if df["baths"].notna().any() else 2.0
    fill_sqft = float(df["sqft"].median()) if df["sqft"].notna().any() else 1800.0

    listing_beds = float(listing_beds) if pd.notna(listing_beds) else fill_beds
    listing_baths = float(listing_baths) if pd.notna(listing_baths) else fill_baths
    listing_sqft = float(listing_sqft) if pd.notna(listing_sqft) and listing_sqft > 0 else fill_sqft

    comp_beds = df["beds"].fillna(fill_beds)
    comp_baths = df["baths"].fillna(fill_baths)
    comp_sqft = df["sqft"].fillna(fill_sqft)

    bed_dist = (comp_beds - listing_beds).abs() / max(1.0, listing_beds)
    bath_dist = (comp_baths - listing_baths).abs() / max(1.0, listing_baths)
    sqft_dist = (comp_sqft - listing_sqft).abs() / max(500.0, listing_sqft)

    area_penalty = 0.0
    if listing_area:
        area_penalty = (~(df["area"].astype(str) == listing_area)).astype(float) * 0.25
    type_penalty = 0.0
    if listing_is_new in (True, False):
        type_penalty = (~(df["is_new_construction"] == listing_is_new)).astype(float) * 0.2

    distance = 0.52 * sqft_dist + 0.26 * bed_dist + 0.22 * bath_dist + area_penalty + type_penalty
    similarity = 1.0 / (1.0 + distance)
    sim_total = float(similarity.sum())
    if sim_total <= 0:
        return None, int(len(df))
    weighted = (similarity * df["reference_price"]).sum() / sim_total

    return float(weighted), int(len(df))


def comparable_market_context(listing: dict, report_df: pd.DataFrame | None) -> dict:
    if report_df is None or report_df.empty:
        return {"comparable_count": 0, "sample_comparables": []}

    df = report_df.copy()
    for col in ("reference_price", "beds", "baths", "sqft", "deal_discount_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    area = str(listing.get("area") or "")
    listing_is_new = listing.get("is_new_construction")
    listing_beds = pd.to_numeric(pd.Series([listing.get("beds")]), errors="coerce").iloc[0]
    listing_baths = pd.to_numeric(pd.Series([listing.get("baths")]), errors="coerce").iloc[0]
    listing_sqft = pd.to_numeric(pd.Series([listing.get("sqft")]), errors="coerce").iloc[0]
    listing_price = price_midpoint(listing.get("price_low"), listing.get("price_high"))

    scoped = df[df["area"] == area].copy() if area else df.copy()
    if bool(scoped.empty):
        scoped = df.copy()

    if listing_is_new in (True, False):
        same_type = scoped[scoped["is_new_construction"] == listing_is_new]
        if len(same_type) >= 5:
            scoped = same_type

    if pd.notna(listing_beds):
        bed_match = scoped[scoped["beds"].between(listing_beds - 1, listing_beds + 1, inclusive="both")]
        if len(bed_match) >= 6:
            scoped = bed_match

    if pd.notna(listing_baths):
        bath_match = scoped[scoped["baths"].between(listing_baths - 1, listing_baths + 1, inclusive="both")]
        if len(bath_match) >= 6:
            scoped = bath_match

    if pd.notna(listing_sqft) and listing_sqft > 0:
        sqft_low, sqft_high = listing_sqft * 0.65, listing_sqft * 1.35
        sqft_match = scoped[scoped["sqft"].between(sqft_low, sqft_high, inclusive="both")]
        if len(sqft_match) >= 6:
            scoped = sqft_match

    scoped = scoped.dropna(subset=["reference_price"]).copy()
    if scoped.empty:
        return {"comparable_count": 0, "sample_comparables": []}

    comp_count = int(len(scoped))
    median_price = float(scoped["reference_price"].median())
    p25 = float(scoped["reference_price"].quantile(0.25))
    p75 = float(scoped["reference_price"].quantile(0.75))
    avg_discount = float(scoped["deal_discount_pct"].dropna().mean()) if scoped["deal_discount_pct"].notna().any() else None
    hot_ratio = float((scoped["is_hot_deal"] == True).sum() / comp_count)
    new_home_ratio = float((scoped["is_new_construction"] == True).sum() / comp_count)
    price_gap_pct = None
    if listing_price and median_price > 0:
        price_gap_pct = (median_price - listing_price) / median_price * 100.0

    sample_cols = ["name", "city", "reference_price", "beds", "baths", "sqft", "deal_discount_pct", "source"]
    sample_df = scoped[sample_cols].copy().sort_values("reference_price", ascending=True).head(15)
    sample_rows = []
    for row in sample_df.to_dict(orient="records"):
        sample_rows.append(
            {
                "name": row.get("name"),
                "city": row.get("city"),
                "price": row.get("reference_price"),
                "beds": row.get("beds"),
                "baths": row.get("baths"),
                "sqft": row.get("sqft"),
                "discount_pct": row.get("deal_discount_pct"),
                "source": row.get("source"),
            }
        )

    feature_fair_value, feature_comp_count = _feature_similarity_fair_value(listing=listing, report_df=report_df)
    area_feature_avg_price, area_feature_comps = _area_feature_average_price(listing=listing, report_df=report_df)

    return {
        "comparable_count": comp_count,
        "median_price": round(median_price, 2),
        "price_p25": round(p25, 2),
        "price_p75": round(p75, 2),
        "avg_discount_pct": round(avg_discount, 2) if avg_discount is not None else None,
        "hot_deal_ratio": round(hot_ratio, 3),
        "new_home_ratio": round(new_home_ratio, 3),
        "listing_vs_comp_median_pct": round(price_gap_pct, 2) if price_gap_pct is not None else None,
        "feature_similarity_fair_value": round(feature_fair_value, 2) if feature_fair_value is not None else None,
        "feature_similarity_comps": feature_comp_count,
        "area_feature_avg_price": round(area_feature_avg_price, 2) if area_feature_avg_price is not None else None,
        "area_feature_comps": area_feature_comps,
        "feature_inputs_used": ["sqft", "beds", "baths", "area", "is_new_construction"],
        "sample_comparables": sample_rows,
    }


@st.cache_data(ttl=60 * 30, show_spinner=False)
def genai_deal_rating_cached(
    listing_json: str,
    profile_json: str,
    market_context_json: str,
    model_name: str,
) -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not HAS_OPENAI or not api_key:
        return {"warning": "OPENAI_API_KEY is not configured; using heuristic rating."}

    listing = json.loads(listing_json)
    profile = json.loads(profile_json)
    market_context = json.loads(market_context_json)

    system_prompt = (
        "You are a senior residential real-estate valuation analyst for New Jersey. "
        "Rate a property deal quality based on listing details, school/crime context, "
        "builder incentives, and comparable market context. "
        "Return strict JSON only."
    )
    user_payload = {
        "task": "Classify this property as Stealer, Good Deal, or Bad Deal.",
        "scoring_guidance": {
            "score_range": "0-100",
            "labels": {
                "Stealer": "materially undervalued or strong incentives with manageable risk",
                "Good Deal": "fair-to-good value with acceptable risk",
                "Bad Deal": "overpriced or risky compared with comps/context",
            },
            "be_conservative": True,
        },
        "property_listing": listing,
        "property_profile": profile,
        "market_context": market_context,
        "output_schema": {
            "label": "Stealer|Good Deal|Bad Deal",
            "score": "number 0-100",
            "confidence": "number 0-1",
            "estimated_fair_value": "number or null",
            "value_gap_pct": "number or null",
            "summary": "short string",
            "positives": ["list up to 4 bullets"],
            "risks": ["list up to 4 bullets"],
        },
    }

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.15,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, default=str)},
            ],
        )
        content = response.choices[0].message.content if response and response.choices else ""
        parsed = json.loads(content or "{}")
    except Exception as exc:
        raw = str(exc).lower()
        if "quota" in raw or "insufficient_quota" in raw:
            reason = "OpenAI quota exceeded"
        elif "timeout" in raw:
            reason = "request timed out"
        elif "auth" in raw or "api key" in raw:
            reason = "API authentication failed"
        else:
            reason = "temporary API issue"
        return {"warning": f"GenAI rating unavailable ({reason}). Using heuristic rating."}

    label = str(parsed.get("label", "")).strip()
    if label not in {"Stealer", "Good Deal", "Bad Deal"}:
        label = "Good Deal"

    score = pd.to_numeric(pd.Series([parsed.get("score")]), errors="coerce").fillna(60.0).iloc[0]
    score = float(max(0.0, min(100.0, score)))

    confidence = pd.to_numeric(pd.Series([parsed.get("confidence")]), errors="coerce").fillna(0.6).iloc[0]
    confidence = float(max(0.0, min(1.0, confidence)))

    fair_value = pd.to_numeric(pd.Series([parsed.get("estimated_fair_value")]), errors="coerce").iloc[0]
    fair_value = float(fair_value) if pd.notna(fair_value) else None

    value_gap = pd.to_numeric(pd.Series([parsed.get("value_gap_pct")]), errors="coerce").iloc[0]
    value_gap = float(value_gap) if pd.notna(value_gap) else None

    positives = parsed.get("positives")
    if not isinstance(positives, list):
        positives = []
    positives = [str(item).strip() for item in positives if str(item).strip()][:4]

    risks = parsed.get("risks")
    if not isinstance(risks, list):
        risks = []
    risks = [str(item).strip() for item in risks if str(item).strip()][:4]

    summary = str(parsed.get("summary") or "").strip()
    return {
        "label": label,
        "score": round(score, 1),
        "confidence": round(confidence, 2),
        "estimated_fair_value": fair_value,
        "value_gap_pct": value_gap,
        "summary": summary,
        "positives": positives,
        "risks": risks,
    }


def heuristic_deal_rating(listing: dict, profile: dict, report_df: pd.DataFrame | None) -> dict[str, float | str | None]:
    actual_price = price_midpoint(listing.get("price_low"), listing.get("price_high"))
    if actual_price is None:
        return {
            "label": "Insufficient Data",
            "score": None,
            "ml_predicted_price": None,
            "raw_ml_predicted_price": None,
            "area_feature_avg_price": None,
            "area_feature_comps": 0,
            "feature_similarity_fair_value": None,
            "raw_feature_similarity_fair_value": None,
            "feature_similarity_comps": 0,
            "blended_fair_value": None,
            "ml_discount_pct": None,
            "school_rating": profile.get("school_rating_avg"),
            "crime_rating": profile.get("crime_rating_proxy"),
        }

    raw_ml_predicted_price = _ml_predicted_price(listing, report_df)
    area_feature_avg_price, area_feature_comps = _area_feature_average_price(listing=listing, report_df=report_df)
    raw_feature_fair_value, feature_comp_count = _feature_similarity_fair_value(listing=listing, report_df=report_df)

    adjusted_raw_ml = raw_ml_predicted_price
    adjusted_feature_fair = raw_feature_fair_value
    if area_feature_avg_price is not None and area_feature_avg_price > 0:
        if raw_ml_predicted_price is not None:
            ml_low = area_feature_avg_price * 0.70
            ml_high = area_feature_avg_price * 1.40
            adjusted_raw_ml = min(max(raw_ml_predicted_price, ml_low), ml_high)
        if raw_feature_fair_value is not None:
            feature_low = area_feature_avg_price * 0.70
            feature_high = area_feature_avg_price * 1.30
            adjusted_feature_fair = min(max(raw_feature_fair_value, feature_low), feature_high)

    # Area average for similar homes (sqft/beds/baths) becomes the anchor for ML fair value.
    ml_fair_price = adjusted_raw_ml
    if area_feature_avg_price is not None and adjusted_raw_ml is not None:
        ml_fair_price = 0.75 * area_feature_avg_price + 0.25 * adjusted_raw_ml
    elif area_feature_avg_price is not None:
        ml_fair_price = area_feature_avg_price

    blended_inputs: list[tuple[float, float]] = []
    if area_feature_avg_price is not None:
        blended_inputs.append((0.55, area_feature_avg_price))
    if adjusted_raw_ml is not None:
        blended_inputs.append((0.30, adjusted_raw_ml))
    if adjusted_feature_fair is not None:
        blended_inputs.append((0.15, adjusted_feature_fair))

    blended_fair_value = None
    if blended_inputs:
        weight_total = sum(weight for weight, _ in blended_inputs)
        if weight_total > 0:
            blended_fair_value = sum(weight * value for weight, value in blended_inputs) / weight_total
    elif ml_fair_price is not None:
        blended_fair_value = ml_fair_price

    ml_discount_pct = None
    if blended_fair_value and blended_fair_value > 0:
        ml_discount_pct = (blended_fair_value - actual_price) / blended_fair_value * 100.0

    school_rating = float(profile.get("school_rating_avg") or 6.0)
    crime_rating = float(profile.get("crime_rating_proxy") or 6.0)
    offer_count = len(profile.get("builder_offers") or [])

    value_signal = ml_discount_pct if ml_discount_pct is not None else float(listing.get("deal_discount_pct") or 0.0)
    value_score = max(0.0, min(1.0, (value_signal + 12.0) / 28.0))
    school_score = max(0.0, min(1.0, school_rating / 10.0))
    crime_score = max(0.0, min(1.0, crime_rating / 10.0))
    offer_score = max(0.0, min(1.0, offer_count / 3.0))

    score = 0.52 * value_score + 0.20 * school_score + 0.18 * crime_score + 0.10 * offer_score
    if score >= 0.78:
        label = "Stealer"
    elif score >= 0.58:
        label = "Good Deal"
    else:
        label = "Bad Deal"

    return {
        "label": label,
        "score": round(score * 100.0, 1),
        "ml_predicted_price": ml_fair_price,
        "raw_ml_predicted_price": raw_ml_predicted_price,
        "area_feature_avg_price": area_feature_avg_price,
        "area_feature_comps": area_feature_comps,
        "feature_similarity_fair_value": adjusted_feature_fair,
        "raw_feature_similarity_fair_value": raw_feature_fair_value,
        "feature_similarity_comps": feature_comp_count,
        "blended_fair_value": blended_fair_value,
        "ml_discount_pct": ml_discount_pct,
        "school_rating": round(school_rating, 2),
        "crime_rating": round(crime_rating, 2),
        "offer_count": offer_count,
    }


def ai_ml_deal_rating(listing: dict, profile: dict, report_df: pd.DataFrame | None) -> dict[str, float | str | None]:
    heuristic = heuristic_deal_rating(listing=listing, profile=profile, report_df=report_df)
    market_context = comparable_market_context(listing=listing, report_df=report_df)

    if heuristic.get("score") is None:
        heuristic["engine"] = "heuristic"
        heuristic["summary"] = "Insufficient priced data for rating."
        heuristic["positives"] = []
        heuristic["risks"] = []
        heuristic["confidence"] = None
        return heuristic

    genai_payload = genai_deal_rating_cached(
        listing_json=json.dumps(listing, default=str, sort_keys=True),
        profile_json=json.dumps(profile, default=str, sort_keys=True),
        market_context_json=json.dumps(market_context, default=str, sort_keys=True),
        model_name=DEFAULT_OPENAI_MODEL,
    )
    if genai_payload.get("warning"):
        return {
            **heuristic,
            "engine": "heuristic",
            "summary": "Fallback heuristic rating used (GenAI unavailable).",
            "positives": [],
            "risks": [str(genai_payload.get("warning"))],
            "confidence": None,
        }

    return {
        "label": genai_payload.get("label") or heuristic.get("label"),
        "score": genai_payload.get("score") if genai_payload.get("score") is not None else heuristic.get("score"),
        "confidence": genai_payload.get("confidence"),
        "ml_predicted_price": heuristic.get("ml_predicted_price"),
        "raw_ml_predicted_price": heuristic.get("raw_ml_predicted_price"),
        "area_feature_avg_price": heuristic.get("area_feature_avg_price"),
        "area_feature_comps": heuristic.get("area_feature_comps"),
        "feature_similarity_fair_value": heuristic.get("feature_similarity_fair_value"),
        "raw_feature_similarity_fair_value": heuristic.get("raw_feature_similarity_fair_value"),
        "feature_similarity_comps": heuristic.get("feature_similarity_comps"),
        "blended_fair_value": heuristic.get("blended_fair_value"),
        "ml_discount_pct": heuristic.get("ml_discount_pct"),
        "school_rating": heuristic.get("school_rating"),
        "crime_rating": heuristic.get("crime_rating"),
        "offer_count": heuristic.get("offer_count"),
        "estimated_fair_value": genai_payload.get("estimated_fair_value"),
        "value_gap_pct": genai_payload.get("value_gap_pct"),
        "summary": genai_payload.get("summary") or "",
        "positives": genai_payload.get("positives") or [],
        "risks": genai_payload.get("risks") or [],
        "engine": "genai",
        "comparable_count": market_context.get("comparable_count"),
    }


def should_show_ui_warning(message: str) -> bool:
    text = str(message or "").strip().lower()
    if not text:
        return False
    if "zillow request failed" in text or "zillow failed for" in text or "zillow property detail request failed" in text:
        return False
    if "newhomesource failed for" in text and "403" in text:
        return False
    if "newhomesource property detail request failed" in text and "403" in text:
        return False
    if "403 client error: forbidden for url: https://www.newhomesource.com" in text:
        return False
    return True


def render_property_detail_page(
    property_token: str,
    report_df: pd.DataFrame | None,
    payload_listing: dict | None = None,
) -> None:
    listing = resolve_listing(property_token, report_df, payload_listing=payload_listing)
    if listing is None:
        st.error("Property not found. It may no longer be in the latest dataset.")
        if st.button("Back to Dashboard"):
            clear_property_query()
            st.rerun()
        return

    if st.button("Back to Dashboard"):
        clear_property_query()
        st.rerun()

    st.title(str(listing.get("name") or "Property"))
    subtitle = " | ".join(
        [
            str(listing.get("source") or ""),
            str(listing.get("area") or ""),
            str(listing.get("city") or ""),
        ]
    )
    st.caption(subtitle)
    if listing.get("url"):
        st.link_button("Open Original Listing", str(listing.get("url")))

    listing_json = json.dumps(listing, default=str, sort_keys=True)
    profile, warnings = load_property_profile_cached(listing_json)
    for warning in warnings:
        if should_show_ui_warning(warning):
            st.warning(warning)

    model_df = report_df
    if model_df is None or model_df.empty:
        snapshot = load_latest_snapshot() or {}
        snapshot_listings = snapshot.get("listings") if isinstance(snapshot, dict) else None
        if isinstance(snapshot_listings, list) and snapshot_listings:
            model_df = apply_deal_scoring(snapshot_listings)

    actual_price = price_midpoint(listing.get("price_low"), listing.get("price_high"))
    deal_discount = float(listing.get("deal_discount_pct") or 0.0)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("List Price", format_price(actual_price))
    col2.metric("Deal Discount", f"{deal_discount:.1f}%")
    col3.metric("Beds / Baths", f"{listing.get('beds', '')} / {listing.get('baths', '')}")
    col4.metric("SqFt", f"{int(float(listing.get('sqft') or 0)):,}" if listing.get("sqft") else "-")

    if profile.get("description"):
        st.markdown("**Property Description**")
        st.write(profile.get("description"))

    if listing.get("is_new_construction"):
        st.markdown("**Builder Offers / Incentives**")
        offers = profile.get("builder_offers") or []
        if offers:
            for offer in offers:
                st.write(f"- {offer}")
        else:
            st.info("No explicit builder incentive text found on the source listing.")

    st.markdown("**Home Variants & Pricing**")
    variants = profile.get("home_variants") if isinstance(profile.get("home_variants"), list) else []
    if variants:
        variants_df = pd.DataFrame(variants).copy()
        if "price" in variants_df.columns:
            variants_df["price"] = pd.to_numeric(variants_df["price"], errors="coerce")
        display_cols = [
            col
            for col in ["variant_type", "name", "price", "beds", "baths", "sqft", "home_style", "url"]
            if col in variants_df.columns
        ]
        st.dataframe(
            variants_df[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "price": st.column_config.NumberColumn("Price", format="$%.0f"),
                "beds": st.column_config.NumberColumn("Beds", format="%.1f"),
                "baths": st.column_config.NumberColumn("Baths", format="%.1f"),
                "sqft": st.column_config.NumberColumn("SqFt", format="%.0f"),
                "url": st.column_config.LinkColumn("Variant Link"),
            },
        )
    else:
        st.info("No additional floor-plan or quick move-in variants were found for this property.")

    st.markdown("**Affordability Calculator**")
    default_tax = float(profile.get("tax_annual_amount") or (actual_price or 0) * 0.022)
    default_insurance = float(profile.get("insurance_annual_estimate") or (actual_price or 0) * 0.0035)
    default_hoa = float(profile.get("hoa_fee_monthly") or 0.0)

    in_col1, in_col2, in_col3 = st.columns(3)
    down_payment_pct = in_col1.number_input("Down Payment (%)", min_value=0.0, max_value=90.0, value=20.0, step=1.0)
    annual_rate = in_col2.number_input("Interest Rate (%)", min_value=1.0, max_value=15.0, value=6.75, step=0.05)
    loan_years = int(in_col3.selectbox("Loan Term (Years)", options=[15, 20, 30], index=2))

    in_col4, in_col5, in_col6 = st.columns(3)
    tax_annual = in_col4.number_input("Estimated Property Tax (Annual $)", min_value=0.0, value=float(default_tax), step=100.0)
    insurance_annual = in_col5.number_input(
        "Possible Home Insurance (Annual $)", min_value=0.0, value=float(default_insurance), step=100.0
    )
    hoa_monthly = in_col6.number_input("HOA (Monthly $)", min_value=0.0, value=float(default_hoa), step=25.0)

    projection = financial_projection(
        purchase_price=float(actual_price or 0.0),
        down_payment_pct=float(down_payment_pct),
        annual_rate_pct=float(annual_rate),
        loan_years=loan_years,
        tax_annual=float(tax_annual),
        insurance_annual=float(insurance_annual),
        hoa_monthly=float(hoa_monthly),
    )
    out_col1, out_col2, out_col3, out_col4, out_col5 = st.columns(5)
    out_col1.metric("Down Payment", format_price(projection["down_payment"]))
    out_col2.metric("Loan Amount", format_price(projection["loan_amount"]))
    out_col3.metric("Monthly PI", format_price(projection["monthly_pi"]))
    out_col4.metric("Monthly Taxes+Insurance", format_price(projection["monthly_tax"] + projection["monthly_insurance"]))
    out_col5.metric("All-in Monthly", format_price(projection["monthly_total"]))

    st.markdown("**AI/ML Deal Rating**")
    rating = ai_ml_deal_rating(listing=listing, profile=profile, report_df=model_df)
    if rating.get("score") is None:
        st.info("Not enough data to compute AI/ML deal rating for this property.")
    else:
        verdict = f"{rating['label']} ({rating['score']}/100)"
        engine = str(rating.get("engine") or "heuristic").upper()
        confidence = rating.get("confidence")
        if confidence is not None:
            verdict = f"{verdict} | Confidence {float(confidence) * 100:.0f}%"
        st.metric("Deal Verdict", verdict)
        st.caption(f"Rating Engine: {engine}")

        expl_col1, expl_col2, expl_col3, expl_col4 = st.columns(4)
        expl_col1.metric("ML Fair Price (Area-Adjusted)", format_price(rating.get("ml_predicted_price")))
        area_avg_price = rating.get("area_feature_avg_price")
        area_avg_comps = rating.get("area_feature_comps")
        expl_col2.metric("Area Similar Avg Price", format_price(area_avg_price) if area_avg_price else "n/a")
        if area_avg_comps:
            expl_col2.caption(f"Based on {int(area_avg_comps)} area comps matching SqFt/Beds/Baths")
        feature_fair = rating.get("feature_similarity_fair_value")
        feature_comps = rating.get("feature_similarity_comps")
        expl_col3.metric("Feature Fair Value", format_price(feature_fair) if feature_fair else "n/a")
        if feature_comps:
            expl_col3.caption(f"Based on {int(feature_comps)} comps using SqFt, Beds, Baths")
        blended_fair = rating.get("blended_fair_value")
        expl_col4.metric("Blended Fair Value", format_price(blended_fair) if blended_fair else "n/a")

        expl_col5, expl_col6, expl_col7 = st.columns(3)
        ml_gap = rating.get("ml_discount_pct")
        expl_col5.metric("Discount vs Fair Value", f"{ml_gap:.1f}%" if ml_gap is not None else "n/a")
        expl_col6.metric(
            "School / Crime Proxy",
            f"{rating.get('school_rating', 0):.1f} / {rating.get('crime_rating', 0):.1f}",
        )
        genai_fair = rating.get("estimated_fair_value")
        expl_col7.metric("GenAI Fair Value", format_price(genai_fair) if genai_fair else "n/a")

        if rating.get("summary"):
            st.write(str(rating.get("summary")))
        positives = rating.get("positives") or []
        risks = rating.get("risks") or []
        if positives:
            st.markdown("**Strengths**")
            for item in positives:
                st.write(f"- {item}")
        if risks:
            st.markdown("**Risks**")
            for item in risks:
                st.write(f"- {item}")

    school_rows = profile.get("school_rows")
    if isinstance(school_rows, list) and school_rows:
        st.markdown("**Schools Snapshot**")
        school_df = pd.DataFrame(school_rows)
        display_cols = [col for col in ["name", "rating", "grades", "distance_miles", "type", "link"] if col in school_df.columns]
        st.dataframe(
            school_df[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "rating": st.column_config.NumberColumn("Rating", format="%.1f"),
                "distance_miles": st.column_config.NumberColumn("Distance (mi)", format="%.1f"),
                "link": st.column_config.LinkColumn("School Link"),
            },
        )

    with st.expander("Raw Property Data"):
        st.code(json.dumps({"listing": listing, "profile": profile, "rating": rating}, indent=2, default=str), language="json")


def warning_block(warnings: list[str]) -> None:
    if not warnings:
        return
    for warning in dict.fromkeys(warnings):
        if should_show_ui_warning(warning):
            st.warning(warning)


def data_source_diagnostics_block(report_df: pd.DataFrame | None, warnings: list[str], include_zillow: bool) -> None:
    if report_df is not None and not report_df.empty:
        return

    raw_warnings = [str(item or "").lower() for item in warnings]
    nhs_forbidden = any("newhomesource failed for" in item and "403" in item for item in raw_warnings)
    zillow_failed = any(
        "zillow request failed" in item or "zillow failed for" in item or "zillow property detail request failed" in item
        for item in raw_warnings
    )
    zillow_missing = not include_zillow

    st.warning("No listings were returned from active sources.")
    if nhs_forbidden:
        st.info("NewHomeSource is blocking this cloud environment right now.")
    if zillow_missing:
        st.info("Zillow is disabled. Add `ZILLOW_API_KEY` in Streamlit app Secrets to enable Zillow data.")
    elif zillow_failed:
        st.info("Zillow API requests are failing or rate-limited. Verify API key, quota, and provider endpoint.")
    if not nhs_forbidden and not zillow_missing and not zillow_failed:
        st.info("Try clicking 'Refresh Listings' and broadening filters (areas/sources/cities).")


def metrics_block(summary_row: dict[str, int], snapshot_path: str) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Listings", summary_row.get("total_count", 0))
    col2.metric("New This Run", summary_row.get("new_count", 0))
    col3.metric("Upcoming", summary_row.get("upcoming_count", 0))
    col4.metric("Hot Deals", summary_row.get("hot_deal_count", 0))
    col5.metric("Dropped Since Last Run", summary_row.get("dropped_count", 0))
    if snapshot_path:
        st.caption(f"Snapshot saved: `{snapshot_path}`")


def apply_listing_filters(df: pd.DataFrame, listing_filters: dict[str, list[str] | str]) -> pd.DataFrame:
    if df.empty:
        return df

    view_df = df.copy()
    selected_areas = listing_filters.get("areas", [])
    selected_sources = listing_filters.get("sources", [])
    selected_cities = listing_filters.get("cities", [])
    selected_home_types = listing_filters.get("home_types", [])
    selected_signals = listing_filters.get("signals", [])
    keyword = str(listing_filters.get("keyword", "")).strip().lower()

    if selected_areas:
        view_df = view_df[view_df["area"].isin(selected_areas)]
    if selected_sources:
        view_df = view_df[view_df["source"].isin(selected_sources)]
    if selected_cities:
        view_df = view_df[view_df["city"].isin(selected_cities)]

    if selected_home_types:
        home_mask = pd.Series(False, index=view_df.index)
        if "New Homes" in selected_home_types:
            home_mask = home_mask | (view_df["is_new_construction"] == True)
        if "Old/Resale Homes" in selected_home_types:
            home_mask = home_mask | (view_df["is_new_construction"] == False)
        view_df = view_df[home_mask]

    if selected_signals:
        signal_mask = pd.Series(False, index=view_df.index)
        if "New This Run" in selected_signals:
            signal_mask = signal_mask | (view_df["is_new_this_week"] == True)
        if "Upcoming" in selected_signals:
            signal_mask = signal_mask | (view_df["is_upcoming"] == True)
        if "Hot Deals" in selected_signals:
            signal_mask = signal_mask | (view_df["is_hot_deal"] == True)
        view_df = view_df[signal_mask]

    if keyword:
        text_cols = view_df[["name", "city", "builder"]].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        view_df = view_df[text_cols.str.contains(keyword, na=False)]

    return view_df


def top_hot_deals_block(df: pd.DataFrame) -> None:
    st.subheader("Top Hot Deals Right Now")
    if df.empty:
        st.info("No listings loaded yet.")
        return
    hot = df[(df["is_hot_deal"] == True) & (df["is_new_construction"] == True)].copy()
    if hot.empty:
        st.info("No new-home hot deals at the current threshold.")
        return

    hot = hot.sort_values(["deal_discount_pct", "is_new_this_week"], ascending=[False, False]).head(10)
    hot_view = add_property_links(hot)[
        ["name_link", "source", "area", "city", "status", "reference_price_text", "deal_discount_pct", "url"]
    ].copy()
    st.dataframe(
        hot_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "name_link": st.column_config.LinkColumn(
                "Property",
                display_text=r".*[?&]label=([^&]+).*",
            ),
            "deal_discount_pct": st.column_config.NumberColumn("Deal Discount %", format="%.1f"),
            "url": st.column_config.LinkColumn("Listing URL"),
        },
    )


def digest_block(df: pd.DataFrame) -> None:
    summary_df = area_summary(df)
    if summary_df.empty:
        st.info("No area summary available yet.")
        return
    view = summary_df[["area", "listings", "new_this_week", "upcoming", "hot_deals", "median_price_text"]]
    st.subheader("Area Summary")
    st.dataframe(view, use_container_width=True, hide_index=True)

    chart_df = summary_df.set_index("area")[["new_this_week", "upcoming", "hot_deals"]]
    st.bar_chart(chart_df, use_container_width=True)


def listings_block(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No listings returned. Try different areas/sources.")
        return

    st.subheader("Listings")
    display_cols = [
        "name_link",
        "source",
        "area",
        "city",
        "status",
        "price_low_text",
        "price_high_text",
        "beds",
        "baths",
        "sqft",
        "is_new_construction",
        "deal_discount_pct",
        "is_hot_deal",
        "is_upcoming",
        "is_new_this_week",
        "url",
    ]
    display_df = add_property_links(df)[display_cols].copy()
    display_df = display_df.sort_values(
        by=["is_new_this_week", "is_hot_deal", "deal_discount_pct"], ascending=[False, False, False]
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "name_link": st.column_config.LinkColumn(
                "Property",
                display_text=r".*[?&]label=([^&]+).*",
            ),
            "deal_discount_pct": st.column_config.NumberColumn("Deal Discount %", format="%.1f"),
            "url": st.column_config.LinkColumn("Listing URL"),
            "is_hot_deal": st.column_config.CheckboxColumn("Hot Deal"),
            "is_upcoming": st.column_config.CheckboxColumn("Upcoming"),
            "is_new_this_week": st.column_config.CheckboxColumn("New"),
            "is_new_construction": st.column_config.CheckboxColumn("New Home"),
            "beds": st.column_config.NumberColumn("Beds", format="%.1f"),
            "baths": st.column_config.NumberColumn("Baths", format="%.1f"),
            "sqft": st.column_config.NumberColumn("SqFt", format="%.0f"),
        },
    )

    csv_data = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Filtered CSV",
        data=csv_data,
        file_name="nj_weekly_property_intelligence.csv",
        mime="text/csv",
    )


def debug_block(raw_listings: list[dict]) -> None:
    with st.expander("Raw JSON (debug)"):
        st.code(json.dumps(raw_listings[:20], indent=2), language="json")


def main() -> None:
    init_state()
    hydrate_env_from_streamlit_secrets()
    header_block()

    property_token = query_property_token()
    property_payload = query_property_payload()
    if property_token:
        render_property_detail_page(
            property_token=property_token,
            report_df=st.session_state["report_df"],
            payload_listing=property_payload,
        )
        return

    area_map = load_area_map()
    listing_filters = sidebar(st.session_state["report_df"])
    include_zillow = bool(os.getenv("ZILLOW_API_KEY", "").strip())

    st.caption(
        "Data defaults: all NJ areas, NewHomeSource enabled, and Zillow enabled when `ZILLOW_API_KEY` is set."
    )
    refresh_now = st.button("Refresh Listings", type="primary")

    if refresh_now:
        with st.spinner("Collecting listings and calculating weekly signals..."):
            run_monitor(
                selected_areas=list(area_map.keys()),
                area_map=area_map,
                include_newhomesource=True,
                include_zillow=include_zillow,
                max_results=DEFAULT_MAX_RESULTS,
                timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
                hot_deal_threshold=DEFAULT_HOT_DEAL_THRESHOLD,
            )
        st.rerun()
    elif st.session_state["report_df"] is None and not st.session_state.get("autoload_done"):
        with st.spinner("Loading top hot deals for all NJ areas..."):
            autoload_hot_deals(area_map)
        st.rerun()

    warning_block(st.session_state["warnings"])
    report_df = st.session_state["report_df"]
    data_source_diagnostics_block(
        report_df=report_df,
        warnings=st.session_state["warnings"],
        include_zillow=include_zillow,
    )
    if report_df is None:
        st.info("Run the monitor from the sidebar to generate this week's NJ report.")
        return

    filtered_df = apply_listing_filters(report_df, listing_filters)

    metrics_block(st.session_state["summary_row"], st.session_state["snapshot_path"])
    top_hot_deals_block(filtered_df)
    digest_tab, listings_tab, debug_tab = st.tabs(["Weekly Digest", "Listings", "Debug"])
    with digest_tab:
        digest_block(filtered_df)
    with listings_tab:
        listings_block(filtered_df)
    with debug_tab:
        debug_block(st.session_state["raw_listings"])


if __name__ == "__main__":
    main()
