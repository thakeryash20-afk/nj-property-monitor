from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
)
NHS_BASE_URL = "https://www.newhomesource.com"
NHS_NJ_STATE_URL = f"{NHS_BASE_URL}/state/new-jersey"

DATA_ROOT = Path("data/real_estate_monitor")
SNAPSHOT_ROOT = DATA_ROOT / "snapshots"
LATEST_SNAPSHOT_POINTER = DATA_ROOT / "latest_snapshot.json"
CLOUD_SNAPSHOT_ROOT = Path("cloud_snapshots")
CLOUD_LATEST_SNAPSHOT_FILE = CLOUD_SNAPSHOT_ROOT / "latest_snapshot.json"

DEFAULT_NJ_AREAS = {
    "Atlantic-Cape May": f"{NHS_BASE_URL}/communities/nj/atlantic-cape-may-area",
    "Bergen County": f"{NHS_BASE_URL}/communities/nj/bergen-county-area",
    "Hudson County": f"{NHS_BASE_URL}/communities/nj/hudson-county-area",
    "Mercer County": f"{NHS_BASE_URL}/communities/nj/mercer-county-area",
    "Middlesex County": f"{NHS_BASE_URL}/communities/nj/middlesex-county-area",
    "Monmouth County": f"{NHS_BASE_URL}/communities/nj/monmouth-county-area",
    "Morris County": f"{NHS_BASE_URL}/communities/nj/morris-county-area",
    "Ocean County": f"{NHS_BASE_URL}/communities/nj/ocean-county-area",
    "Passaic County": f"{NHS_BASE_URL}/communities/nj/passaic-county-area",
    "Somerset County": f"{NHS_BASE_URL}/communities/nj/somerset-county-area",
}
DEFAULT_NJ_PROPERTY_TAX_RATE = 0.022
DEFAULT_NJ_INSURANCE_RATE = 0.0035
AREA_TAX_RATE_MAP = {
    "Atlantic-Cape May": 0.022,
    "Bergen County": 0.021,
    "Hudson County": 0.020,
    "Mercer County": 0.024,
    "Middlesex County": 0.023,
    "Monmouth County": 0.021,
    "Morris County": 0.022,
    "Ocean County": 0.023,
    "Passaic County": 0.024,
    "Somerset County": 0.021,
}
AREA_CRIME_RATING_PROXY = {
    "Atlantic-Cape May": 5.5,
    "Bergen County": 7.5,
    "Hudson County": 5.0,
    "Mercer County": 6.0,
    "Middlesex County": 6.5,
    "Monmouth County": 7.2,
    "Morris County": 8.0,
    "Ocean County": 6.3,
    "Passaic County": 5.6,
    "Somerset County": 8.2,
}
COUNTY_CRIME_RATING_PROXY = {
    "atlantic county": 5.6,
    "bergen county": 7.5,
    "burlington county": 6.8,
    "camden county": 4.9,
    "cape may county": 5.4,
    "cumberland county": 4.8,
    "essex county": 4.7,
    "gloucester county": 6.7,
    "hudson county": 5.0,
    "hunterdon county": 8.6,
    "mercer county": 6.0,
    "middlesex county": 6.5,
    "monmouth county": 7.2,
    "morris county": 8.0,
    "ocean county": 6.3,
    "passaic county": 5.6,
    "salem county": 5.2,
    "somerset county": 8.2,
    "sussex county": 7.7,
    "union county": 5.7,
    "warren county": 7.4,
}
OFFER_KEYWORDS = (
    "offer",
    "incentive",
    "promotion",
    "builder credit",
    "closing cost",
    "rate buy down",
    "rate buydown",
    "special financing",
    "price reduced",
    "limited time",
    "seller credit",
)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def to_absolute_url(raw_url: str) -> str:
    if not raw_url:
        return ""
    if raw_url.startswith("http://") or raw_url.startswith("https://"):
        return raw_url
    if raw_url.startswith("//"):
        return f"https:{raw_url}"
    if raw_url.startswith("/"):
        return f"{NHS_BASE_URL}{raw_url}"
    return f"{NHS_BASE_URL}/{raw_url.lstrip('/')}"


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def parse_stat_value(card_stat_text: str) -> float | None:
    return safe_float(card_stat_text)


def price_midpoint(price_low: float | None, price_high: float | None) -> float | None:
    if price_low is not None and price_high is not None:
        return round((price_low + price_high) / 2.0, 2)
    if price_low is not None:
        return price_low
    return price_high


def format_price(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"${value:,.0f}"


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return normalize_space(str(value))


def first_valid_float(*values: Any) -> float | None:
    for value in values:
        parsed = safe_float(value)
        if parsed is not None:
            return parsed
    return None


def inferred_tax_rate(area: str | None) -> float:
    area_key = safe_text(area)
    if area_key and area_key in AREA_TAX_RATE_MAP:
        return AREA_TAX_RATE_MAP[area_key]
    return DEFAULT_NJ_PROPERTY_TAX_RATE


def inferred_crime_rating(area: str | None, county: str | None = None) -> float:
    area_key = safe_text(area)
    if area_key and area_key in AREA_CRIME_RATING_PROXY:
        return AREA_CRIME_RATING_PROXY[area_key]
    county_key = safe_text(county).lower()
    if county_key and county_key in COUNTY_CRIME_RATING_PROXY:
        return COUNTY_CRIME_RATING_PROXY[county_key]
    return 6.0


def extract_offer_snippets(text: str, max_items: int = 5) -> list[str]:
    cleaned = safe_text(text)
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+|\s+\|\s+|\s+•\s+", cleaned)
    snippets: list[str] = []
    for sentence in sentences:
        candidate = safe_text(sentence)
        if len(candidate) < 16:
            continue
        lower = candidate.lower()
        if any(keyword in lower for keyword in OFFER_KEYWORDS):
            if candidate not in snippets:
                snippets.append(candidate)
        if len(snippets) >= max_items:
            break
    return snippets


def parse_hoa_monthly(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    text = safe_text(raw_value)
    value = safe_float(text)
    if value is None:
        return None
    if any(token in text.lower() for token in ("year", "annual", "yr")):
        return round(value / 12.0, 2)
    return value


def parse_hoa_from_text(text: str) -> float | None:
    cleaned = safe_text(text)
    if not cleaned:
        return None

    hoa_patterns = [
        r"hoa(?: fee)?[^$]{0,40}\$([\d,]+(?:\.\d+)?)\s*(monthly|month|per month|yearly|annual|year)?",
        r"\$([\d,]+(?:\.\d+)?)\s*(monthly|month|per month|yearly|annual|year)\s*hoa",
    ]
    for pattern in hoa_patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if not match:
            continue
        amount = safe_float(match.group(1))
        cadence = safe_text(match.group(2)).lower()
        if amount is None:
            continue
        if cadence in {"yearly", "annual", "year"}:
            return round(amount / 12.0, 2)
        return amount
    return None


def new_http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return session


def _nhs_headers(referer: str | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
    }
    if referer:
        headers["Referer"] = referer
    return headers


def fetch_nhs_nj_areas(timeout_seconds: int = 25) -> dict[str, str]:
    session = new_http_session()
    response = session.get(NHS_NJ_STATE_URL, timeout=timeout_seconds, headers=_nhs_headers(referer=NHS_BASE_URL))
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    discovered: dict[str, str] = {}
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "")
        if "/communities/nj/" not in href or not href.endswith("-area"):
            continue
        label = normalize_space(anchor.get_text(" ", strip=True))
        if not label:
            continue
        discovered[label] = to_absolute_url(href)

    if not discovered:
        return DEFAULT_NJ_AREAS.copy()

    return dict(sorted(discovered.items(), key=lambda item: item[0]))


def extract_nhs_listing_id(url: str) -> str:
    if not url:
        return ""
    path_bits = [bit for bit in urlparse(url).path.split("/") if bit]
    if path_bits:
        candidate = path_bits[-1]
        if candidate.isdigit():
            return candidate
    return url


def parse_nhs_card(card: Any, area_name: str, fetched_at: str) -> dict[str, Any]:
    link_tag = (
        card.select_one("a[href*='/community/']")
        or card.select_one("a[href*='newhomesource.com/community/']")
        or card.select_one("a[href]")
    )
    url = to_absolute_url(link_tag.get("href", "")) if link_tag else ""
    listing_id = card.get("data-community-id") or card.get("data-id") or extract_nhs_listing_id(url)
    name = card.get("data-community-name") or card.get("data-name")
    if not name and link_tag:
        name = normalize_space(link_tag.get_text(" ", strip=True))

    status = normalize_space(card.get("data-marketing-status-label", ""))
    is_coming_soon = str(card.get("data-is-coming-soon", "")).lower() == "true"
    card_text = normalize_space(card.get_text(" ", strip=True)).lower()
    if "pricing coming soon" in card_text:
        status = status or "Pricing coming soon"
        is_coming_soon = True

    if not status:
        status = "Coming soon" if is_coming_soon else "Active"

    beds = None
    baths = None
    sqft = None
    for stat in card.select("li.nhs-n1-c-card__stats-stat"):
        raw_text = normalize_space(stat.get_text(" ", strip=True))
        raw_text_lower = raw_text.lower()
        parsed = parse_stat_value(raw_text)
        if parsed is None:
            continue
        if "bed" in raw_text_lower:
            beds = parsed
        elif "bath" in raw_text_lower:
            baths = parsed
        elif "sqft" in raw_text_lower:
            sqft = parsed

    price_low = safe_float(card.get("data-price-low"))
    price_high = safe_float(card.get("data-price-high"))
    if price_low is None and price_high is None:
        price_tag = card.select_one(".nhs-n1-c-card__price")
        if price_tag is not None:
            fallback_price = safe_float(price_tag.get_text(" ", strip=True))
            price_low = fallback_price
            price_high = fallback_price

    listing = {
        "source": "NewHomeSource",
        "listing_id": f"NHS-{listing_id}",
        "name": name or "Unnamed Listing",
        "area": area_name,
        "city": normalize_space(card.get("data-city", "")),
        "state": "NJ",
        "zip_code": normalize_space(card.get("data-zip", "")),
        "status": status,
        "is_upcoming": is_coming_soon or ("coming soon" in status.lower()),
        "is_new_construction": True,
        "builder": normalize_space(card.get("data-brand-name", "")),
        "price_low": price_low,
        "price_high": price_high,
        "beds": beds,
        "baths": baths,
        "sqft": sqft,
        "url": url,
        "last_seen_at": fetched_at,
    }
    return listing


def fetch_newhomesource_area_listings(
    area_name: str,
    area_url: str,
    max_results: int = 60,
    timeout_seconds: int = 25,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    session = session or new_http_session()
    if not getattr(session, "_nhs_warmed", False):
        try:
            session.get(NHS_BASE_URL, timeout=timeout_seconds, headers=_nhs_headers())
            session.get(NHS_NJ_STATE_URL, timeout=timeout_seconds, headers=_nhs_headers(referer=NHS_BASE_URL))
        except requests.RequestException:
            pass
        setattr(session, "_nhs_warmed", True)

    response: requests.Response | None = None
    last_exc: requests.RequestException | None = None
    for attempt in range(2):
        try:
            response = session.get(
                area_url,
                timeout=timeout_seconds,
                headers=_nhs_headers(referer=NHS_NJ_STATE_URL),
            )
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(0.6)
            continue

        if response.status_code == 403 and attempt == 0:
            try:
                session.get(NHS_NJ_STATE_URL, timeout=timeout_seconds, headers=_nhs_headers(referer=NHS_BASE_URL))
            except requests.RequestException:
                pass
            time.sleep(0.8)
            continue
        break

    if response is None:
        if last_exc is not None:
            raise last_exc
        raise requests.RequestException("NewHomeSource area request failed without response.")
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    cards = soup.select(".nhs-n1-c-card--housing")
    fetched_at = now_utc_iso()

    listings: list[dict[str, Any]] = []
    for card in cards[:max_results]:
        listing = parse_nhs_card(card, area_name=area_name, fetched_at=fetched_at)
        if listing.get("url"):
            listings.append(listing)
    return listings


def _zillow_api_settings() -> dict[str, str]:
    api_url = os.getenv(
        "ZILLOW_API_URL", "https://zillow-com1.p.rapidapi.com/propertyExtendedSearch"
    ).strip()
    provider = os.getenv("ZILLOW_PROVIDER", "").strip().lower()
    if not provider:
        provider = "hasdata" if "api.hasdata.com/scrape/zillow/listing" in api_url else "rapidapi"
    return {
        "provider": provider,
        "api_key": os.getenv("ZILLOW_API_KEY", "").strip(),
        "api_url": api_url,
        "api_host": os.getenv("ZILLOW_API_HOST", "zillow-com1.p.rapidapi.com").strip(),
        "key_header": os.getenv("ZILLOW_API_KEY_HEADER", "x-rapidapi-key").strip(),
        "host_header": os.getenv("ZILLOW_API_HOST_HEADER", "x-rapidapi-host").strip(),
        "area_param": os.getenv("ZILLOW_AREA_PARAM", "location").strip(),
        "area_format": os.getenv("ZILLOW_AREA_FORMAT", "{area}, NJ").strip(),
        "hasdata_type": os.getenv("ZILLOW_HASDATA_TYPE", "forSale").strip(),
    }


def _extract_zillow_results(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("props", "results", "listings", "properties", "data", "items", "homeResults"):
        candidate = payload.get(key)
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]

    cat1 = payload.get("cat1")
    if isinstance(cat1, dict):
        search_results = cat1.get("searchResults")
        if isinstance(search_results, dict):
            for key in ("listResults", "mapResults"):
                candidate = search_results.get(key)
                if isinstance(candidate, list):
                    return [item for item in candidate if isinstance(item, dict)]

    return []


def _zillow_request_headers(settings: dict[str, str]) -> dict[str, str]:
    provider = settings.get("provider", "").lower()
    if provider == "hasdata":
        return {
            "x-api-key": settings["api_key"],
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }

    headers = {
        settings["key_header"]: settings["api_key"],
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    if settings["api_host"]:
        headers[settings["host_header"]] = settings["api_host"]
    return headers


def _zillow_request_params(settings: dict[str, str], area_name: str) -> dict[str, Any]:
    provider = settings.get("provider", "").lower()
    if provider == "hasdata":
        params: dict[str, Any] = {
            "keyword": settings.get("area_format", "{area}, NJ").format(area=area_name),
            "type": settings.get("hasdata_type", "forSale") or "forSale",
        }
    else:
        area_param = settings.get("area_param", "location") or "location"
        area_format = settings.get("area_format", "{area}, NJ") or "{area}, NJ"
        params = {area_param: area_format.format(area=area_name)}

    extra_params_raw = os.getenv("ZILLOW_EXTRA_QUERY_JSON", "").strip()
    if extra_params_raw:
        try:
            extra_params = json.loads(extra_params_raw)
            if isinstance(extra_params, dict):
                params.update(extra_params)
        except json.JSONDecodeError:
            return {"__error__": "Zillow skipped: ZILLOW_EXTRA_QUERY_JSON is not valid JSON."}
    return params


def _zillow_status(record: dict[str, Any]) -> tuple[str, bool, bool]:
    listing_sub_type = record.get("listing_sub_type")
    if not isinstance(listing_sub_type, dict):
        listing_sub_type = {}
    sub_type_flags = {str(k).lower(): bool(v) for k, v in listing_sub_type.items()}

    home_status = str(
        record.get("homeStatus")
        or record.get("homeStatusForHDP")
        or record.get("statusType")
        or record.get("statusText")
        or ""
    ).strip()
    status_lower = home_status.lower()

    is_coming_soon = "coming" in status_lower and "soon" in status_lower
    if not is_coming_soon:
        is_coming_soon = any("coming" in key and value for key, value in sub_type_flags.items())

    is_new_construction = bool(record.get("isNewConstruction") or record.get("newConstruction"))
    if not is_new_construction:
        is_new_construction = any("new" in key and value for key, value in sub_type_flags.items())

    if not home_status:
        if is_coming_soon:
            home_status = "Coming soon"
        elif is_new_construction:
            home_status = "New construction"
        else:
            home_status = "For sale"

    return home_status, is_coming_soon, is_new_construction


def _zillow_address_bits(record: dict[str, Any]) -> tuple[str, str, str]:
    address = record.get("address")
    if not isinstance(address, dict):
        address = {}
    street = (
        address.get("streetAddress")
        or record.get("streetAddress")
        or record.get("addressStreet")
        or ""
    )
    city = address.get("city") or record.get("city") or ""
    zip_code = address.get("zipcode") or record.get("zipcode") or ""
    name = normalize_space(
        str(record.get("address") if isinstance(record.get("address"), str) else "")
    )
    if not name:
        name = normalize_space(" ".join(str(part) for part in (street, city, "NJ", zip_code) if part))
    return name or "Zillow listing", normalize_space(city), normalize_space(zip_code)


def _zillow_url(record: dict[str, Any]) -> str:
    path = (
        record.get("detailUrl")
        or record.get("hdpUrl")
        or record.get("url")
        or record.get("detailURL")
        or ""
    )
    if not path:
        return ""
    if isinstance(path, str) and path.startswith("/"):
        return f"https://www.zillow.com{path}"
    return str(path)


def _zillow_listing_id(record: dict[str, Any], url: str) -> str:
    zpid = record.get("zpid") or record.get("id") or record.get("listing_id")
    if zpid:
        return f"Z-{zpid}"
    if url:
        parsed = urlparse(url)
        parts = [bit for bit in parsed.path.split("/") if bit]
        if parts:
            maybe_id = parts[-1]
            if maybe_id.isdigit():
                return f"Z-{maybe_id}"
        qs = parse_qs(parsed.query)
        if "zpid" in qs and qs["zpid"]:
            return f"Z-{qs['zpid'][0]}"
    return f"Z-{url}"


def fetch_zillow_area_listings(
    area_name: str,
    max_results: int = 60,
    timeout_seconds: int = 25,
) -> tuple[list[dict[str, Any]], str | None]:
    settings = _zillow_api_settings()
    if not settings["api_key"]:
        return [], "Zillow skipped: set ZILLOW_API_KEY to enable Zillow ingestion."

    headers = _zillow_request_headers(settings)
    params = _zillow_request_params(settings, area_name=area_name)
    if "__error__" in params:
        return [], str(params["__error__"])

    session = new_http_session()
    try:
        response = session.get(settings["api_url"], headers=headers, params=params, timeout=timeout_seconds)
    except requests.RequestException as exc:
        return [], f"Zillow request failed for '{area_name}': {exc}"
    if response.status_code >= 400:
        # Retry once for transient provider/API gateway errors.
        try:
            time.sleep(0.8)
            response = session.get(settings["api_url"], headers=headers, params=params, timeout=timeout_seconds)
        except requests.RequestException:
            pass
    if response.status_code >= 400:
        error_message = ""
        try:
            payload = response.json()
            if isinstance(payload, dict):
                error_message = safe_text(payload.get("message") or payload.get("error"))
        except Exception:
            error_message = safe_text(response.text[:160])
        suffix = f": {error_message}" if error_message else ""
        return [], f"Zillow request failed ({response.status_code}) for '{area_name}'{suffix}"

    try:
        payload = response.json()
    except json.JSONDecodeError:
        return [], "Zillow request returned non-JSON content."

    raw_records = _extract_zillow_results(payload)
    fetched_at = now_utc_iso()

    listings: list[dict[str, Any]] = []
    for record in raw_records[:max_results]:
        if not isinstance(record, dict):
            continue
        status, is_coming_soon, is_new_construction = _zillow_status(record)
        name, city, zip_code = _zillow_address_bits(record)
        url = _zillow_url(record)

        price = safe_float(record.get("price") or record.get("unformattedPrice"))
        price_low = safe_float(record.get("minPrice")) or price
        price_high = safe_float(record.get("maxPrice")) or price
        if price_low is None and isinstance(record.get("priceRange"), dict):
            price_low = safe_float(record["priceRange"].get("min"))
            price_high = safe_float(record["priceRange"].get("max"))

        listing = {
            "source": "Zillow",
            "listing_id": _zillow_listing_id(record, url),
            "name": name,
            "area": area_name,
            "city": city,
            "state": "NJ",
            "zip_code": zip_code,
            "status": status,
            "is_upcoming": is_coming_soon,
            "is_new_construction": is_new_construction,
            "builder": normalize_space(str(record.get("builderName") or record.get("brokerName") or "")),
            "price_low": price_low,
            "price_high": price_high,
            "beds": safe_float(record.get("bedrooms") or record.get("beds")),
            "baths": safe_float(record.get("bathrooms") or record.get("baths")),
            "sqft": safe_float(record.get("livingArea") or record.get("area") or record.get("sqft")),
            "url": url,
            "last_seen_at": fetched_at,
        }
        listings.append(listing)

    return listings, None


def listing_key(listing: dict[str, Any]) -> str:
    return f"{listing.get('source', '')}::{listing.get('listing_id') or listing.get('url', '')}"


def deduplicate_listings(listings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for listing in listings:
        deduped[listing_key(listing)] = listing
    return list(deduped.values())


def collect_listings(
    area_map: dict[str, str],
    selected_areas: list[str],
    include_newhomesource: bool = True,
    include_zillow: bool = True,
    max_results_per_area: int = 60,
    timeout_seconds: int = 25,
) -> tuple[list[dict[str, Any]], list[str]]:
    all_listings: list[dict[str, Any]] = []
    warnings: list[str] = []
    nhs_session = new_http_session() if include_newhomesource else None
    nhs_blocked = False

    zillow_api_enabled = include_zillow and bool(_zillow_api_settings().get("api_key"))
    if include_zillow and not zillow_api_enabled:
        warnings.append("Zillow skipped: set ZILLOW_API_KEY to enable Zillow ingestion.")

    for area_name in selected_areas:
        if include_newhomesource and not nhs_blocked:
            area_url = area_map.get(area_name)
            if not area_url:
                warnings.append(f"NewHomeSource skipped '{area_name}': missing area URL.")
            else:
                try:
                    nhs_rows = fetch_newhomesource_area_listings(
                        area_name=area_name,
                        area_url=area_url,
                        max_results=max_results_per_area,
                        timeout_seconds=timeout_seconds,
                        session=nhs_session,
                    )
                    all_listings.extend(nhs_rows)
                except Exception as exc:
                    exc_text = str(exc).lower()
                    if "403" in exc_text and "newhomesource" in exc_text:
                        warnings.append("NewHomeSource blocked requests from this environment; skipping NewHomeSource for this run.")
                        nhs_blocked = True
                        continue
                    warnings.append(f"NewHomeSource failed for '{area_name}': {exc}")

        if zillow_api_enabled:
            try:
                zillow_rows, zillow_warning = fetch_zillow_area_listings(
                    area_name=area_name,
                    max_results=max_results_per_area,
                    timeout_seconds=timeout_seconds,
                )
                all_listings.extend(zillow_rows)
                if zillow_warning:
                    warnings.append(zillow_warning)
            except Exception as exc:
                warnings.append(f"Zillow failed for '{area_name}': {exc}")

    return deduplicate_listings(all_listings), warnings


def ensure_storage() -> None:
    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)


def _is_snapshot_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("listings"), list)


def _load_cloud_snapshot_file() -> dict[str, Any] | None:
    if not CLOUD_LATEST_SNAPSHOT_FILE.exists():
        return None
    try:
        payload = json.loads(CLOUD_LATEST_SNAPSHOT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    if _is_snapshot_payload(payload):
        return payload
    return None


def _load_remote_snapshot() -> dict[str, Any] | None:
    snapshot_url = os.getenv("CLOUD_SNAPSHOT_URL", "").strip()
    if not snapshot_url:
        return None
    session = new_http_session()
    try:
        response = session.get(snapshot_url, timeout=20)
    except requests.RequestException:
        return None
    if response.status_code >= 400:
        return None
    try:
        payload = response.json()
    except Exception:
        return None
    if _is_snapshot_payload(payload):
        return payload
    return None


def save_snapshot(
    listings: list[dict[str, Any]],
    selected_areas: list[str],
    enabled_sources: list[str],
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    ensure_storage()
    created_at = now_utc_iso()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot_path = SNAPSHOT_ROOT / f"snapshot_{ts}.json"
    payload: dict[str, Any] = {
        "created_at": created_at,
        "selected_areas": selected_areas,
        "enabled_sources": enabled_sources,
        "listing_count": len(listings),
        "listings": listings,
    }
    if extra_metadata:
        payload["meta"] = extra_metadata

    snapshot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pointer = {"created_at": created_at, "snapshot_file": snapshot_path.name}
    LATEST_SNAPSHOT_POINTER.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    return snapshot_path


def load_latest_snapshot() -> dict[str, Any] | None:
    ensure_storage()
    if LATEST_SNAPSHOT_POINTER.exists():
        try:
            pointer = json.loads(LATEST_SNAPSHOT_POINTER.read_text(encoding="utf-8"))
            target_name = pointer.get("snapshot_file")
            if target_name:
                target_path = SNAPSHOT_ROOT / str(target_name)
                if target_path.exists():
                    return json.loads(target_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    snapshots = sorted(SNAPSHOT_ROOT.glob("snapshot_*.json"))
    if snapshots:
        latest_path = snapshots[-1]
        return json.loads(latest_path.read_text(encoding="utf-8"))

    # Streamlit Cloud fallback: use repo-published snapshot when live collection is blocked.
    cloud_snapshot = _load_cloud_snapshot_file()
    if cloud_snapshot is not None:
        return cloud_snapshot
    return _load_remote_snapshot()


def publish_cloud_snapshot(snapshot: dict[str, Any], output_path: Path = CLOUD_LATEST_SNAPSHOT_FILE) -> Path:
    if not _is_snapshot_payload(snapshot):
        raise ValueError("Invalid snapshot payload: expected a dict with a listings array.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return output_path


def annotate_changes(
    current_listings: list[dict[str, Any]],
    previous_snapshot: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    previous_records = []
    if previous_snapshot and isinstance(previous_snapshot, dict):
        previous_records = previous_snapshot.get("listings") or []
    previous_keys = {listing_key(item) for item in previous_records if isinstance(item, dict)}
    current_keys = {listing_key(item) for item in current_listings}

    annotated = []
    new_count = 0
    for listing in current_listings:
        item = dict(listing)
        is_new = listing_key(listing) not in previous_keys
        item["is_new_this_week"] = is_new
        if is_new:
            new_count += 1
        annotated.append(item)

    dropped_count = len(previous_keys - current_keys)
    return annotated, {"new_count": new_count, "dropped_count": dropped_count}


def apply_deal_scoring(
    listings: list[dict[str, Any]],
    hot_deal_threshold_pct: float = 15.0,
) -> pd.DataFrame:
    if not listings:
        return pd.DataFrame()

    df = pd.DataFrame(listings).copy()
    for column in ("price_low", "price_high", "beds", "baths", "sqft"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["reference_price"] = df.apply(
        lambda row: price_midpoint(row.get("price_low"), row.get("price_high")),
        axis=1,
    )
    area_median = df.groupby("area")["reference_price"].median()
    global_median = df["reference_price"].median()
    df["area_reference_price"] = df["area"].map(area_median)
    df["area_reference_price"] = df["area_reference_price"].fillna(global_median)

    df["deal_discount_pct"] = (
        (df["area_reference_price"] - df["reference_price"]) / df["area_reference_price"] * 100.0
    )
    df.loc[df["reference_price"].isna() | (df["area_reference_price"] <= 0), "deal_discount_pct"] = pd.NA
    df["is_hot_deal"] = df["deal_discount_pct"].fillna(-999) >= hot_deal_threshold_pct

    df["price_low_text"] = df["price_low"].apply(format_price)
    df["price_high_text"] = df["price_high"].apply(format_price)
    df["reference_price_text"] = df["reference_price"].apply(format_price)
    return df


def area_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    summary = (
        df.groupby("area", dropna=False)
        .agg(
            listings=("listing_id", "count"),
            new_this_week=("is_new_this_week", "sum"),
            upcoming=("is_upcoming", "sum"),
            hot_deals=("is_hot_deal", "sum"),
            median_price=("reference_price", "median"),
        )
        .reset_index()
        .sort_values(["new_this_week", "hot_deals", "listings"], ascending=False)
    )
    summary["median_price_text"] = summary["median_price"].apply(format_price)
    return summary


def _extract_school_ratings(schools_payload: dict[str, Any]) -> tuple[list[float], list[dict[str, Any]]]:
    school_rows: list[dict[str, Any]] = []
    ratings: list[float] = []
    for key in ("assignedSchools", "nearbySchools"):
        records = schools_payload.get(key)
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            rating = safe_float(record.get("rating"))
            if rating is not None:
                ratings.append(rating)
            school_rows.append(
                {
                    "name": safe_text(record.get("name")),
                    "rating": rating,
                    "grades": safe_text(record.get("grades")),
                    "distance_miles": safe_float(record.get("distance")),
                    "type": safe_text(record.get("type")),
                    "link": safe_text(record.get("link")),
                }
            )
    # Deduplicate by school name for a cleaner table.
    deduped: dict[str, dict[str, Any]] = {}
    for row in school_rows:
        deduped[row["name"] or str(len(deduped))] = row
    return ratings, list(deduped.values())[:10]


def _extract_number_from_text(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text or "", flags=re.IGNORECASE)
    if not match:
        return None
    return safe_float(match.group(1))


def _newhomesource_variant_rows(
    soup: BeautifulSoup,
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    container_specs = [
        ("plans-container", "Floor Plan"),
        ("specs-container", "Quick Move-in"),
    ]

    for container_id, variant_type in container_specs:
        container = soup.find(id=container_id)
        if container is None:
            continue

        cards = container.find_all(
            "div",
            class_=lambda classes: classes and "swiper-slide" in classes,
        )
        for card in cards:
            if not hasattr(card, "get_text"):
                continue

            card_text = normalize_space(card.get_text(" ", strip=True))
            if not card_text:
                continue

            title_link = card.find("a", href=True)
            name = safe_text(title_link.get_text(" ", strip=True) if title_link else "")
            variant_url = to_absolute_url(safe_text(title_link.get("href")) if title_link else "")
            dedupe_key = (variant_type, name, variant_url)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            price = first_valid_float(
                card.get("data-price"),
                (card.find(attrs={"data-card-element": "price"}) or {}).get_text(" ", strip=True)
                if card.find(attrs={"data-card-element": "price"})
                else None,
            )
            beds = first_valid_float(
                card.get("data-bedrooms"),
                _extract_number_from_text(card_text, r"(\d[\d,]*(?:\.\d+)?)\s*Beds?\b"),
            )
            baths = first_valid_float(
                card.get("data-bathrooms"),
                _extract_number_from_text(card_text, r"(\d[\d,]*(?:\.\d+)?)\s*Baths?\b"),
            )
            sqft = _extract_number_from_text(card_text, r"(\d[\d,]*(?:\.\d+)?)\s*(?:Sq\s*Ft|SqFt|Square Feet)\b")

            style_match = re.search(
                r"(?:Quick move-?in|Floor plan)\s+([A-Za-z][A-Za-z\s\-]+)$",
                card_text,
                flags=re.IGNORECASE,
            )
            home_style = safe_text(style_match.group(1)) if style_match else ""

            variants.append(
                {
                    "variant_type": variant_type,
                    "name": name,
                    "price": price,
                    "beds": beds,
                    "baths": baths,
                    "sqft": sqft,
                    "home_style": home_style,
                    "url": variant_url,
                }
            )

    variants.sort(
        key=lambda row: (
            str(row.get("variant_type") or ""),
            float(row.get("price")) if row.get("price") is not None else float("inf"),
            str(row.get("name") or ""),
        )
    )
    return variants


def fetch_zillow_property_details(
    listing_url: str,
    area: str | None = None,
    timeout_seconds: int = 35,
) -> tuple[dict[str, Any], str | None]:
    if not listing_url:
        return {}, "Property details skipped: missing Zillow listing URL."

    settings = _zillow_api_settings()
    if settings.get("provider", "").lower() != "hasdata":
        return {}, "Property details are currently supported for HasData Zillow provider."

    property_api_url = os.getenv(
        "ZILLOW_PROPERTY_API_URL", "https://api.hasdata.com/scrape/zillow/property"
    ).strip()
    headers = _zillow_request_headers(settings)
    params = {"url": listing_url}

    session = new_http_session()
    try:
        response = session.get(property_api_url, headers=headers, params=params, timeout=timeout_seconds)
    except requests.RequestException as exc:
        return {}, f"Zillow property detail request failed: {exc}"
    if response.status_code >= 400:
        # Retry once for transient provider/API gateway errors.
        try:
            time.sleep(0.8)
            response = session.get(property_api_url, headers=headers, params=params, timeout=timeout_seconds)
        except requests.RequestException:
            pass
    if response.status_code >= 400:
        error_message = ""
        try:
            payload = response.json()
            if isinstance(payload, dict):
                error_message = safe_text(payload.get("message") or payload.get("error"))
        except Exception:
            error_message = safe_text(response.text[:160])
        suffix = f": {error_message}" if error_message else ""
        return {}, f"Zillow property detail request failed ({response.status_code}){suffix}"

    try:
        payload = response.json()
    except json.JSONDecodeError:
        return {}, "Zillow property detail response was not valid JSON."

    if not isinstance(payload, dict):
        return {}, "Zillow property detail payload format was unexpected."
    property_obj = payload.get("property")
    if not isinstance(property_obj, dict):
        return {}, "Zillow property detail payload did not include a property object."

    reso = property_obj.get("resoData")
    if not isinstance(reso, dict):
        reso = {}
    schools = property_obj.get("schools")
    if not isinstance(schools, dict):
        schools = {}
    address = property_obj.get("address")
    if not isinstance(address, dict):
        address = {}

    school_ratings, school_rows = _extract_school_ratings(schools)
    school_rating_avg = round(mean(school_ratings), 2) if school_ratings else None

    tax_annual_amount = first_valid_float(
        reso.get("taxAnnualAmount"),
        property_obj.get("taxAnnualAmount"),
    )
    if tax_annual_amount is None:
        tax_history = property_obj.get("taxHistory")
        if isinstance(tax_history, list) and tax_history:
            tax_annual_amount = safe_float(tax_history[0].get("taxPaid"))

    hoa_fee_monthly = parse_hoa_monthly(
        reso.get("hoaFee")
        or reso.get("hoaFeeTotal")
        or reso.get("associationFee")
        or first_valid_float(
            ((reso.get("associations") or [{}])[0]).get("feeFrequency")
            if isinstance(reso.get("associations"), list) and reso.get("associations")
            else None
        )
    )
    if hoa_fee_monthly is None:
        fees_and_dues = reso.get("feesAndDues")
        if isinstance(fees_and_dues, list):
            for fee in fees_and_dues:
                if not isinstance(fee, dict):
                    continue
                hoa_fee_monthly = parse_hoa_monthly(fee.get("fee"))
                if hoa_fee_monthly is not None:
                    break

    description = safe_text(property_obj.get("description"))
    offer_text_bits: list[str] = [description]
    at_a_glance = reso.get("atAGlanceFacts")
    if isinstance(at_a_glance, list):
        for fact in at_a_glance:
            if not isinstance(fact, dict):
                continue
            label = safe_text(fact.get("factLabel"))
            value = safe_text(fact.get("factValue"))
            if label or value:
                offer_text_bits.append(f"{label}: {value}")
    builder_offers = extract_offer_snippets(" ".join(offer_text_bits))

    county_name = safe_text(address.get("county"))
    details = {
        "description": description,
        "home_type": safe_text(property_obj.get("homeType") or reso.get("propertySubType")),
        "year_built": first_valid_float(property_obj.get("yearBuilt"), reso.get("yearBuilt")),
        "price": first_valid_float(property_obj.get("price")),
        "price_per_sqft": first_valid_float(reso.get("pricePerSquareFoot")),
        "lot_size": safe_text(reso.get("lotSize")),
        "tax_annual_amount": tax_annual_amount,
        "tax_assessed_value": first_valid_float(reso.get("taxAssessedValue")),
        "hoa_fee_monthly": hoa_fee_monthly,
        "school_rating_avg": school_rating_avg,
        "school_rows": school_rows,
        "crime_rating_proxy": inferred_crime_rating(area=area, county=county_name),
        "county": county_name,
        "days_on_market": first_valid_float(property_obj.get("daysOnZillow")),
        "builder_offers": builder_offers,
        "home_variants": [],
    }
    return details, None


def fetch_newhomesource_property_details(
    listing_url: str,
    area: str | None = None,
    timeout_seconds: int = 35,
) -> tuple[dict[str, Any], str | None]:
    if not listing_url:
        return {}, "Property details skipped: missing NewHomeSource listing URL."

    session = new_http_session()
    try:
        session.get(NHS_NJ_STATE_URL, timeout=timeout_seconds, headers={"Referer": f"{NHS_BASE_URL}/"})
        response = session.get(
            listing_url,
            timeout=timeout_seconds,
            headers={"Referer": NHS_NJ_STATE_URL},
        )
    except requests.RequestException as exc:
        return {}, f"NewHomeSource property detail request failed: {exc}"
    if response.status_code >= 400:
        return {}, f"NewHomeSource property detail request failed ({response.status_code})."

    soup = BeautifulSoup(response.text, "html.parser")
    page_text = normalize_space(soup.get_text(" ", strip=True))
    description_meta = soup.find("meta", attrs={"name": "description"})
    description = safe_text(description_meta.get("content") if description_meta else "")
    if not description:
        description = page_text[:600]

    title = safe_text(soup.title.get_text(" ", strip=True) if soup.title else "")
    hoa_fee_monthly = parse_hoa_from_text(page_text)
    builder_offers = extract_offer_snippets(f"{description} {page_text}")
    home_variants = _newhomesource_variant_rows(soup)
    details = {
        "description": description,
        "home_type": "New Construction",
        "year_built": None,
        "price": None,
        "price_per_sqft": None,
        "lot_size": None,
        "tax_annual_amount": None,
        "tax_assessed_value": None,
        "hoa_fee_monthly": hoa_fee_monthly,
        "school_rating_avg": None,
        "school_rows": [],
        "crime_rating_proxy": inferred_crime_rating(area=area, county=None),
        "county": safe_text(area),
        "days_on_market": None,
        "builder_offers": builder_offers,
        "home_variants": home_variants,
        "page_title": title,
    }
    return details, None


def fetch_property_profile(
    listing: dict[str, Any],
    timeout_seconds: int = 35,
) -> tuple[dict[str, Any], list[str]]:
    source = safe_text(listing.get("source"))
    area = safe_text(listing.get("area"))
    listing_url = safe_text(listing.get("url"))
    warnings: list[str] = []

    details: dict[str, Any] = {}
    warning: str | None = None
    if source == "Zillow":
        details, warning = fetch_zillow_property_details(
            listing_url=listing_url,
            area=area,
            timeout_seconds=timeout_seconds,
        )
    elif source == "NewHomeSource":
        details, warning = fetch_newhomesource_property_details(
            listing_url=listing_url,
            area=area,
            timeout_seconds=timeout_seconds,
        )
    else:
        warning = f"No detail connector configured for source '{source}'."

    if warning:
        warnings.append(warning)

    ref_price = price_midpoint(
        safe_float(listing.get("price_low")),
        safe_float(listing.get("price_high")),
    )
    tax_annual_amount = first_valid_float(details.get("tax_annual_amount"))
    if tax_annual_amount is None and ref_price is not None:
        tax_annual_amount = round(ref_price * inferred_tax_rate(area), 2)

    insurance_annual_estimate = None
    if ref_price is not None:
        insurance_annual_estimate = round(ref_price * DEFAULT_NJ_INSURANCE_RATE, 2)

    profile = {
        **details,
        "tax_annual_amount": tax_annual_amount,
        "insurance_annual_estimate": insurance_annual_estimate,
        "hoa_fee_monthly": first_valid_float(details.get("hoa_fee_monthly")),
        "school_rating_avg": first_valid_float(details.get("school_rating_avg")),
        "crime_rating_proxy": first_valid_float(details.get("crime_rating_proxy"))
        or inferred_crime_rating(area=area, county=safe_text(details.get("county"))),
        "builder_offers": details.get("builder_offers") if isinstance(details.get("builder_offers"), list) else [],
        "home_variants": details.get("home_variants") if isinstance(details.get("home_variants"), list) else [],
    }
    return profile, warnings
