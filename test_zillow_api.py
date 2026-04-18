from __future__ import annotations

import argparse
import json
from pprint import pprint

from nj_property_monitor_core import fetch_zillow_area_listings


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Zillow API config for this project.")
    parser.add_argument("--area", default="Monmouth County", help="NJ area name, e.g. 'Bergen County'.")
    parser.add_argument("--max-results", type=int, default=5, help="Max listings to print.")
    args = parser.parse_args()

    listings, warning = fetch_zillow_area_listings(
        area_name=args.area,
        max_results=args.max_results,
        timeout_seconds=25,
    )

    if warning:
        print(f"Warning: {warning}")

    if not listings:
        print("No listings returned.")
        return

    print(f"Returned {len(listings)} listings for {args.area}:")
    for idx, listing in enumerate(listings[: args.max_results], start=1):
        preview = {
            "source": listing.get("source"),
            "listing_id": listing.get("listing_id"),
            "name": listing.get("name"),
            "city": listing.get("city"),
            "status": listing.get("status"),
            "price_low": listing.get("price_low"),
            "price_high": listing.get("price_high"),
            "url": listing.get("url"),
        }
        print(f"\n{idx}.")
        pprint(preview, sort_dicts=False)


if __name__ == "__main__":
    main()
