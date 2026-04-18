from __future__ import annotations

import argparse
from pathlib import Path

from nj_property_monitor_core import (
    CLOUD_LATEST_SNAPSHOT_FILE,
    load_latest_snapshot,
    publish_cloud_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish latest collected snapshot to repo path for Streamlit Cloud fallback."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(CLOUD_LATEST_SNAPSHOT_FILE),
        help="Output JSON file path to commit to GitHub.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = load_latest_snapshot()
    if snapshot is None:
        raise SystemExit("No snapshot found. Run weekly_nj_digest.py or the monitor first.")

    output_path = Path(args.output)
    result_path = publish_cloud_snapshot(snapshot=snapshot, output_path=output_path)
    listing_count = len(snapshot.get("listings") or []) if isinstance(snapshot, dict) else 0
    print(f"Published cloud snapshot: {result_path}")
    print(f"Listings: {listing_count}")


if __name__ == "__main__":
    main()
