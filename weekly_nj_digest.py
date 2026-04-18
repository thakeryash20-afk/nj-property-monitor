from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from nj_property_monitor_core import (
    DEFAULT_NJ_AREAS,
    annotate_changes,
    apply_deal_scoring,
    area_summary,
    collect_listings,
    fetch_nhs_nj_areas,
    load_latest_snapshot,
    save_snapshot,
)

REPORT_ROOT = Path("data/real_estate_monitor/reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate weekly NJ new/upcoming property intelligence report."
    )
    parser.add_argument(
        "--areas",
        type=str,
        default="Monmouth County,Middlesex County,Bergen County",
        help="Comma-separated NJ areas. Example: 'Monmouth County,Bergen County'",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="newhomesource,zillow",
        help="Comma-separated sources: newhomesource,zillow",
    )
    parser.add_argument("--max-results", type=int, default=40, help="Max listings per area per source.")
    parser.add_argument("--timeout", type=int, default=25, help="HTTP timeout seconds.")
    parser.add_argument(
        "--hot-deal-threshold",
        type=float,
        default=15.0,
        help="Discount threshold percent vs area median to tag hot deals.",
    )
    return parser.parse_args()


def normalize_selection(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def markdown_report(
    selected_areas: list[str],
    source_labels: list[str],
    scored_df,
    summary_df,
    snapshot_path: Path,
    warnings: list[str],
    hot_deal_threshold: float,
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    lines.append("# NJ Weekly Property Intelligence")
    lines.append("")
    lines.append(f"- Generated: {generated_at}")
    lines.append(f"- Areas: {', '.join(selected_areas)}")
    lines.append(f"- Sources: {', '.join(source_labels)}")
    lines.append(f"- Hot deal threshold: {hot_deal_threshold:.1f}%")
    lines.append(f"- Snapshot: `{snapshot_path}`")
    lines.append("")

    total_count = int(len(scored_df))
    new_count = int(scored_df["is_new_this_week"].fillna(False).sum()) if total_count else 0
    upcoming_count = int(scored_df["is_upcoming"].fillna(False).sum()) if total_count else 0
    hot_count = int(scored_df["is_hot_deal"].fillna(False).sum()) if total_count else 0
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"- Listings: **{total_count}** | New this run: **{new_count}** | "
        f"Upcoming: **{upcoming_count}** | Hot deals: **{hot_count}**"
    )
    lines.append("")

    if not summary_df.empty:
        lines.append("## Area Summary")
        lines.append("")
        lines.append("| Area | Listings | New | Upcoming | Hot Deals | Median Price |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in summary_df.itertuples(index=False):
            lines.append(
                f"| {row.area} | {int(row.listings)} | {int(row.new_this_week)} | "
                f"{int(row.upcoming)} | {int(row.hot_deals)} | {row.median_price_text} |"
            )
        lines.append("")

    hot_df = scored_df[scored_df["is_hot_deal"] == True].copy() if total_count else scored_df
    if not hot_df.empty:
        lines.append("## Top Hot Deals")
        lines.append("")
        lines.append("| Source | Area | Name | City | Price | Discount % | Status | URL |")
        lines.append("| --- | --- | --- | --- | ---: | ---: | --- | --- |")
        hot_df = hot_df.sort_values("deal_discount_pct", ascending=False).head(15)
        for row in hot_df.itertuples(index=False):
            lines.append(
                f"| {row.source} | {row.area} | {row.name} | {row.city} | {row.reference_price_text} | "
                f"{(row.deal_discount_pct or 0):.1f} | {row.status} | [link]({row.url}) |"
            )
        lines.append("")

    upcoming_df = scored_df[scored_df["is_upcoming"] == True].copy() if total_count else scored_df
    if not upcoming_df.empty:
        lines.append("## Upcoming Properties")
        lines.append("")
        lines.append("| Source | Area | Name | City | Status | URL |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in upcoming_df.head(20).itertuples(index=False):
            lines.append(
                f"| {row.source} | {row.area} | {row.name} | {row.city} | {row.status} | [link]({row.url}) |"
            )
        lines.append("")

    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in dict.fromkeys(warnings):
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    selected_areas = normalize_selection(args.areas)
    selected_sources = {value.lower() for value in normalize_selection(args.sources)}
    include_newhomesource = "newhomesource" in selected_sources
    include_zillow = "zillow" in selected_sources

    if not selected_areas:
        raise SystemExit("No areas selected.")
    if not include_newhomesource and not include_zillow:
        raise SystemExit("No valid sources selected. Use newhomesource and/or zillow.")

    try:
        area_map = fetch_nhs_nj_areas(timeout_seconds=args.timeout)
    except Exception:
        area_map = DEFAULT_NJ_AREAS.copy()

    missing = [area for area in selected_areas if area not in area_map]
    if missing:
        raise SystemExit(f"Unknown areas: {', '.join(missing)}")

    previous_snapshot = load_latest_snapshot()
    listings, warnings = collect_listings(
        area_map=area_map,
        selected_areas=selected_areas,
        include_newhomesource=include_newhomesource,
        include_zillow=include_zillow,
        max_results_per_area=args.max_results,
        timeout_seconds=args.timeout,
    )
    annotated, delta = annotate_changes(listings, previous_snapshot)
    scored_df = apply_deal_scoring(annotated, hot_deal_threshold_pct=args.hot_deal_threshold)
    summary_df = area_summary(scored_df)
    snapshot_path = save_snapshot(
        listings=annotated,
        selected_areas=selected_areas,
        enabled_sources=[name for name, enabled in [("NewHomeSource", include_newhomesource), ("Zillow", include_zillow)] if enabled],
        extra_metadata={
            "hot_deal_threshold_pct": args.hot_deal_threshold,
            "max_results_per_area": args.max_results,
            "timeout_seconds": args.timeout,
        },
    )

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_ROOT / f"weekly_digest_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.md"
    report_content = markdown_report(
        selected_areas=selected_areas,
        source_labels=[name for name, enabled in [("NewHomeSource", include_newhomesource), ("Zillow", include_zillow)] if enabled],
        scored_df=scored_df,
        summary_df=summary_df,
        snapshot_path=snapshot_path,
        warnings=warnings,
        hot_deal_threshold=args.hot_deal_threshold,
    )
    report_path.write_text(report_content, encoding="utf-8")

    print(f"Report: {report_path}")
    print(f"Snapshot: {snapshot_path}")
    print(f"Listings: {len(scored_df)} | New this run: {delta['new_count']} | Dropped: {delta['dropped_count']}")
    if warnings:
        print("Warnings:")
        for warning in dict.fromkeys(warnings):
            print(f"- {warning}")


if __name__ == "__main__":
    main()
