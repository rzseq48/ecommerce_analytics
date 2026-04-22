"""Generate reusable ecommerce analytics features from a cleaned dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH_DIRS = (
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "data" / "raw",
)
SUPPORTED_SUFFIXES = {".csv", ".parquet"}


def find_dataset(search_dirs: Iterable[Path]) -> Path:
    candidates: list[Path] = []
    for search_dir in search_dirs:
        if search_dir.exists():
            candidates.extend(
                path
                for path in sorted(search_dir.iterdir())
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
            )

    if not candidates:
        locations = ", ".join(path.as_posix() for path in search_dirs)
        raise FileNotFoundError(
            f"No CSV or Parquet dataset found in {locations}. "
            "Add a file or pass --input data/processed/cleaned_data.csv."
        )

    if len(candidates) > 1:
        options = "\n".join(f"  - {path.relative_to(PROJECT_ROOT)}" for path in candidates)
        raise ValueError(f"Multiple datasets found. Choose one with --input:\n{options}")

    return candidates[0]


def load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported output file type: {path.suffix}")


def choose_column(df: pd.DataFrame, keywords: tuple[str, ...]) -> str | None:
    lowered_map = {column: column.lower() for column in df.columns}
    for column, lowered in lowered_map.items():
        if any(keyword in lowered for keyword in keywords):
            return column
    return None


def add_time_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    created: list[str] = []
    date_column = choose_column(df, ("order_date", "purchase_date", "created", "date", "timestamp", "time"))
    if date_column is None:
        return df, created

    parsed = pd.to_datetime(df[date_column], errors="coerce")
    if parsed.notna().sum() == 0:
        return df, created

    df[date_column] = parsed
    df["order_year"] = parsed.dt.year
    df["order_month"] = parsed.dt.month
    df["order_day"] = parsed.dt.day
    df["order_week"] = parsed.dt.isocalendar().week.astype("Int64")
    df["order_weekday"] = parsed.dt.day_name()
    df["order_month_name"] = parsed.dt.month_name()
    created.extend(
        [
            "order_year",
            "order_month",
            "order_day",
            "order_week",
            "order_weekday",
            "order_month_name",
        ]
    )
    return df, created


def add_revenue_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    created: list[str] = []
    price_column = choose_column(df, ("price", "amount", "sales", "revenue", "total"))
    quantity_column = choose_column(df, ("quantity", "qty", "units"))
    discount_column = choose_column(df, ("discount",))
    cost_column = choose_column(df, ("cost",))

    if price_column:
        df["unit_value"] = pd.to_numeric(df[price_column], errors="coerce")
        created.append("unit_value")

    if price_column and quantity_column:
        qty = pd.to_numeric(df[quantity_column], errors="coerce").fillna(0)
        unit_value = pd.to_numeric(df[price_column], errors="coerce").fillna(0)
        df["gross_revenue_estimate"] = unit_value * qty
        created.append("gross_revenue_estimate")

    if "gross_revenue_estimate" in df.columns and discount_column:
        discount = pd.to_numeric(df[discount_column], errors="coerce").fillna(0)
        df["net_revenue_estimate"] = df["gross_revenue_estimate"] - discount
        created.append("net_revenue_estimate")

    if "net_revenue_estimate" in df.columns and cost_column:
        cost = pd.to_numeric(df[cost_column], errors="coerce").fillna(0)
        df["estimated_profit"] = df["net_revenue_estimate"] - cost
        created.append("estimated_profit")

    return df, created


def add_customer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    created: list[str] = []
    customer_column = choose_column(df, ("customer_id", "customer", "user_id", "buyer"))
    order_column = choose_column(df, ("order_id", "invoice", "transaction_id"))

    if customer_column is None:
        return df, created

    customer_orders = df.groupby(customer_column).cumcount() + 1
    df["customer_order_sequence"] = customer_orders
    df["is_repeat_customer"] = customer_orders > 1
    created.extend(["customer_order_sequence", "is_repeat_customer"])

    if order_column:
        order_counts = df.groupby(customer_column)[order_column].transform("nunique")
        df["customer_total_orders"] = order_counts
        created.append("customer_total_orders")

    return df, created


def add_average_order_value(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    created: list[str] = []
    order_column = choose_column(df, ("order_id", "invoice", "transaction_id"))
    revenue_column = choose_column(df, ("net_revenue_estimate", "gross_revenue_estimate", "revenue", "sales", "total"))

    if order_column is None or revenue_column is None:
        return df, created

    revenue = pd.to_numeric(df[revenue_column], errors="coerce").fillna(0)
    order_totals = revenue.groupby(df[order_column]).transform("sum")
    df["order_total_value"] = order_totals
    df["order_avg_value"] = order_totals
    created.extend(["order_total_value", "order_avg_value"])
    return df, created


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    feature_map: dict[str, list[str]] = {}

    df, created = add_time_features(df)
    feature_map["time_features"] = created

    df, created = add_revenue_features(df)
    feature_map["revenue_features"] = created

    df, created = add_customer_features(df)
    feature_map["customer_features"] = created

    df, created = add_average_order_value(df)
    feature_map["order_features"] = created

    return df, feature_map


def render_summary(input_path: Path, output_path: Path, feature_map: dict[str, list[str]]) -> str:
    lines = [
        "# Feature Engineering Summary",
        "",
        f"- Input: `{input_path.relative_to(PROJECT_ROOT)}`",
        f"- Output: `{output_path.relative_to(PROJECT_ROOT)}`",
        "",
    ]

    for section, columns in feature_map.items():
        rendered = ", ".join(f"`{column}`" for column in columns) if columns else "_No features created_"
        lines.extend([f"## {section.replace('_', ' ').title()}", "", f"- {rendered}", ""])

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create derived ecommerce analytics features.")
    parser.add_argument("--input", type=Path, help="Optional input dataset path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "featured_data.csv",
        help="Engineered dataset output path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "feature_summary.md",
        help="Markdown feature report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve() if args.input else find_dataset(DEFAULT_SEARCH_DIRS)
    df = load_dataset(input_path)
    featured_df, feature_map = engineer_features(df)

    save_dataset(featured_df, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_summary(input_path, args.output, feature_map), encoding="utf-8")

    print(f"Featured dataset written to {args.output.relative_to(PROJECT_ROOT)}")
    print(f"Feature report written to {args.report.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
