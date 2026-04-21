"""Create a quick data quality profile for ecommerce datasets.

This script is meant to be run before notebook-based EDA. It finds a CSV or
Parquet dataset, summarizes schema, missing values, duplicates, and likely
date/category/numeric fields, then writes a Markdown report to outputs/reports.
"""

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
    """Return the only available dataset, or raise with a helpful message."""
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
            "Add a file or pass --input data/raw/your_file.csv."
        )

    if len(candidates) > 1:
        options = "\n".join(f"  - {path.relative_to(PROJECT_ROOT)}" for path in candidates)
        raise ValueError(f"Multiple datasets found. Choose one with --input:\n{options}")

    return candidates[0]


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV or Parquet file into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact per-column profile."""
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "non_null": df.notna().sum().values,
            "missing": df.isna().sum().values,
            "missing_pct": (df.isna().mean() * 100).round(2).values,
            "unique_values": df.nunique(dropna=True).values,
        }
    )
    summary["unique_pct"] = (summary["unique_values"] / max(len(df), 1) * 100).round(2)
    return summary.sort_values(["missing_pct", "unique_values"], ascending=[False, False])


def likely_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Infer column roles from names and dtypes for quick EDA direction."""
    columns = list(df.columns)
    lowered = {col: col.lower() for col in columns}

    date_keywords = ("date", "time", "created", "updated", "timestamp")
    amount_keywords = ("amount", "price", "revenue", "sales", "total", "cost", "profit")
    id_keywords = ("id", "key", "number", "code")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        col
        for col in columns
        if df[col].dtype == "object" and df[col].nunique(dropna=True) <= max(50, len(df) * 0.1)
    ]

    return {
        "date_like": [col for col in columns if any(token in lowered[col] for token in date_keywords)],
        "amount_like": [col for col in columns if any(token in lowered[col] for token in amount_keywords)],
        "id_like": [col for col in columns if any(token in lowered[col] for token in id_keywords)],
        "numeric": numeric_columns,
        "categorical": categorical_columns,
    }


def maybe_date_columns(df: pd.DataFrame, threshold: float = 0.8) -> list[str]:
    """Find object columns that mostly parse as datetimes."""
    candidates: list[str] = []
    for column in df.select_dtypes(include=["object"]).columns:
        sample = df[column].dropna().head(200)
        if sample.empty:
            continue

        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= threshold:
            candidates.append(column)

    return candidates


def display_path(path: Path) -> str:
    """Show project-relative paths when possible."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a Markdown table without extra dependencies."""
    if df.empty:
        return "_No rows to display_"

    headers = [str(column) for column in df.columns]
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return compact descriptive stats for numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()

    described = numeric_df.describe().T.reset_index().rename(columns={"index": "column"})
    selected = described[["column", "mean", "std", "min", "25%", "50%", "75%", "max"]].copy()
    selected.insert(1, "negative_values", [int((df[column] < 0).sum()) for column in selected["column"]])
    return selected.round(2).sort_values("std", ascending=False)


def top_categories_summary(df: pd.DataFrame, max_categories: int = 5) -> dict[str, pd.DataFrame]:
    """Return the most common values for low-cardinality categorical columns."""
    summaries: dict[str, pd.DataFrame] = {}
    for column in df.select_dtypes(include=["object", "category"]).columns:
        unique_count = df[column].nunique(dropna=True)
        if 1 < unique_count <= 20:
            counts = (
                df[column]
                .fillna("<<missing>>")
                .value_counts()
                .head(max_categories)
                .rename_axis("value")
                .reset_index(name="count")
            )
            summaries[column] = counts
    return summaries


def render_report(dataset_path: Path, df: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Render the profile as Markdown."""
    roles = likely_columns(df)
    parsed_date_candidates = maybe_date_columns(df)
    duplicate_rows = int(df.duplicated().sum())
    total_cells = int(df.shape[0] * df.shape[1])
    missing_cells = int(df.isna().sum().sum())
    missing_pct = round((missing_cells / total_cells * 100) if total_cells else 0, 2)

    top_missing = summary.head(10)[["column", "dtype", "missing", "missing_pct", "unique_values"]]
    numeric_stats = numeric_summary(df).head(10)
    top_categories = top_categories_summary(df)

    lines = [
        "# Ecommerce Data Profile",
        "",
        f"Dataset: `{display_path(dataset_path)}`",
        "",
        "## Shape",
        "",
        f"- Rows: {len(df):,}",
        f"- Columns: {df.shape[1]:,}",
        f"- Duplicate rows: {duplicate_rows:,}",
        f"- Missing cells: {missing_cells:,} ({missing_pct}%)",
        "",
        "## Likely Column Roles",
        "",
    ]

    for role, columns in roles.items():
        rendered = ", ".join(f"`{col}`" for col in columns) if columns else "_None detected_"
        lines.append(f"- {role}: {rendered}")

    rendered_dates = ", ".join(f"`{col}`" for col in parsed_date_candidates) if parsed_date_candidates else "_None detected_"
    lines.append(f"- parseable_dates: {rendered_dates}")

    lines.extend(
        [
            "",
            "## Columns With Most Missing Values",
            "",
            dataframe_to_markdown(top_missing),
            "",
            "## Numeric Snapshot",
            "",
            dataframe_to_markdown(numeric_stats),
            "",
            "## Top Category Values",
            "",
        ]
    )

    if top_categories:
        for column, category_df in top_categories.items():
            lines.extend(
                [
                    f"### `{column}`",
                    "",
                    dataframe_to_markdown(category_df),
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "_No low-cardinality categorical columns detected_",
                "",
            ]
        )

    lines.extend(
        [
            "## Suggested Next EDA Questions",
            "",
            "- What columns identify orders, customers, products, and dates?",
            "- Are revenue or price fields clean, non-negative, and consistently typed?",
            "- Which categorical fields explain sales patterns best?",
            "- Are duplicate rows expected, or do they indicate ingestion issues?",
            "- Which date column should be used for trend analysis?",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a quick ecommerce data profile.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional dataset path. If omitted, data/processed and data/raw are searched.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "data_profile.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "column_summary.csv",
        help="CSV path for the per-column summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.input.resolve() if args.input else find_dataset(DEFAULT_SEARCH_DIRS)

    df = load_dataset(dataset_path)
    summary = column_summary(df)
    report = render_report(dataset_path, df, summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    summary.to_csv(args.summary_csv, index=False)

    print(f"Profile written to {display_path(args.output)}")
    print(f"Column summary written to {display_path(args.summary_csv)}")


if __name__ == "__main__":
    main()
