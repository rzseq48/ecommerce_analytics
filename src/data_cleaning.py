"""Clean ecommerce datasets into a notebook-friendly processed file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH_DIRS = (
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "data" / "processed",
)
SUPPORTED_SUFFIXES = {".csv", ".parquet"}

DATE_KEYWORDS = ("date", "time", "created", "updated", "timestamp")
NUMERIC_KEYWORDS = (
    "price",
    "amount",
    "sales",
    "revenue",
    "cost",
    "profit",
    "discount",
    "quantity",
    "qty",
    "total",
)
TEXT_FILL_DEFAULTS = {
    "country": "unknown",
    "city": "unknown",
    "state": "unknown",
    "category": "unknown",
    "segment": "unknown",
    "channel": "unknown",
    "status": "unknown",
    "payment": "unknown",
}


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
            "Add a file or pass --input data/raw/your_file.csv."
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


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        column: (
            column.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )
        for column in df.columns
    }
    return df.rename(columns=renamed)


def coerce_datetime_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    converted: list[str] = []
    for column in df.columns:
        lowered = column.lower()
        if any(keyword in lowered for keyword in DATE_KEYWORDS):
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().sum() > 0:
                df[column] = parsed
                converted.append(column)
    return df, converted


def coerce_numeric_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    converted: list[str] = []
    for column in df.columns:
        lowered = column.lower()
        if not any(keyword in lowered for keyword in NUMERIC_KEYWORDS):
            continue

        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            converted.append(column)
            continue

        cleaned = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .replace({"nan": None, "None": None, "": None})
        )
        parsed = pd.to_numeric(cleaned, errors="coerce")
        if parsed.notna().sum() > 0:
            df[column] = parsed
            converted.append(column)
    return df, converted


def fill_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    applied: dict[str, str] = {}

    for column in df.columns:
        series = df[column]
        lowered = column.lower()

        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        if pd.api.types.is_numeric_dtype(series):
            if series.isna().any():
                df[column] = series.fillna(series.median())
                applied[column] = "filled numeric nulls with median"
            continue

        for keyword, fill_value in TEXT_FILL_DEFAULTS.items():
            if keyword in lowered and series.isna().any():
                df[column] = series.fillna(fill_value)
                applied[column] = f"filled text nulls with '{fill_value}'"
                break

    return df, applied


def drop_sparse_columns(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    missing_ratio = df.isna().mean()
    dropped = missing_ratio[missing_ratio >= threshold].index.tolist()
    if dropped:
        df = df.drop(columns=dropped)
    return df, dropped


def clean_dataset(df: pd.DataFrame, missing_threshold: float) -> tuple[pd.DataFrame, dict[str, object]]:
    original_rows = len(df)
    original_columns = list(df.columns)

    df = standardize_column_names(df)
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    df, date_columns = coerce_datetime_columns(df)
    df, numeric_columns = coerce_numeric_columns(df)
    df, fill_actions = fill_missing_values(df)
    df, dropped_columns = drop_sparse_columns(df, threshold=missing_threshold)

    summary = {
        "rows_before": original_rows,
        "rows_after": len(df),
        "columns_before": len(original_columns),
        "columns_after": len(df.columns),
        "duplicates_removed": duplicate_rows,
        "date_columns": date_columns,
        "numeric_columns": numeric_columns,
        "dropped_columns": dropped_columns,
        "fill_actions": fill_actions,
    }
    return df, summary


def render_summary(input_path: Path, output_path: Path, summary: dict[str, object]) -> str:
    fill_actions = summary["fill_actions"]
    fill_lines = (
        [f"- `{column}`: {action}" for column, action in fill_actions.items()]
        if fill_actions
        else ["- No missing-value fills were applied"]
    )

    dropped_columns = summary["dropped_columns"] or ["None"]
    date_columns = summary["date_columns"] or ["None"]
    numeric_columns = summary["numeric_columns"] or ["None"]

    lines = [
        "# Data Cleaning Summary",
        "",
        f"- Input: `{input_path.relative_to(PROJECT_ROOT)}`",
        f"- Output: `{output_path.relative_to(PROJECT_ROOT)}`",
        f"- Rows before: {summary['rows_before']:,}",
        f"- Rows after: {summary['rows_after']:,}",
        f"- Duplicates removed: {summary['duplicates_removed']:,}",
        f"- Columns before: {summary['columns_before']:,}",
        f"- Columns after: {summary['columns_after']:,}",
        "",
        "## Converted Columns",
        "",
        f"- Datetime: {', '.join(f'`{column}`' for column in date_columns)}",
        f"- Numeric: {', '.join(f'`{column}`' for column in numeric_columns)}",
        "",
        "## Dropped Sparse Columns",
        "",
        f"- {', '.join(f'`{column}`' for column in dropped_columns)}",
        "",
        "## Fill Actions",
        "",
    ]
    lines.extend(fill_lines)
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean an ecommerce dataset for downstream analysis.")
    parser.add_argument("--input", type=Path, help="Optional input dataset path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv",
        help="Cleaned dataset output path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "cleaning_summary.md",
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.95,
        help="Drop columns with missing ratio greater than or equal to this value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve() if args.input else find_dataset(DEFAULT_SEARCH_DIRS)
    df = load_dataset(input_path)
    cleaned_df, summary = clean_dataset(df, missing_threshold=args.missing_threshold)

    save_dataset(cleaned_df, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_summary(input_path, args.output, summary), encoding="utf-8")

    print(f"Cleaned dataset written to {args.output.relative_to(PROJECT_ROOT)}")
    print(f"Cleaning report written to {args.report.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
