# Ecommerce Analytics Project

This project is a lightweight ecommerce analytics workspace for:

- cleaning raw datasets
- creating reusable analytics features
- generating quick profiling reports
- exploring data in Jupyter notebooks

## Project Structure

```text
ecommerce-analytics/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_eda_analysis.ipynb
├── outputs/
│   ├── figures/
│   └── reports/
├── sql/
├── src/
│   ├── data_cleaning.py
│   ├── data_profile.py
│   ├── feature_engineering.py
│   └── utils.py
└── requirements.txt
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

If you already have the local virtual environment in this repo, use:

```bash
data_analyst/bin/python --version
```

## Recommended Workflow

1. Add your dataset to `data/raw/`
2. Clean it with `src/data_cleaning.py`
3. Create derived fields with `src/feature_engineering.py`
4. Generate a profile report with `src/data_profile.py`
5. Open the notebooks for deeper EDA

## Commands

Clean the dataset:

```bash
data_analyst/bin/python src/data_cleaning.py --input data/raw/your_file.csv
```

Create features:

```bash
data_analyst/bin/python src/feature_engineering.py --input data/processed/cleaned_data.csv
```

Generate a profile report:

```bash
data_analyst/bin/python src/data_profile.py --input data/processed/featured_data.csv
```

Run the exploration notebook:

```bash
data_analyst/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_exploration.ipynb
```

Run the EDA notebook:

```bash
data_analyst/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda_analysis.ipynb
```

## Outputs

The scripts write results to these locations:

- cleaned dataset: `data/processed/cleaned_data.csv`
- featured dataset: `data/processed/featured_data.csv`
- cleaning report: `outputs/reports/cleaning_summary.md`
- feature report: `outputs/reports/feature_summary.md`
- data profile report: `outputs/reports/data_profile.md`
- column summary: `outputs/reports/column_summary.csv`

## Notes

- The notebooks need at least one CSV or Parquet file in `data/raw/` or `data/processed/`
- The scripts use keyword-based heuristics, so columns like `order_date`, `customer_id`, `price`, `quantity`, and `discount` work best
- You can use CSV or Parquet for both input and output files
