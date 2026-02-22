# Global Maternal Mortality Inequality Dashboard

Production-oriented Python project for ingesting global maternal health indicators, building reproducible inequality datasets, fitting ecological regression models, and serving an interactive Plotly Dash dashboard.

## Project Structure

```text
.
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── pyproject.toml
├── data
│   ├── external
│   ├── interim
│   ├── processed
│   └── raw
├── logs
├── scripts
│   ├── run_dashboard.py
│   └── run_pipeline.py
├── src
│   └── maternal_mortality_dashboard
│       ├── __init__.py
│       ├── config.py
│       ├── exceptions.py
│       ├── io.py
│       ├── logging_config.py
│       ├── data_ingestion
│       │   ├── __init__.py
│       │   ├── ingest.py
│       │   └── world_bank_client.py
│       ├── data_cleaning
│       │   ├── __init__.py
│       │   ├── schema.py
│       │   ├── transform.py
│       │   └── validation.py
│       ├── modeling
│       │   ├── __init__.py
│       │   ├── ecological_regression.py
│       │   └── inequality_metrics.py
│       ├── pipeline
│       │   ├── __init__.py
│       │   └── orchestrator.py
│       └── dashboard
│           ├── __init__.py
│           ├── app.py
│           ├── callbacks.py
│           ├── data_access.py
│           ├── figures.py
│           └── layout.py
└── tests
    ├── __init__.py
    ├── test_clean_transform.py
    ├── test_ecological_regression.py
    └── test_inequality_metrics.py
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

3. Copy environment template and customize:

```bash
cp .env.example .env
```

## Run Pipeline

```bash
python scripts/run_pipeline.py
```

Pipeline writes deterministic artifacts to:
- `data/raw/world_bank_indicators_<start>_<end>.parquet`
- `data/interim/clean_panel_<start>_<end>.parquet`
- `data/processed/dashboard_country_year.parquet`
- `data/processed/dashboard_yearly_inequality.parquet`
- `data/processed/dashboard_income_group_summary.parquet`
- `data/processed/dashboard_latest_country_snapshot.parquet`
- `data/processed/ecological_regression_dataset.parquet`
- `data/processed/ecological_regression_coefficients.csv`
- `data/processed/ecological_regression_model_summary.csv`
- `data/processed/ecological_regression_model_summary.txt`
- `data/processed/ecological_regression_vif.csv`
- `data/processed/ecological_regression_residuals_vs_fitted.png`
- `data/processed/ecological_regression_residuals_qq_plot.png`
- `data/processed/pipeline_metadata.json`

## Run Dashboard

```bash
python scripts/run_dashboard.py
```

Default URL: `http://localhost:8050`

## Reproducibility Notes

- Pipeline behavior is controlled only by environment variables.
- Source extraction windows (`PIPELINE_START_YEAR`, `PIPELINE_END_YEAR`) are explicit.
- Output row ordering and artifact names are deterministic for fixed inputs.
- A metadata manifest with run settings and input signature is generated each run.
