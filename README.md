# Predictive Analytics with Historical Data

This project demonstrates how to build predictive models using historical time-series data.
It includes data preprocessing, regression and time-series modeling, evaluation, and visualization.

## Key Features

- Clean and preprocess historical datasets
- Train regression and ARIMA forecasting models
- Evaluate model accuracy with MAE, RMSE, and R²
- Visualize actual data and forecasted trends

## Setup

1. Create a Python environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the script with the sample dataset:

```powershell
python predictive_analytics.py --data data/sample_historical.csv --model regression --horizon 12
```

## Usage

```powershell
python predictive_analytics.py --data data/sample_historical.csv --model arima --horizon 12 --output forecast.png
```

Options:
- `--data`: path to a CSV file with `date` and `value` columns
- `--model`: `regression` or `arima`
- `--horizon`: number of future periods to forecast
- `--output`: optional plot output file

## Data Format

The sample dataset includes:

- `date`: date or timestamp
- `value`: numeric observation for each date

You can replace `data/sample_historical.csv` with your own historical data.
