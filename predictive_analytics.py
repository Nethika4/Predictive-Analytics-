import argparse
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("CSV must contain 'date' and 'value' columns")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).reset_index(drop=True)
    df["time_index"] = (df["date"] - df["date"].min()).dt.days
    df["lag_1"] = df["value"].shift(1)
    df["rolling_mean_3"] = df["value"].rolling(window=3, min_periods=1).mean()
    df = df.dropna().reset_index(drop=True)
    return df


def build_regression_model(df: pd.DataFrame) -> LinearRegression:
    features = ["time_index", "lag_1", "rolling_mean_3"]
    X = df[features].values
    y = df["value"].values
    model = LinearRegression()
    model.fit(X, y)
    return model


def forecast_regression(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    base = df.copy()
    model = build_regression_model(base)
    last_date = base["date"].max()
    last_time_index = base["time_index"].iloc[-1]
    predictions = []

    for step in range(1, horizon + 1):
        forecast_date = last_date + timedelta(days=step * int((base["date"].diff().median() or 30).days))
        time_index = last_time_index + step * int((base["date"].diff().median() or 30).days)
        lag_1 = base["value"].iloc[-1]
        rolling_mean_3 = base["value"].iloc[-3:].mean()
        row = [time_index, lag_1, rolling_mean_3]
        forecast_value = model.predict([row])[0]
        predictions.append({
            "date": forecast_date,
            "value": forecast_value,
            "model": "regression",
        })
        new_row = pd.DataFrame([
            {
                "date": forecast_date,
                "value": forecast_value,
                "time_index": time_index,
                "lag_1": lag_1,
                "rolling_mean_3": rolling_mean_3,
            }
        ])
        base = pd.concat([base, new_row], ignore_index=True)

    return pd.DataFrame(predictions)


def build_arima_model(series: pd.Series, order=(1, 1, 1)) -> ARIMA:
    model = ARIMA(series, order=order)
    fitted = model.fit()
    return fitted


def forecast_arima(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    series = df.set_index("date")["value"].asfreq(pd.infer_freq(df["date"]))
    if series.isnull().any():
        series = series.interpolate().ffill()
    fitted = build_arima_model(series)
    forecast = fitted.forecast(steps=horizon)
    result = forecast.reset_index()
    result.columns = ["date", "value"]
    result["model"] = "arima"
    return result


def evaluate_predictions(actual: pd.Series, predicted: pd.Series) -> dict:
    mse = mean_squared_error(actual, predicted)
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mse),
        "r2": r2_score(actual, predicted),
    }


def plot_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame, output_path: str | None = None):
    plt.figure(figsize=(10, 6))
    plt.plot(df["date"], df["value"], label="Historical", marker="o")
    plt.plot(forecast_df["date"], forecast_df["value"], label="Forecast", marker="x")
    plt.title("Historical Data and Forecast")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Predictive analytics with historical data")
    parser.add_argument("--data", required=True, help="Path to historical CSV file")
    parser.add_argument("--model", choices=["regression", "arima"], default="regression")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon")
    parser.add_argument("--output", help="Optional output image for forecast plot")
    args = parser.parse_args()

    df = load_data(args.data)
    preprocessed = preprocess_data(df)

    if args.model == "regression":
        forecast_df = forecast_regression(preprocessed, args.horizon)
    else:
        forecast_df = forecast_arima(df, args.horizon)

    combined = pd.concat([df[["date", "value"]], forecast_df[["date", "value"]]], ignore_index=True)
    print("Forecast results:\n", forecast_df)
    plot_forecast(df, forecast_df, args.output)

    if args.model == "regression" and len(preprocessed) > 0:
        features = ["time_index", "lag_1", "rolling_mean_3"]
        X = preprocessed[features].values
        y = preprocessed["value"].values
        model = build_regression_model(preprocessed)
        y_pred = model.predict(X)
        metrics = evaluate_predictions(y, y_pred)
        print("Regression model evaluation:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

    if args.model == "arima":
        series = df.set_index("date")["value"].asfreq(pd.infer_freq(df["date"]))
        series = series.interpolate().ffill()
        fitted = build_arima_model(series)
        print("ARIMA summary:\n", fitted.summary())


if __name__ == "__main__":
    main()
