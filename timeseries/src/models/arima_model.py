import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.statsmodels
import os
import warnings

warnings.filterwarnings("ignore")

# MLflow server connection
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Create experiment
mlflow.set_experiment("BTCUSD_ARIMA_Forecasting")


def train_arima_model(p=5, d=1, q=0):

    data_path = "data/processed/btcusd_processed.csv"

    if not os.path.exists(data_path):
        print("Dataset not found")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    series = df["Close"]

    split = int(len(series) * 0.8)

    train = series[:split]
    test = series[split:]

    with mlflow.start_run(run_name="ARIMA_Model"):

        print(f"Training ARIMA({p},{d},{q})")

        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()

        predictions = model_fit.forecast(steps=len(test))

        mse = mean_squared_error(test, predictions)
        rmse = np.sqrt(mse)

        # log parameters
        mlflow.log_param("model_type", "ARIMA")
        mlflow.log_param("p", p)
        mlflow.log_param("d", d)
        mlflow.log_param("q", q)

        # log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)

        # log model
        mlflow.statsmodels.log_model(
            model_fit,
            artifact_path="model",
            registered_model_name="BTCUSD_ARIMA_Model"
        )

        print("Training Completed")
        print("MSE:", mse)

        return model_fit, mse


if __name__ == "__main__":
    train_arima_model()