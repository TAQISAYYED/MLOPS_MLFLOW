from django.shortcuts import render
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import pandas as pd
import os
from django.conf import settings
import numpy as np

# Connect MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")


def dashboard_overview(request):

    client = MlflowClient()

    experiment_names = [
        "BTCUSD_Linear_Regression",
        "BTCUSD_ARIMA_Forecasting"
    ]

    latest_prediction = "N/A"
    model_accuracy = "N/A"

    runs_data = []

    best_run = None
    best_mse = float("inf")

    for experiment_name in experiment_names:

        experiment = client.get_experiment_by_name(experiment_name)

        if experiment:

            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.mse ASC"],
                max_results=5
            )

            for run in runs:

                mse = run.data.metrics.get("mse")

                start_time = (
                    pd.to_datetime(run.info.start_time, unit='ms', utc=True)
                    .tz_convert('Asia/Kolkata')
                    .strftime('%Y-%m-%d %H:%M')
                )

                runs_data.append({
                    'run_id': run.info.run_id[:10],
                    'status': run.info.status,
                    'model_type': run.data.params.get('model_type', experiment_name),
                    'mse': mse,
                    'start_time': start_time
                })

                # Track best model
                if mse and mse < best_mse:
                    best_mse = mse
                    best_run = run

    # Best model accuracy calculation
    if best_run:

        rmse = np.sqrt(best_mse)

        model_accuracy = f"{(1/(1+rmse))*100:.2f}%"

    # Prediction using best model
    try:

        model_uri = f"runs:/{best_run.info.run_id}/model"

        model = mlflow.pyfunc.load_model(model_uri)

        file_path = os.path.join(
            settings.BASE_DIR,
            "..",
            "data",
            "processed",
            "btcusd_processed.csv"
        )

        df = pd.read_csv(file_path)

        features = ['Close', 'MA7', 'MA21', 'Daily_Return', 'Volume']

        latest_row = df[features].tail(1)

        prediction = model.predict(latest_row)

        latest_prediction = f"${prediction[0]:,.2f}"

    except Exception as e:

        latest_prediction = "Prediction Not Available"

    context = {

        'latest_prediction': latest_prediction,
        'model_accuracy': model_accuracy,
        'runs': runs_data

    }

    return render(request, 'dashboard/overview.html', context)  