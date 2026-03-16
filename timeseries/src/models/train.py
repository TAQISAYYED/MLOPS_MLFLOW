import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Input
from sklearn.preprocessing import MinMaxScaler
import mlflow
import os

# ── MLflow Setup ──────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("BTCUSD_Forecasting")

# ✅ NO autolog — causes WinError 5 on this machine
# ✅ NO log_artifact — causes WinError 5 on this machine
# ✅ NO register_model — causes WinError 5 on this machine
# We log params + metrics manually, and save model to local disk only

ARTIFACT_DIR = "C:/Users/Admin/mlflow_prac/timeseries/mlartifacts"


def prepare_sequences(data, window_size=60):
    scaler   = MinMaxScaler()
    features = ['Close', 'MA7', 'MA21', 'Daily_Return', 'Volume']
    scaled   = scaler.fit_transform(data[features])

    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i - window_size: i])
        y.append(scaled[i, 0])

    return np.array(X), np.array(y), scaler


def build_model(input_shape, model_type='LSTM', units=50, dropout=0.2):
    model = Sequential()
    model.add(Input(shape=input_shape))

    if model_type == 'LSTM':
        model.add(LSTM(units=units, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(units=units, return_sequences=False))
        model.add(Dropout(dropout))
    else:  # GRU
        model.add(GRU(units=units, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(GRU(units=units, return_sequences=False))
        model.add(Dropout(dropout))

    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(model_type='LSTM', window_size=60, epochs=5, batch_size=64):

    # ── Load data ─────────────────────────────────────────
    data_path = "data/processed/btcusd_processed.csv"
    if not os.path.exists(data_path):
        print(f"❌ {data_path} not found. Run ingestion.py first.")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df = df.tail(500)

    X, y, scaler = prepare_sequences(df, window_size=window_size)

    split   = int(0.8 * len(X))
    X_train = X[:split];  X_test = X[split:]
    y_train = y[:split];  y_test = y[split:]

    print(f"\n🚀 Training {model_type} model...")
    print(f"   Train samples : {len(X_train)}")
    print(f"   Test samples  : {len(X_test)}")

    # ── MLflow run ────────────────────────────────────────
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Log parameters
        mlflow.log_param("model_type",    model_type)
        mlflow.log_param("window_size",   window_size)
        mlflow.log_param("epochs",        epochs)
        mlflow.log_param("batch_size",    batch_size)
        mlflow.log_param("units",         50)
        mlflow.log_param("dropout",       0.2)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples",  len(X_test))

        # Build and train
        model = build_model(
            (X_train.shape[1], X_train.shape[2]),
            model_type=model_type
        )

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )

        # Log per-epoch metrics
        for i, (loss, val_loss) in enumerate(
            zip(history.history["loss"], history.history["val_loss"])
        ):
            mlflow.log_metric("train_loss", float(loss),     step=i)
            mlflow.log_metric("val_loss",   float(val_loss), step=i)

        # Evaluate
        test_loss = float(model.evaluate(X_test, y_test, verbose=0))
        test_rmse = float(np.sqrt(test_loss))

        mlflow.log_metric("test_mse",  test_loss)
        mlflow.log_metric("test_rmse", test_rmse)

        # ── Save model to local disk only ─────────────────
        # ✅ No mlflow.log_artifact — that causes WinError 5
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        save_path = os.path.join(
            ARTIFACT_DIR, f"{model_type}_model_{run_id[:8]}.keras"
        )
        model.save(save_path)

        # Log the local path as a param so it's visible in MLflow UI
        mlflow.log_param("local_model_path", save_path)

        print(f"\n✅ Training complete!")
        print(f"   Model type : {model_type}")
        print(f"   Test MSE   : {test_loss:.6f}")
        print(f"   Test RMSE  : {test_rmse:.6f}")
        print(f"   Model saved: {save_path}")
        print(f"   Run ID     : {run_id}")
        print(f"   View run   : http://127.0.0.1:5000/#/experiments/1/runs/{run_id}")

    return model, history, scaler


if __name__ == "__main__":
    try:
        # Train LSTM
        train_model(model_type='LSTM', epochs=5)

        # Uncomment to also train GRU
        # train_model(model_type='GRU', epochs=5)

    except Exception as e:
        import traceback
        print(f"\n❌ Training failed: {e}")
        traceback.print_exc()
        print("\nMake sure MLflow server is running:")
        print("  python -m mlflow server --host 127.0.0.1 --port 5000 "
              "--backend-store-uri sqlite:///C:/Users/Admin/mlflow_prac/timeseries/mlflow.db "
              "--artifacts-destination C:/Users/Admin/mlflow_prac/timeseries/mlartifacts "
              "--serve-artifacts")
