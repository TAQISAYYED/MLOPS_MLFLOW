# MLFLOW_PRAC

A time series forecasting project using MLflow for experiment tracking and model management.

## Project Structure

```
MLFLOW_PRAC/
├── timeseries/
│   ├── data/               # Raw and processed datasets
│   ├── env3/               # Virtual environment
│   ├── mlartifacts/        # MLflow artifact storage
│   ├── mlruns/             # MLflow run metadata
│   ├── monitoring/         # Model monitoring utilities
│   ├── roi/                # ROI analysis
│   └── src/
│       ├── data/
│       │   └── ingestion.py
│       └── models/
│           ├── arima_model.py
│           ├── linear_regression.py
│           └── train.py
├── arima_model.pkl         # Saved ARIMA model
├── mlflow.db               # MLflow tracking database
└── requirements.txt
```

## Models

- **ARIMA** - Autoregressive Integrated Moving Average model for time series forecasting
- **Linear Regression** - Baseline regression model
- **LSTM** - Long Short-Term Memory neural network model (stored in mlartifacts)

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv env3
   source env3/bin/activate  # Linux/Mac
   env3\Scripts\activate     # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run data ingestion:
```bash
python src/data/ingestion.py
```

Train a model:
```bash
python src/models/train.py
```

Launch the MLflow UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://localhost:5000` in your browser to view experiments and runs.

## Requirements

See `requirements.txt` for the full list of dependencies.
