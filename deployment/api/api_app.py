
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow import keras


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

app = FastAPI(
    title="Stock Forecasting API",
    description="Minimal API for Task 1/2 price forecasting and Task 4 portfolio output.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    artifact_name: str
    records: list[dict]


def get_artifact_path(artifact_name: str) -> Path:
    artifact_path = ARTIFACT_DIR / artifact_name

    if not artifact_path.exists():
        available_artifacts = [
            path.name for path in ARTIFACT_DIR.iterdir() if path.is_dir()
        ] if ARTIFACT_DIR.exists() else []

        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Artifact not found: {artifact_name}",
                "available_artifacts": available_artifacts,
            },
        )

    return artifact_path


def load_metadata(artifact_path: Path) -> dict:
    with open(artifact_path / "metadata.json", "r", encoding="utf-8") as file:
        return json.load(file)


def prepare_price_features_for_api(data: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    date_column = metadata["date_column"]
    target_column = metadata["target_column"]
    required_price_columns = metadata["required_price_columns"]
    feature_columns = metadata["feature_columns"]

    required_columns = [date_column] + required_price_columns
    missing_columns = [column for column in required_columns if column not in data.columns]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing_columns}",
        )

    model_data = data.copy()
    model_data[date_column] = pd.to_datetime(model_data[date_column], errors="coerce")

    for column in required_price_columns:
        model_data[column] = pd.to_numeric(model_data[column], errors="coerce")

    model_data = model_data.dropna(subset=required_columns)
    model_data = model_data.sort_values(date_column).reset_index(drop=True)

    model_data["Log Return"] = np.log(
        model_data[target_column] / model_data[target_column].shift(1)
    )
    model_data["MA10"] = model_data[target_column].rolling(window=10).mean()
    model_data["MA20"] = model_data[target_column].rolling(window=20).mean()
    model_data["Volatility10"] = model_data["Log Return"].rolling(window=10).std()

    model_data = model_data.replace([np.inf, -np.inf], np.nan)
    model_data = model_data.dropna(subset=feature_columns).reset_index(drop=True)

    return model_data


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    if not ARTIFACT_DIR.exists():
        return {"models": []}

    models = []

    for artifact_path in ARTIFACT_DIR.iterdir():
        if artifact_path.is_dir() and (artifact_path / "metadata.json").exists():
            models.append(load_metadata(artifact_path))

    return {"models": models}


@app.post("/predict-price")
def predict_price(request: PredictionRequest):
    artifact_path = get_artifact_path(request.artifact_name)
    metadata = load_metadata(artifact_path)

    if metadata["task_type"] != "price_forecasting":
        raise HTTPException(
            status_code=400,
            detail="Selected artifact is not a price forecasting model.",
        )

    model = keras.models.load_model(artifact_path / "model.keras", compile=False)
    feature_scaler = joblib.load(artifact_path / "feature_scaler.pkl")
    target_scaler = joblib.load(artifact_path / "target_scaler.pkl")

    input_data = pd.DataFrame(request.records)
    model_data = prepare_price_features_for_api(input_data, metadata)

    lookback_days = int(metadata["lookback_days"])
    feature_columns = metadata["feature_columns"]

    if len(model_data) < lookback_days:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {lookback_days} usable rows after feature engineering.",
        )

    latest_window = model_data.tail(lookback_days)
    x_scaled = feature_scaler.transform(latest_window[feature_columns])
    x_input = x_scaled.reshape(1, lookback_days, len(feature_columns))

    prediction_scaled = model.predict(x_input, verbose=0).reshape(-1)
    prediction = target_scaler.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    ).reshape(-1)

    return {
        "artifact_name": request.artifact_name,
        "ticker": metadata["ticker"],
        "market": metadata["market"],
        "target_column": metadata["target_column"],
        "forecast_horizon": metadata["forecast_horizon"],
        "latest_input_date": str(latest_window[metadata["date_column"]].iloc[-1]),
        "prediction": prediction.tolist(),
    }


@app.get("/portfolio/{artifact_name}")
def get_portfolio(artifact_name: str):
    artifact_path = get_artifact_path(artifact_name)
    metadata = load_metadata(artifact_path)

    if metadata["task_type"] != "portfolio_construction":
        raise HTTPException(
            status_code=400,
            detail="Selected artifact is not a portfolio artifact.",
        )

    result = {"metadata": metadata}

    for file_name in [
        "stock_scores.csv",
        "risk_taking_portfolio.csv",
        "prudent_portfolio.csv",
        "portfolio_backtest_metrics.csv",
    ]:
        file_path = artifact_path / file_name

        if file_path.exists():
            result[file_name] = pd.read_csv(file_path).to_dict(orient="records")

    return result
