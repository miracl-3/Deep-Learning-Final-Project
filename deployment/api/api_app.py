from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="CS313 Stock Prediction API")


API_DIR = Path(__file__).resolve().parent
DEPLOYMENT_DIR = API_DIR.parent


def find_artifacts_dir() -> Path:
    env_dir = os.environ.get("ARTIFACTS_DIR")
    if env_dir:
        return Path(env_dir)

    candidates = [
        DEPLOYMENT_DIR / "artifacts",
        API_DIR / "artifacts",
    ]

    for candidate in candidates:
        if (candidate / "task1_nasdaq_next_day").exists():
            return candidate

    return DEPLOYMENT_DIR / "artifacts"


ARTIFACTS_DIR = find_artifacts_dir()

TASKS = {
    "task1": {
        "name": "Task 1.1 - Nasdaq next-day prediction",
        "dir": ARTIFACTS_DIR / "task1_nasdaq_next_day",
        "model": "model.keras",
        "feature_scaler": "feature_scaler.pkl",
        "target_scaler": "target_scaler.pkl",
    },
    "task2": {
        "name": "Task 2.1 - Vietnam next-day prediction",
        "dir": ARTIFACTS_DIR / "task2_vietnam_next_day",
        "model": "model.keras",
        "feature_scaler": "feature_scaler.pkl",
        "target_scaler": "target_scaler.pkl",
    },
    "task4": {
        "name": "Task 4 - Vietnam portfolio recommendation",
        "dir": ARTIFACTS_DIR / "task4_vietnam_portfolio",
        "model": "task4_return_model.keras",
        "feature_scaler": "task4_feature_scaler.pkl",
    },
}


class SequenceInput(BaseModel):
    features: list[list[float]]
    scaled: bool = False


class PortfolioInput(BaseModel):
    features: list[list[float]] | None = None
    scaled: bool = False


def require_file(path: Path) -> Path:
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Missing artifact: {path}")
    return path


@lru_cache(maxsize=None)
def load_model(task_key: str):
    task = TASKS[task_key]
    return tf.keras.models.load_model(require_file(task["dir"] / task["model"]), compile=False)


@lru_cache(maxsize=None)
def load_joblib(path_text: str):
    return joblib.load(require_file(Path(path_text)))


@lru_cache(maxsize=None)
def load_metadata(task_key: str) -> dict[str, Any]:
    path = TASKS[task_key]["dir"] / "metadata.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_feature_names(metadata: dict[str, Any]) -> list[str]:
    for key in ["features", "feature_columns", "model_features", "columns"]:
        value = metadata.get(key)
        if isinstance(value, list):
            return value
    return []


def prepare_sequence_input(task_key: str, payload: SequenceInput) -> np.ndarray:
    task = TASKS[task_key]
    metadata = load_metadata(task_key)

    raw = np.asarray(payload.features, dtype=float)

    if raw.ndim != 2:
        raise HTTPException(
            status_code=400,
            detail="features must be a 2D list: rows of time steps by feature columns.",
        )

    feature_names = get_feature_names(metadata)
    if feature_names and raw.shape[1] != len(feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(feature_names)} features per row, got {raw.shape[1]}. "
                   f"Expected features: {feature_names}",
        )

    seq_len = metadata.get("seq_len") or metadata.get("sequence_length")
    if seq_len and raw.shape[0] != int(seq_len):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {seq_len} rows/time steps, got {raw.shape[0]}.",
        )

    if payload.scaled:
        scaled = raw
    else:
        scaler = load_joblib(str(task["dir"] / task["feature_scaler"]))
        scaled = scaler.transform(raw)

    return np.expand_dims(scaled, axis=0)


def inverse_target(task_key: str, prediction_scaled: np.ndarray) -> np.ndarray:
    task = TASKS[task_key]
    target_scaler_path = task["dir"] / task["target_scaler"]

    if not target_scaler_path.exists():
        return prediction_scaled

    target_scaler = load_joblib(str(target_scaler_path))
    original_shape = prediction_scaled.shape

    return target_scaler.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    ).reshape(original_shape)


def predict_next_day(task_key: str, payload: SequenceInput) -> dict[str, Any]:
    model = load_model(task_key)
    metadata = load_metadata(task_key)
    x = prepare_sequence_input(task_key, payload)

    prediction_scaled = model.predict(x, verbose=0)
    prediction = inverse_target(task_key, prediction_scaled)

    outputs = prediction.reshape(-1).astype(float).tolist()

    return {
        "task": TASKS[task_key]["name"],
        "artifact_folder": TASKS[task_key]["dir"].name,
        "ticker": metadata.get("ticker"),
        "prediction_type": "next_day_prediction",
        "predicted_next_day_value": outputs[0],
        "all_model_outputs": outputs,
        "note": "Only the first model output is used for the next-day task.",
    }


def read_csv_if_exists(path: Path, limit: int = 30) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    df = pd.read_csv(path).head(limit)
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient="records")


@app.get("/")
def root():
    return {
        "message": "Stock Prediction API is running.",
        "docs": "/docs",
        "artifacts_dir": str(ARTIFACTS_DIR),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "artifacts_dir": str(ARTIFACTS_DIR),
        "available_tasks": list(TASKS.keys()),
    }


@app.get("/models")
def models():
    result = {}

    for key, task in TASKS.items():
        metadata = load_metadata(key)
        result[key] = {
            "name": task["name"],
            "folder": str(task["dir"]),
            "model_file": task["model"],
            "metadata": metadata,
        }

    return result


@app.post("/predict/task1")
@app.post("/predict/task1-nasdaq-next-day")
def predict_task1(payload: SequenceInput):
    return predict_next_day("task1", payload)


@app.post("/predict/task2")
@app.post("/predict/task2-vietnam-next-day")
def predict_task2(payload: SequenceInput):
    return predict_next_day("task2", payload)


@app.get("/portfolio/task4")
def get_task4_portfolios():
    task_dir = TASKS["task4"]["dir"]

    return {
        "task": TASKS["task4"]["name"],
        "prudent_portfolio": read_csv_if_exists(task_dir / "prudent_portfolio.csv"),
        "risk_taking_portfolio": read_csv_if_exists(task_dir / "risk_taking_portfolio.csv"),
        "stock_scores": read_csv_if_exists(task_dir / "stock_scores.csv"),
    }


@app.post("/predict/task4")
@app.post("/predict/task4-vietnam-portfolio")
def predict_task4(payload: PortfolioInput):
    task = TASKS["task4"]
    task_dir = task["dir"]

    response = get_task4_portfolios()

    if payload.features is None:
        response["note"] = "No custom features provided. Returning saved Task 4 portfolio outputs."
        return response

    raw = np.asarray(payload.features, dtype=float)

    if raw.ndim != 2:
        raise HTTPException(
            status_code=400,
            detail="features must be a 2D list: rows by feature columns.",
        )

    if payload.scaled:
        x = raw
    else:
        scaler = load_joblib(str(task_dir / task["feature_scaler"]))
        x = scaler.transform(raw)

    model = load_model("task4")
    prediction = model.predict(x, verbose=0)

    response["predicted_returns"] = prediction.reshape(-1).astype(float).tolist()
    response["note"] = "Task 4 model prediction plus saved portfolio recommendation files."

    return response
