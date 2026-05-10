import json

import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Stock Forecasting SaaS Prototype", layout="wide")

st.title("Stock Forecasting SaaS Prototype")
st.write("Minimal SaaS prototype for stock price prediction and Vietnam portfolio output.")


def get_api_json(path):
    response = requests.get(API_URL + path, timeout=10)
    response.raise_for_status()
    return response.json()


def post_api_json(path, payload):
    response = requests.post(API_URL + path, json=payload, timeout=30)
    return response


try:
    health = get_api_json("/health")
    models_response = get_api_json("/models")
except Exception as exc:
    st.error(f"FastAPI is not reachable at {API_URL}. Start FastAPI first.")
    st.code(str(exc))
    st.stop()


st.success(f"FastAPI connected: {health.get('status', 'ok')}")

# Your /models endpoint returns a dictionary: task1, task2, task4.
models = {
    key: value
    for key, value in models_response.items()
    if key in ["task1", "task2", "task4"]
}

if not models:
    st.warning("No API models found. Check the /models response.")
    st.json(models_response)
    st.stop()


task_label = st.selectbox(
    "Choose deployed model",
    [
        "Task 1.1 - Nasdaq next-day prediction",
        "Task 2.1 - Vietnam next-day prediction",
        "Task 4 - Vietnam portfolio recommendation",
    ],
)


if task_label.startswith("Task 1.1"):
    task_key = "task1"
    endpoint = "/predict/task1"
elif task_label.startswith("Task 2.1"):
    task_key = "task2"
    endpoint = "/predict/task2"
else:
    task_key = "task4"
    endpoint = "/portfolio/task4"


st.subheader("Model Metadata")
st.json(models.get(task_key, {}))


if task_key in ["task1", "task2"]:
    metadata = models[task_key].get("metadata", {})

    feature_names = (
        metadata.get("features")
        or metadata.get("feature_columns")
        or metadata.get("model_features")
        or metadata.get("columns")
        or []
    )

    seq_len = (
        metadata.get("seq_len")
        or metadata.get("sequence_length")
        or 30
    )

    feature_count = len(feature_names) if feature_names else 13

    sample_row = [0.5] * feature_count
    sample_payload = {
        "features": [sample_row for _ in range(int(seq_len))],
        "scaled": True,
    }

    st.subheader("Prediction Input")
    st.write(
        f"This test input uses {seq_len} time steps and {feature_count} features per row."
    )

    user_json = st.text_area(
        "Input JSON",
        value=json.dumps(sample_payload, indent=2),
        height=350,
    )

    if st.button("Predict"):
        try:
            payload = json.loads(user_json)
        except json.JSONDecodeError as exc:
            st.error("Invalid JSON input.")
            st.code(str(exc))
            st.stop()

        response = post_api_json(endpoint, payload)

        st.subheader("API Response")

        if response.status_code == 200:
            st.success("Prediction completed.")
            st.json(response.json())
        else:
            st.error(f"Prediction failed with status code {response.status_code}.")
            st.json(response.json())


else:
    st.subheader("Task 4 Portfolio Output")

    if st.button("Load Portfolio Recommendation"):
        response = requests.get(API_URL + endpoint, timeout=30)

        if response.status_code == 200:
            result = response.json()
            st.success("Portfolio loaded.")
            st.json(result)
        else:
            st.error(f"Request failed with status code {response.status_code}.")
            st.json(response.json())