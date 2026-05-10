
import pandas as pd
import requests
import streamlit as st


API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Stock Forecasting SaaS",
    layout="wide",
)

st.title("Stock Forecasting SaaS Prototype")

st.write(
    "Minimal SaaS prototype for stock price prediction and Vietnam portfolio output."
)

try:
    response = requests.get(f"{API_URL}/models", timeout=5)
    models = response.json().get("models", [])
except Exception:
    models = []

if not models:
    st.warning("No API models found. Start FastAPI first.")
    st.stop()

model_names = [model["artifact_name"] for model in models]
selected_artifact = st.selectbox("Select artifact", model_names)

selected_metadata = next(
    model for model in models if model["artifact_name"] == selected_artifact
)

st.subheader("Artifact Metadata")
st.json(selected_metadata)

if selected_metadata["task_type"] == "price_forecasting":
    uploaded_file = st.file_uploader(
        "Upload recent stock history CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

        st.subheader("Input Preview")
        st.dataframe(input_df.tail(30))

        if st.button("Predict Price"):
            payload = {
                "artifact_name": selected_artifact,
                "records": input_df.to_dict(orient="records"),
            }

            result = requests.post(
                f"{API_URL}/predict-price",
                json=payload,
                timeout=30,
            )

            if result.status_code == 200:
                st.success("Prediction completed.")
                st.json(result.json())
            else:
                st.error("Prediction failed.")
                st.json(result.json())

elif selected_metadata["task_type"] == "portfolio_construction":
    if st.button("Load Portfolio Output"):
        result = requests.get(
            f"{API_URL}/portfolio/{selected_artifact}",
            timeout=30,
        )

        if result.status_code == 200:
            portfolio_data = result.json()

            for key, value in portfolio_data.items():
                if key.endswith(".csv"):
                    st.subheader(key)
                    st.dataframe(pd.DataFrame(value))
        else:
            st.error("Portfolio loading failed.")
            st.json(result.json())
