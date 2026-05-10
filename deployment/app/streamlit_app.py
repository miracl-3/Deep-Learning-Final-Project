import json

import pandas as pd
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Stock Forecasting SaaS Prototype", layout="wide")


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    .app-title {
        font-size: 2.4rem;
        font-weight: 760;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        color: #b8c0cc;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }
    .status-card {
        border: 1px solid rgba(130, 148, 170, 0.22);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        background: rgba(31, 41, 55, 0.62);
    }
    .small-muted {
        color: #aeb7c3;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">Stock Forecasting SaaS Prototype</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">FastAPI model service with a lightweight Streamlit interface for stock prediction and portfolio output.</div>',
    unsafe_allow_html=True,
)


def get_api_json(path):
    response = requests.get(API_URL + path, timeout=10)
    response.raise_for_status()
    return response.json()


def post_api_json(path, payload):
    return requests.post(API_URL + path, json=payload, timeout=30)


def model_metadata(model_entry):
    return model_entry.get("metadata", {}) if isinstance(model_entry, dict) else {}


def get_feature_names(metadata):
    return (
        metadata.get("feature_columns")
        or metadata.get("features")
        or metadata.get("model_features")
        or metadata.get("columns")
        or []
    )


def get_sequence_length(metadata):
    return (
        metadata.get("lookback_days")
        or metadata.get("seq_len")
        or metadata.get("sequence_length")
        or 30
    )


def format_market_label(metadata):
    ticker = metadata.get("ticker", "Unknown")
    market = metadata.get("market", "Unknown market")
    target = metadata.get("target_column", "target")
    return ticker, market, target


def build_sample_payload(seq_len, feature_names, feature_count):
    rows = []
    denom = max(seq_len - 1, 1)

    for idx in range(seq_len):
        trend = 0.35 + 0.30 * (idx / denom)
        row = []

        for col_idx in range(feature_count):
            name = feature_names[col_idx].lower() if col_idx < len(feature_names) else ""

            if "high" in name:
                value = trend + 0.04
            elif "low" in name:
                value = trend - 0.04
            elif "volume" in name or "liquidity" in name:
                value = 0.45 + 0.08 * ((idx % 5) / 4)
            elif "return" in name:
                value = 0.50 + 0.03 * ((idx % 6) - 2.5)
            elif "volatility" in name:
                value = 0.36 + 0.05 * ((idx % 4) / 3)
            elif "ma" in name:
                value = trend - 0.02
            else:
                value = trend

            row.append(round(float(min(max(value, 0.0), 1.0)), 4))

        rows.append(row)

    return {"features": rows, "scaled": True}


def parse_payload(user_json):
    try:
        payload = json.loads(user_json)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict) or "features" not in payload:
        return None
    return payload


def features_to_frame(payload, feature_names):
    features = payload.get("features", [])
    if not features:
        return pd.DataFrame()

    first_row = features[0]
    columns = feature_names if len(feature_names) == len(first_row) else None
    return pd.DataFrame(features, columns=columns)


def display_portfolio_frame(df, title):
    st.subheader(title)

    if df.empty:
        st.info("No rows returned for this portfolio.")
        return

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Holdings", f"{len(df):,}")
    with metric_cols[1]:
        if "Weight" in df.columns:
            st.metric("Total Weight", f"{df['Weight'].sum():.2%}")
    with metric_cols[2]:
        if "Predicted Return" in df.columns:
            st.metric("Avg Predicted Return", f"{df['Predicted Return'].mean():.2%}")

    st.dataframe(df, use_container_width=True, hide_index=True)

    if "Ticker" in df.columns and "Weight" in df.columns:
        chart_df = df.set_index("Ticker")[["Weight"]]
        st.caption("Portfolio weights")
        st.bar_chart(chart_df)

    score_cols = [
        col
        for col in ["Predicted Return", "Risk Adjusted Score", "Portfolio Ranking Score"]
        if col in df.columns
    ]
    if "Ticker" in df.columns and score_cols:
        st.caption("Return and ranking metrics")
        st.bar_chart(df.set_index("Ticker")[score_cols])

    st.download_button(
        f"Download {title} CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{title.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )


def show_workflow_section():
    st.subheader("Automation and BI Workflow")

    flow_rows = [
        ("Airbyte", "Collects updated Nasdaq and Vietnam stock data."),
        ("PostgreSQL", "Stores raw prices, transformed features, predictions, and portfolio outputs."),
        ("dbt / Python", "Builds model-ready features such as returns, moving averages, volatility, and risk scores."),
        ("Airflow", "Schedules ingestion, transformation, prediction, storage, and dashboard refresh."),
        ("FastAPI", "Serves Task 1.1, Task 2.1, and Task 4 outputs as REST endpoints."),
        ("Streamlit / Power BI", "Displays model predictions and portfolio tables for end users."),
    ]

    st.dataframe(
        pd.DataFrame(flow_rows, columns=["Tool", "Role in the pipeline"]),
        use_container_width=True,
        hide_index=True,
    )

    st.code(
        """Airbyte -> PostgreSQL raw tables -> dbt/Python feature tables
-> Airflow scheduled inference -> FastAPI model endpoints
-> PostgreSQL prediction tables -> Streamlit or Power BI dashboard""",
        language="text",
    )

    st.info(
        "Task 3 is intentionally excluded from this deployment. "
        "The deployed scope is Task 1.1, Task 2.1, and Task 4."
    )


try:
    health = get_api_json("/health")
    models_response = get_api_json("/models")
except Exception as exc:
    st.error(f"FastAPI is not reachable at {API_URL}. Start FastAPI first.")
    st.code(str(exc))
    st.stop()

models = {
    key: value
    for key, value in models_response.items()
    if key in ["task1", "task2", "task4"]
}

if not models:
    st.warning("No API models found. Check the /models response.")
    st.json(models_response)
    st.stop()

left, mid, right = st.columns(3)
with left:
    st.markdown(
        f'<div class="status-card"><b>API Status</b><br><span class="small-muted">{health.get("status", "ok")}</span></div>',
        unsafe_allow_html=True,
    )
with mid:
    st.markdown(
        f'<div class="status-card"><b>Model Service</b><br><span class="small-muted">{API_URL}</span></div>',
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        '<div class="status-card"><b>Deployment Scope</b><br><span class="small-muted">Task 1.1, Task 2.1, Task 4</span></div>',
        unsafe_allow_html=True,
    )

forecast_tab, portfolio_tab, workflow_tab = st.tabs(
    ["Price Prediction", "Portfolio Dashboard", "Automation Workflow"]
)

with forecast_tab:
    task_label = st.selectbox(
        "Choose deployed forecasting model",
        [
            "Task 1.1 - Nasdaq next-day prediction",
            "Task 2.1 - Vietnam next-day prediction",
        ],
    )

    if task_label.startswith("Task 1.1"):
        task_key = "task1"
        endpoint = "/predict/task1"
    else:
        task_key = "task2"
        endpoint = "/predict/task2"

    metadata = model_metadata(models.get(task_key, {}))
    feature_names = get_feature_names(metadata)
    seq_len = int(get_sequence_length(metadata))
    feature_count = len(feature_names) if feature_names else 13
    ticker, market, target = format_market_label(metadata)

    top_left, top_right = st.columns([2, 1])
    with top_left:
        st.subheader(f"{ticker} Next-Day Forecast")
        st.write(
            "Enter the latest stock price history and technical indicators. "
            "For this prototype, the sample input uses scaled values so the deployment can be tested quickly."
        )
    with top_right:
        st.metric("Market", market)
        st.metric("Target", target)

    st.info(
        f"The model expects {seq_len} trading days and {feature_count} features per day. "
        "The fields represent recent prices and indicators such as returns, moving averages, and volatility."
    )

    if feature_names:
        st.caption("Feature order used by the model")
        st.dataframe(
            pd.DataFrame({"Feature": feature_names}),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Model metadata", expanded=False):
        st.json(models.get(task_key, {}))

    sample_payload = build_sample_payload(seq_len, feature_names, feature_count)

    user_json = st.text_area(
        "Recent Price and Indicator Input",
        value=json.dumps(sample_payload, indent=2),
        height=280,
    )

    preview_payload = parse_payload(user_json)
    if preview_payload:
        preview_df = features_to_frame(preview_payload, feature_names)
        if not preview_df.empty:
            st.caption("Input preview")
            st.dataframe(preview_df.tail(8), use_container_width=True, hide_index=True)

            numeric_df = preview_df.select_dtypes(include="number")
            if not numeric_df.empty:
                st.caption("Recent input trend")
                st.line_chart(numeric_df.iloc[:, : min(4, len(numeric_df.columns))])

    if st.button("Predict Next-Day Price", type="primary"):
        try:
            payload = json.loads(user_json)
        except json.JSONDecodeError as exc:
            st.error("Invalid JSON input.")
            st.code(str(exc))
            st.stop()

        response = post_api_json(endpoint, payload)

        if response.status_code == 200:
            result = response.json()
            predicted_value = result.get("predicted_next_day_value")

            st.success("Prediction completed.")
            if predicted_value is not None:
                st.metric("Predicted Next-Day Price", f"{predicted_value:,.2f}")

            with st.expander("Full API response", expanded=True):
                st.json(result)
        else:
            st.error(f"Prediction failed with status code {response.status_code}.")
            st.json(response.json())

with portfolio_tab:
    st.subheader("Vietnam Portfolio Recommendation")
    st.write(
        "Task 4 returns the saved portfolio output generated from the portfolio scoring workflow. "
        "The output includes prudent, risk-taking, and stock-score tables."
    )

    with st.expander("Model metadata", expanded=False):
        st.json(models.get("task4", {}))

    if st.button("Load Portfolio Recommendation", type="primary"):
        response = requests.get(API_URL + "/portfolio/task4", timeout=30)

        if response.status_code == 200:
            result = response.json()
            st.success("Portfolio loaded.")

            prudent_df = pd.DataFrame(result.get("prudent_portfolio", []))
            risk_df = pd.DataFrame(result.get("risk_taking_portfolio", []))
            scores_df = pd.DataFrame(result.get("stock_scores", []))

            if not scores_df.empty:
                st.subheader("Stock Score Overview")
                overview_cols = st.columns(3)
                with overview_cols[0]:
                    st.metric("Scored Stocks", f"{len(scores_df):,}")
                with overview_cols[1]:
                    if "Predicted Return" in scores_df.columns:
                        st.metric("Best Predicted Return", f"{scores_df['Predicted Return'].max():.2%}")
                with overview_cols[2]:
                    if "Risk Adjusted Score" in scores_df.columns:
                        st.metric("Best Risk-Adjusted Score", f"{scores_df['Risk Adjusted Score'].max():.3f}")

                top_scores = scores_df.sort_values("Risk Adjusted Score", ascending=False).head(12)
                if "Ticker" in top_scores.columns and "Risk Adjusted Score" in top_scores.columns:
                    st.caption("Top stocks by risk-adjusted score")
                    st.bar_chart(top_scores.set_index("Ticker")[["Risk Adjusted Score"]])

                st.download_button(
                    "Download Power BI Stock Scores CSV",
                    data=scores_df.to_csv(index=False).encode("utf-8"),
                    file_name="powerbi_stock_scores.csv",
                    mime="text/csv",
                )

            display_portfolio_frame(prudent_df, "Prudent Portfolio")
            display_portfolio_frame(risk_df, "Risk-Taking Portfolio")

            with st.expander("Full API response", expanded=False):
                st.json(result)
        else:
            st.error(f"Request failed with status code {response.status_code}.")
            st.json(response.json())

with workflow_tab:
    show_workflow_section()
