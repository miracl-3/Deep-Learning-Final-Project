# Task 5 AI Engineering Workflow

## Goal

The goal is to deploy the trained stock forecasting and portfolio models in a form that
other users or systems can access.

## 5.1 API Deployment

The trained GRU models are exported as Keras `.keras` files with preprocessing scalers
and metadata. A FastAPI service loads the artifacts and exposes REST endpoints:

- `/health`: check if the API is running
- `/models`: list available artifacts
- `/predict-price`: predict next-day stock price using Task 1.1 or Task 2.1 models
- `/portfolio/{artifact_name}`: return Task 4 portfolio recommendations and backtest metrics

## 5.2 SaaS Interface

A Streamlit app provides a simple web-based interface. Users can select a model artifact,
upload recent historical stock data, call the API, and view the prediction or portfolio output.

## 5.3 Automation Workflow

A realistic production workflow would use:

1. Airbyte to ingest new stock-market data from external sources.
2. SQL or MongoDB to store raw prices, processed features, predictions, and portfolio outputs.
3. dbt or Python feature scripts to compute technical indicators such as moving averages,
   volatility, RSI, MACD, and momentum.
4. Airflow to schedule the pipeline:
   - ingest data
   - validate data
   - transform features
   - run model inference
   - store predictions
   - refresh dashboards
5. FastAPI to serve trained models.
6. Streamlit, Superset, Tableau, or PowerBI to expose predictions and portfolio recommendations.

The notebook itself is not deployed directly. Instead, the notebook exports reusable artifacts
that can be loaded by API and SaaS applications.