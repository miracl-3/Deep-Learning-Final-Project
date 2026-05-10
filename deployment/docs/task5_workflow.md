# Task 5 AI Engineering Workflow

## Goal

The goal is to deploy the trained stock forecasting and portfolio models in a form
that other users or systems can access. This deployment scope includes Task 1.1,
Task 2.1, and Task 4. Task 3 is intentionally excluded.

## 5.1 API Deployment

The trained GRU models are exported as Keras `.keras` files with preprocessing scalers
and metadata. A FastAPI service loads the artifacts and exposes REST endpoints:

- `/health`: check if the API is running
- `/models`: list available artifacts
- `/predict/task1`: Task 1.1 Nasdaq next-day price prediction
- `/predict/task2`: Task 2.1 Vietnam next-day price prediction
- `/portfolio/task4`: Task 4 Vietnam portfolio recommendation

## 5.2 SaaS Interface

A Streamlit app provides a simple web-based interface. Users can select a deployed model,
enter recent historical stock features, call the API, and view the prediction or portfolio output.
The web app also shows input previews, simple trend charts, portfolio tables, portfolio weight
charts, and CSV downloads that can be imported into Power BI.

## 5.3 Automation Workflow

A realistic production workflow would use:

1. Airbyte to ingest new stock-market data from external sources.
2. PostgreSQL to store raw prices, processed features, predictions, and portfolio outputs.
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

## Data Movement

```text
Airbyte
  -> PostgreSQL raw_stock_prices
  -> dbt/Python model_features
  -> Airflow prediction task
  -> FastAPI /predict/task1, /predict/task2, /portfolio/task4
  -> PostgreSQL model_predictions and portfolio_recommendations
  -> Streamlit dashboard and Power BI CSV exports
```

## PostgreSQL Storage Design

The file `deployment/docs/postgres_schema.sql` defines a minimal PostgreSQL schema with:

- `stock_ml.raw_stock_prices`
- `stock_ml.model_features`
- `stock_ml.model_predictions`
- `stock_ml.portfolio_recommendations`

This schema is included as a deployment blueprint. The current local prototype can run
without PostgreSQL, while the workflow design shows where PostgreSQL would fit in an
industry-style deployment.

The notebook itself is not deployed directly. Instead, the notebook exports reusable artifacts
that can be loaded by API and SaaS applications.
