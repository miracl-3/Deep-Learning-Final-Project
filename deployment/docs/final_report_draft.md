# Final Project Report Draft

## Overview

This project applies deep learning and time-series modeling to stock-market prediction
and portfolio construction. The work covers Nasdaq forecasting, Vietnam market forecasting,
Vietnam buy/sell signal classification, portfolio composition and risk management, and a
deployment prototype. The final deployment scope focuses on Task 1.1, Task 2.1, and Task 4.
Task 3 is analyzed in the notebook but is intentionally excluded from deployment to keep
the web service focused on price prediction and portfolio output.

The main implementation is contained in `230070_project_notebook.ipynb`. The deployment
package is under `deployment/`, where trained Keras models, scalers, metadata, FastAPI
routes, a Streamlit interface, and workflow documentation are stored.

## Reusable Modeling Pipeline

Before solving the individual tasks, the notebook builds reusable utilities for data
loading, feature engineering, time-series splitting, scaling, sequence creation, recurrent
model construction, evaluation, plotting, and cross-validation. This keeps the modeling
pipeline consistent across Nasdaq and Vietnam data. The project uses chronological
train-validation-test splits instead of random splits, because stock prediction is a
time-series problem and future data must not leak into training.

The recurrent forecasting models use GRU layers. GRU was selected because it is simpler
and faster than a larger LSTM architecture while still being appropriate for sequential
price data. The main feature sets include price and volume columns plus technical
features such as log return, moving averages, and volatility.

## Task 1: Nasdaq Stock Price Prediction

Task 1 predicts Nasdaq stock prices. The deployed Task 1.1 model is a multi-feature
next-day forecaster for ticker `GT`. The model uses 30 recent trading days and predicts
the next adjusted close value. The feature set includes open, high, low, close, adjusted
close, volume, log return, moving averages, and volatility.

The Task 1 experiments include next-day prediction, k-th day prediction, and k consecutive
day prediction. The final summary reports GRU models for all three subtasks. Task 1.1
achieved MAE 2.3361, RMSE 3.0346, MAPE 13.5473, sMAPE 13.1027, R2 0.8358, and directional
accuracy 42.31%. Task 1.2 achieved RMSE 3.2099 and R2 0.8166. Task 1.3 achieved RMSE
3.1904 and R2 0.8185. These results show that the recurrent model captured a substantial
part of the Nasdaq ticker's price movement, especially for the regression target, although
directional accuracy remained moderate.

## Task 2: Vietnam Stock Price Prediction

Task 2 adapts the forecasting workflow to the Vietnam market. The deployed Task 2.1 model
is a next-day GRU forecaster for ticker `VCB`. The Vietnam data has a different file layout
and market convention from the Nasdaq data, so the notebook includes separate data resolution,
cleaning, and feature setup logic. The feature set uses open, high, low, close, volume, log
return, moving averages, and volatility.

The Task 2 summary includes next-day, k-th day, and k consecutive day forecasting. Task 2.1
achieved MAE 4328.9255, RMSE 4979.1999, MAPE 5.3452, sMAPE 5.5286, R2 0.2291, and directional
accuracy 44.20%. Task 2.2 achieved RMSE 5474.1722 and R2 0.0690. Task 2.3 achieved RMSE
7960.2206 and R2 -0.9911. The next-day model was the strongest Vietnam forecasting result,
so Task 2.1 was selected for deployment rather than the longer-horizon variants.

## Task 3: Vietnam Trading Signal Identification

Task 3 turns Vietnam market data into classification labels for buy and sell signals. The
pipeline uses technical indicators and sequence preparation for a GRU classifier. Task 3.1
focuses on buy signal identification and Task 3.2 focuses on sell signal identification.

The final Task 3 summary reports that the buy signal model achieved accuracy 0.4531,
precision 0.3099, recall 0.5301, F1 0.3911, ROC AUC 0.4642, and PR AUC 0.2995. The sell
signal model achieved accuracy 0.5170, precision 0.3582, recall 0.6235, F1 0.4550, ROC
AUC 0.5511, and PR AUC 0.3410. These results indicate that trading signal identification
is more difficult than direct price forecasting in this setup. Because Task 3 is less
stable and is not part of the selected deployment scope, it is not exposed through the
Task 5 API or web application.

## Task 4: Portfolio Composition and Risk Management

Task 4 constructs Vietnam market portfolios using predicted returns and risk-aware scoring.
The workflow builds a ticker universe, creates sequence features across multiple tickers,
trains a GRU return model, scores stocks using profitability and risk indicators, and
constructs prudent and risk-taking portfolios.

The selected universe contains 40 Vietnam stocks with a forecast horizon of 20 days. The
top profitable stocks include LGC, CLC, DHG, DHA, TDH, HMC, VIP, DRC, PAC, SAV, DMC, BMC,
PJT, FMC, and BMP. Risky stocks such as HAS, SCD, GIL, MCP, HBC, ITA, and SMC are excluded
from the safer portfolio construction step. The risk-taking portfolio includes stocks such
as CLC, DHG, LGC, DHA, DRC, HMC, SAV, and DMC with optimized weights. The prudent portfolio
includes SJD, DMC, DHG, TMS, CLC, SAV, RAL, and BMP. These portfolio outputs are saved as
CSV artifacts and deployed through the Task 5 API and dashboard.

## Task 5: Deployment

Task 5 converts the notebook work into a deployable prototype. The deployment exports three
artifact groups: `task1_nasdaq_next_day`, `task2_vietnam_next_day`, and
`task4_vietnam_portfolio`. The price prediction artifacts include a Keras model, feature
scaler, target scaler, and metadata. The portfolio artifact includes the Task 4 return
model, feature scaler, metadata, stock scores, prudent portfolio, and risk-taking portfolio.

The FastAPI service in `deployment/api/api_app.py` exposes REST endpoints. `/health` checks
server status, `/models` lists available artifacts, `/predict/task1` serves Task 1.1,
`/predict/task2` serves Task 2.1, and `/portfolio/task4` returns Task 4 portfolio output.
The API loads trained models and preprocessing files from disk, validates request shapes,
performs prediction, and returns JSON responses.

The Streamlit app in `deployment/app/streamlit_app.py` provides the SaaS interface. Users
can choose Task 1.1 or Task 2.1, edit recent price rows, upload a CSV, or use advanced JSON
input. The app computes technical indicators from user-friendly price tables, sends the
model-ready payload to FastAPI, and displays the predicted next-day price clearly. The
portfolio dashboard displays Task 4 portfolio tables, score charts, weight charts, and CSV
downloads for optional BI tools such as Power BI or Tableau.

## Automation Workflow

The proposed production workflow uses Airbyte, PostgreSQL, dbt or Python, Airflow, FastAPI,
and Streamlit. Airbyte ingests stock data. PostgreSQL stores raw prices, transformed features,
predictions, and portfolio recommendations. dbt or Python transforms raw data into model
features such as returns, moving averages, and volatility. Airflow schedules ingestion,
validation, transformation, inference, storage, and dashboard refresh. FastAPI serves the
trained models, and Streamlit displays predictions and portfolio recommendations. The file
`deployment/docs/postgres_schema.sql` documents the database tables that would support this
workflow.

## Conclusion

The project demonstrates a complete path from data preparation and recurrent modeling to
deployment. Task 1.1 and Task 2.1 provide next-day stock prediction services, while Task 4
provides portfolio recommendations with risk-aware scoring. Task 5 packages these outputs
as a REST API and SaaS-style dashboard. The final system is reproducible locally, documented
with a runbook and workflow design, and suitable for demonstration as an industry-style
machine learning deployment prototype.
