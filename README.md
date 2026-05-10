# CS313 Deep Learning Stock Market Project

This repository contains a stock-market deep learning project covering Nasdaq price
forecasting, Vietnam market forecasting, trading signal classification, portfolio
construction, and a Task 5 deployment prototype.

## Project Structure

```text
.
├── 230070_project_notebook.ipynb       # Main notebook for Tasks 1-5
├── data/
│   └── raw/                            # Nasdaq and Vietnam historical data
├── deployment/
│   ├── api/api_app.py                  # FastAPI model service
│   ├── app/streamlit_app.py            # Streamlit SaaS dashboard
│   ├── artifacts/                      # Saved models, scalers, metadata, portfolios
│   └── docs/                           # Task 5 runbook, workflow, PostgreSQL schema
└── requirements.txt
```

## Deployed Scope

Task 5 deploys:

- Task 1.1: Nasdaq next-day prediction for `GT`
- Task 2.1: Vietnam next-day prediction for `VCB`
- Task 4: Vietnam portfolio recommendation

Task 3 is included in the notebook analysis but is not deployed.

## Setup

Create and activate a Python environment, then install dependencies:

```powershell
cd "D:\Projects\Deep Learning Project"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the API

```powershell
cd "D:\Projects\Deep Learning Project\deployment"
python -m uvicorn api.api_app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Useful endpoints:

- `GET /health`
- `GET /models`
- `POST /predict/task1`
- `POST /predict/task2`
- `GET /portfolio/task4`

## Run the Web App

Open a second terminal:

```powershell
cd "D:\Projects\Deep Learning Project\deployment"
streamlit run .\app\streamlit_app.py
```

Open:

```text
http://localhost:8501
```

The app supports editable recent price tables, CSV upload, advanced JSON API input,
portfolio dashboard tables/charts, and CSV export for optional BI tools.

## Task 5 Documentation

See:

- `deployment/docs/task5_runbook.md`
- `deployment/docs/task5_workflow.md`
- `deployment/docs/postgres_schema.sql`
- `deployment/docs/final_report_draft.md`

## Notes

The saved Task 1.1 and Task 2.1 models are ticker-specific deployment artifacts. The
current app is designed to demonstrate deployment for the trained tickers, not to claim
that one ticker-specific model generalizes to every ticker in the raw data folder.
