# CS313 Deep Learning Stock Market Project

This repository contains the final CS313 deep learning project on stock-market
forecasting, trading signals, portfolio construction, and model deployment.

The main notebook implements Tasks 1-5. The deployment package exposes selected trained
models through a FastAPI REST service and a Streamlit web interface.

## Project Scope

| Task | Summary | Deployment Status |
|---|---|---|
| Task 1 | Nasdaq stock price forecasting | Task 1.1 deployed |
| Task 2 | Vietnam stock price forecasting | Task 2.1 deployed |
| Task 3 | Vietnam buy/sell signal classification | Analyzed, not deployed |
| Task 4 | Vietnam portfolio composition and risk management | Deployed as portfolio output |
| Task 5 | API deployment, SaaS interface, and AI workflow design | Implemented |

The deployed Task 5 scope is intentionally limited to:

- Task 1.1: Nasdaq next-day prediction for `GT`
- Task 2.1: Vietnam next-day prediction for `VCB`
- Task 4: Vietnam portfolio recommendation

## Repository Structure

```text
.
|-- 230070_project_notebook.ipynb       # Main notebook for Tasks 1-5
|-- 230070-project-report.pdf           # Final written report
|-- requirements.txt                    # Python dependencies
|-- data/
|   `-- raw/                            # Nasdaq and Vietnam raw datasets
|-- deployment/
|   |-- api/
|   |   `-- api_app.py                  # FastAPI REST API
|   |-- app/
|   |   `-- streamlit_app.py            # Streamlit SaaS dashboard
|   |-- artifacts/                      # Saved models, scalers, metadata, portfolios
|   `-- docs/
|       |-- task5_runbook.md            # Deployment commands and sample requests
|       |-- task5_workflow.md           # AI engineering workflow explanation
|       |-- postgres_schema.sql         # PostgreSQL schema blueprint
|       `-- final_report_draft.md       # Draft report text
|-- figures/                            # Optional generated figures
|-- models/                             # Optional model outputs
`-- results/                            # Optional result outputs
```

## Environment Setup

Use Python 3.11 if available.

```powershell
cd "D:\Projects\Deep Learning Project"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If the environment already exists, activate it only:

```powershell
cd "D:\Projects\Deep Learning Project"
.\.venv\Scripts\activate
```

## Reproduce the Notebook

Open and run:

```text
230070_project_notebook.ipynb
```

The notebook contains:

- reusable preprocessing, feature engineering, scaling, and sequence-building utilities
- Task 1 Nasdaq forecasting experiments
- Task 2 Vietnam forecasting experiments
- Task 3 trading signal classification
- Task 4 portfolio construction and backtesting
- Task 5 artifact export and deployment setup

The notebook is the source of the trained artifacts saved under:

```text
deployment/artifacts/
```

## Run Task 5.1 - FastAPI Model Service

Start the API server:

```powershell
cd "D:\Projects\Deep Learning Project\deployment"
python -m uvicorn api.api_app:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Main endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check that the API is running |
| `GET /models` | List deployed model artifacts |
| `POST /predict/task1` | Predict Task 1.1 Nasdaq next-day price |
| `POST /predict/task2` | Predict Task 2.1 Vietnam next-day price |
| `GET /portfolio/task4` | Load Task 4 portfolio recommendations |

Example Task 1.1 request body:

```json
{
  "features": [
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
  ],
  "scaled": true
}
```

The actual Task 1.1 request must contain 30 rows because the model uses a 30-day
lookback window. Task 2.1 also uses 30 rows, but each row has 9 features instead of 10.

## Run Task 5.2 - Streamlit SaaS Interface

Keep the FastAPI server running. In a second terminal:

```powershell
cd "D:\Projects\Deep Learning Project\deployment"
streamlit run .\app\streamlit_app.py
```

Open:

```text
http://localhost:8501
```

The web interface supports:

- editable recent price tables
- CSV upload
- advanced JSON input for API testing
- next-day prediction display
- Task 4 portfolio dashboard
- portfolio charts and CSV export for optional BI tools

## Task 5.3 - AI Engineering Workflow

The proposed production workflow is:

```text
Airbyte
  -> PostgreSQL raw stock data
  -> dbt/Python feature engineering
  -> Airflow scheduled prediction pipeline
  -> FastAPI model service
  -> PostgreSQL prediction and portfolio tables
  -> Streamlit or Power BI dashboard
```

Supporting documentation:

- `deployment/docs/task5_workflow.md`
- `deployment/docs/postgres_schema.sql`
- `deployment/docs/task5_runbook.md`

PostgreSQL, Airbyte, dbt, and Airflow are included as the production workflow design.
The local runnable prototype uses FastAPI and Streamlit directly.

## Notes on Model Scope

The deployed Task 1.1 and Task 2.1 artifacts are ticker-specific demonstrations:

- `task1_nasdaq_next_day`: trained/deployed for `GT`
- `task2_vietnam_next_day`: trained/deployed for `VCB`

The app should not be interpreted as a universal predictor for every ticker in the raw
data folder unless a generalized multi-ticker model is trained and exported.

## Troubleshooting

If Streamlit cannot connect to the API, make sure FastAPI is running at:

```text
http://127.0.0.1:8000
```

If Swagger returns `422 Unprocessable Entity`, check that:

- `features` is a list of numeric lists
- Task 1.1 has 30 rows x 10 features
- Task 2.1 has 30 rows x 9 features
- explanatory text such as `"... repeated for 30 days ..."` is not included in the actual JSON request

## Submission Artifacts

Recommended files to include for grading:

- `230070_project_notebook.ipynb`
- `230070-project-report.pdf`
- `requirements.txt`
- `deployment/api/api_app.py`
- `deployment/app/streamlit_app.py`
- `deployment/artifacts/`
- `deployment/docs/`
