# Task 5 Deployment Runbook

This runbook documents how to start and test the local Task 5 deployment.

## Scope

The deployment includes:

- Task 1.1: Nasdaq next-day price prediction for GT
- Task 2.1: Vietnam next-day price prediction for VCB
- Task 4: Vietnam portfolio recommendation

Task 3 is not deployed.

## Start the API Server

From the project root:

```powershell
cd "D:\Projects\Deep Learning Project\deployment"
python -m uvicorn api.api_app:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

Useful endpoints:

- `GET /health`
- `GET /models`
- `POST /predict/task1`
- `POST /predict/task2`
- `GET /portfolio/task4`

## Start the Web Interface

Open a second terminal:

```powershell
cd "D:\Projects\Deep Learning Project\deployment"
streamlit run .\app\streamlit_app.py
```

Open:

```text
http://localhost:8501
```

The Price Prediction tab has three input modes:

- `Edit recent price table`: recommended for demos. The user edits recent Open/High/Low/Close/Volume rows, and the app computes indicators automatically.
- `Upload CSV`: upload a recent price CSV with the required columns.
- `Advanced JSON`: show the raw FastAPI request body for API documentation screenshots.

## Sample API Request

The local demo uses scaled sample values to verify that the model server loads the
trained model and returns a prediction.

```powershell
$body = @{
  features = @(
    @(0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5)
  ) * 30
  scaled = $true
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict/task1" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Expected output shape:

```json
{
  "task": "Task 1.1 - Nasdaq next-day prediction",
  "artifact_folder": "task1_nasdaq_next_day",
  "ticker": "GT",
  "prediction_type": "next_day_prediction",
  "predicted_next_day_value": 28.739456176757812
}
```

## Task 4 Portfolio Output

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/portfolio/task4"
```

Expected output includes:

- `prudent_portfolio`
- `risk_taking_portfolio`
- `stock_scores`

## Power BI Export

The Streamlit Portfolio Dashboard tab provides CSV download buttons for:

- stock score table
- prudent portfolio
- risk-taking portfolio

These files can be imported into Power BI, Tableau, or Superset for dashboarding.
