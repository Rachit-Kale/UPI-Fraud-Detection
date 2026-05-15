# UPI Fraud Detection Project Run Commands

This guide assumes the project folder is:

```powershell
D:\MAJOR PROJECT\Demo\upi-fraud-detection
```

Run all commands from PowerShell.

## 1. Open Project Folder

```powershell
cd "D:\MAJOR PROJECT\Demo\upi-fraud-detection"
```

## 2. Add Datasets

Place the dataset CSV files inside:

```text
D:\MAJOR PROJECT\Demo\upi-fraud-detection\data\raw
```

Expected examples:

```text
data\raw\digital_payment_transactions.csv
data\raw\ieee_transaction.csv
data\raw\paysim.csv
data\raw\upi_transaction_2024.csv
```

## 3. Create Python Virtual Environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\activate
```

You should see this at the beginning of the terminal line:

```text
(.venv)
```

## 4. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

The React API uses FastAPI, so make sure these packages install successfully:

```powershell
pip install fastapi uvicorn pydantic
```

## 5. Run Full ML Pipeline

```powershell
python main.py --all
```

This performs:

- dataset loading
- schema mapping
- preprocessing
- feature engineering
- supervised model training
- anomaly model training
- model saving into `models\`
- report saving into `reports\`

Important: the datasets are large, so this may take time.

## 6. Run Only Specific Pipeline Stages

Load and map datasets only:

```powershell
python main.py --load
```

Preprocess only:

```powershell
python main.py --preprocess
```

Feature engineering only:

```powershell
python main.py --features
```

Train supervised models only:

```powershell
python main.py --train-supervised
```

Train anomaly models only:

```powershell
python main.py --train-anomaly
```

## 7. Run Streamlit Dashboard

Use this if you want the original Python dashboard:

```powershell
streamlit run app\app.py
```

Open the URL shown in the terminal, usually:

```text
http://localhost:8501
```

## 8. Run React Dashboard With FastAPI

React cannot directly load Python `.pkl` model files, so run the local FastAPI service first.

Terminal 1: start the API from the project root:

```powershell
cd "D:\MAJOR PROJECT\Demo\upi-fraud-detection"
.\.venv\Scripts\activate
uvicorn api.main:app --reload
```

API URL:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

Terminal 2: start the React app:

```powershell
cd "D:\MAJOR PROJECT\Demo\upi-fraud-detection\frontend"
npm install
npm run dev
```

React dashboard URL:

```text
http://localhost:5173
```

React dashboard pages:

- `Simulation`: manual transaction testing with supervised and anomaly outputs.
- `Workflow`: animated visual explanation of the full offline project pipeline.
- `Analytics`: imported dataset statistics, fraud rate, transaction mix, amount distribution, hourly volume, device mix, and location charts.

Useful API endpoints:

```text
http://127.0.0.1:8000/models/status
http://127.0.0.1:8000/analytics/data
http://127.0.0.1:8000/reports/summary
```

## 9. Node.js Requirement For React

Vite React needs a modern Node.js version.

Recommended:

```text
Node.js 18 or newer
```

Check your version:

```powershell
node --version
```

Check npm:

```powershell
npm --version
```

If `npm install` or `npm run dev` fails because Node is old, install the latest LTS version of Node.js, then reopen PowerShell.

## 10. Expected Model Files

After successful training, these files should exist:

```text
models\xgboost_model.pkl
models\random_forest.pkl
models\isolation_forest.pkl
models\lof_model.pkl
models\preprocessor.pkl
models\anomaly_preprocessor.pkl
models\scaler.pkl
```

If the React or Streamlit dashboard says models are missing, rerun:

```powershell
python main.py --all
```

## 11. Common Problems

If FastAPI does not start:

```powershell
pip install fastapi uvicorn pydantic
```

If React cannot connect to the API, make sure this command is still running in Terminal 1:

```powershell
uvicorn api.main:app --reload
```

If port `8000` is busy:

```powershell
uvicorn api.main:app --reload --port 8001
```

Then update `frontend\vite.config.js` proxy target from:

```js
target: "http://127.0.0.1:8000"
```

to:

```js
target: "http://127.0.0.1:8001"
```

If port `5173` is busy:

```powershell
npm run dev -- --port 5174
```

Then open:

```text
http://localhost:5174
```

## 12. Recommended Demo Flow

1. Run `python main.py --all`.
2. Confirm model files exist inside `models\`.
3. Start FastAPI with `uvicorn api.main:app --reload`.
4. Start React with `npm run dev`.
5. Open `http://localhost:5173`.
6. Select a preset transaction.
7. Click `Run Test`.
8. Show supervised output and anomaly output separately.

Remember: this project stops at supervised fraud detection and unsupervised anomaly detection. It does not implement a final grey-area fusion engine yet.
