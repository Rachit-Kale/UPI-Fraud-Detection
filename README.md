# Hybrid Anomaly Detection for Identifying Ambiguous UPI Transactions

Offline, batch-processing research project for post-transaction fraud analysis.

The current implementation covers:

- Dataset loading and schema mapping
- Data preprocessing
- Feature engineering
- Supervised fraud detection with XGBoost and Random Forest
- Unsupervised anomaly detection with Isolation Forest and Local Outlier Factor
- Streamlit testing dashboard for manually checking model behavior

The project intentionally does not implement real-time streaming, APIs, banking integration,
authentication, production deployment, hybrid risk fusion, or a final grey-area decision engine.

## Datasets

Use only the following datasets and place downloaded files under `data/raw/`:

1. PaySim Dataset from Kaggle
2. UPI Transaction 2024 from Kaggle
3. IEEE Fraud Detection from Kaggle
4. Digital Payment Transactions from Zenodo

No synthetic datasets are required or generated.

## Folder Structure

```text
upi-fraud-detection/
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- merged/
|-- notebooks/
|-- src/
|-- models/
|-- app/
|   `-- components/
|-- reports/
|-- requirements.txt
|-- README.md
`-- main.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Batch Pipeline

```bash
python main.py --all
```

The pipeline discovers supported dataset files in `data/raw/`, maps them into the
common schema, preprocesses them, engineers features, trains supervised models, and
trains anomaly detection models.

## Streamlit Testing Dashboard

```bash
streamlit run app/app.py
```

The dashboard is a local testing interface only. It lets you enter or select a sample
transaction, then displays supervised and unsupervised model outputs separately.

## Vite React Testing Dashboard

The React dashboard uses a small local FastAPI service because browser JavaScript cannot
load Python `joblib`, scikit-learn, XGBoost, or Isolation Forest model artifacts directly.

Start the API:

```bash
uvicorn api.main:app --reload
```

Start the React app in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

The API is local-only and is used only for manual offline model testing.

## Common Schema

Every dataset is mapped into:

- `transaction_id`
- `timestamp`
- `amount`
- `sender_id`
- `receiver_id`
- `device_type`
- `merchant_category`
- `location`
- `transaction_type`
- `fraud_label`

## Important Boundary

This project stops at supervised fraud detection and unsupervised anomaly detection.
It does not combine both outputs into a final grey-area or risk-fusion decision.
