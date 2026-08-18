# UPI Fraud Detection: Run Commands

This guide is for the offline research pipeline in:

```text
D:/MAJOR PROJECT/Demo/upi-fraud-detection
```

Run Python commands from the project root. Run React commands from the frontend folder.

## 1. Open the Project

```powershell
cd "D:/MAJOR PROJECT/Demo/upi-fraud-detection"
```

## 2. Create and Activate the Python Environment

Create the environment once:

```powershell
python -m venv .venv
```

Activate it for every new PowerShell window:

```powershell
./.venv/Scripts/activate
```

Confirm that the prompt starts with (.venv).

## 3. Install Python Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The requirements include the Parquet engine, PyArrow, and the API packages used by the React frontend.

## 4. Add the Datasets

Copy only the approved dataset CSV files into:

```text
D:/MAJOR PROJECT/Demo/upi-fraud-detection/data/raw
```

Expected files are:

```text
data/raw/digital_payment_transactions.csv
data/raw/ieee_transaction.csv
data/raw/paysim.csv
data/raw/upi_transaction_2024.csv
```

Check that the files are present:

```powershell
Get-ChildItem ./data/raw
```

Do not place generated Parquet files in data/raw. The pipeline creates them in data/merged and data/processed.

## 5. Run the Complete ML Pipeline

For the normal run, use:

```powershell
python main.py --all
```

The complete execution order is:

1. Load each raw CSV in chunks and map it to the common schema.
2. Write the mapped records to compressed Parquet at data/merged/mapped_common_schema.parquet.
3. Fit the preprocessing state once on a bounded fitting sample.
4. Read mapped Parquet chunks, apply the same preprocessing and feature engineering, and write data/processed/processed_features.parquet.
5. Train XGBoost and Random Forest once using the configured supervised sample.
6. Train Isolation Forest and LOF once using the configured anomaly sample.
7. Save models, preprocessing state, evaluation metrics, and reports.

The process is batch-based and may take time because it scans millions of records. It should not keep the complete raw dataset in RAM.

## 6. Recommended Demonstration Run

This command keeps the default chunk size at 100,000 rows and uses bounded training samples:

```powershell
python main.py --all --chunk-size 100000 --fit-rows 100000 --supervised-rows 500000 --legitimate-ratio 3 --anomaly-rows 200000
```

Use this first to verify the complete project. Increase the training row limits later if your machine has enough memory and time.

## 7. Adjust Chunk and Training Sizes

The chunk size controls how many mapped Parquet rows are processed at a time:

```powershell
python main.py --all --chunk-size 50000
python main.py --all --chunk-size 250000
```

Use a smaller chunk size when RAM is limited. Use a larger chunk size when disk I/O is the bottleneck.

The fitting and model-training limits are separate:

```powershell
python main.py --all --fit-rows 200000 --supervised-rows 700000 --anomaly-rows 300000
```

These limits do not delete records from Parquet. They only bound the samples materialized for state fitting and batch model training.

For supervised training, `--legitimate-ratio 3` keeps all available fraud rows when possible and selects approximately three legitimate rows per fraud row. Legitimate rows are selected proportionally across available hour/day strata. Random Forest and XGBoost also retain class weighting.

For anomaly training, the sampler is uniform and does not use fraud labels because anomaly detection is unsupervised.

## 8. Run Individual Stages

The safest option is always --all, because later stages depend on the Parquet artifacts produced earlier. Individual commands are available for reruns:

Load and create mapped Parquet:

```powershell
python main.py --load --chunk-size 100000
```

Stream preprocessing and feature engineering into processed Parquet:

```powershell
python main.py --preprocess --chunk-size 100000 --fit-rows 100000
```

Run the streaming feature stage explicitly:

```powershell
python main.py --features --chunk-size 100000 --fit-rows 100000
```

Train supervised models:

```powershell
python main.py --train-supervised --supervised-rows 500000 --legitimate-ratio 3
```

Train anomaly models:

```powershell
python main.py --train-anomaly --anomaly-rows 200000
```

If a required Parquet artifact does not exist, rerun:

```powershell
python main.py --all
```

## 9. Verify Parquet Output and Row Counts

Check the mapped Parquet file:

```powershell
python -c "import pyarrow.parquet as pq; p=pq.ParquetFile('data/merged/mapped_common_schema.parquet'); print('mapped rows:', p.metadata.num_rows); print('row groups:', p.num_row_groups); print('columns:', p.schema_arrow.names)"
```

Check the processed feature Parquet file:

```powershell
python -c "import pyarrow.parquet as pq; p=pq.ParquetFile('data/processed/processed_features.parquet'); print('processed rows:', p.metadata.num_rows); print('row groups:', p.num_row_groups); print('columns:', p.schema_arrow.names)"
```

Check saved model and report files:

```powershell
Get-ChildItem ./models
Get-ChildItem ./reports
```

The mapped and processed row counts should match. Duplicate transaction IDs now stop the pipeline instead of being silently removed. Detailed row accounting is saved to `reports/pipeline_row_counts.json`.

## 10. Start the FastAPI Backend

Keep the virtual environment active and open a new PowerShell window in the project root:

```powershell
cd "D:/MAJOR PROJECT/Demo/upi-fraud-detection"
./.venv/Scripts/activate
uvicorn api.main:app --reload
```

Leave this terminal running. Verify the backend in another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/models/status
```

The backend reads the generated models and Parquet analytics data. Run the ML pipeline before starting the backend if the model files do not exist.

## 11. Start the React Frontend

Open a second PowerShell window:

```powershell
cd "D:/MAJOR PROJECT/Demo/upi-fraud-detection/frontend"
npm install
npm run dev
```

Open the URL printed by Vite, normally:

```text
http://localhost:5173
```

npm install is required only after a fresh checkout or when package.json changes. If PowerShell blocks the npm wrapper, use:

```powershell
npm.cmd install
npm.cmd run dev
```

The frontend pages are:

- Simulation: enter and test a transaction.
- Workflow: view the animated end-to-end research workflow.
- Analytics: inspect imported dataset distributions and model metrics.
- Finance Research Report: view the one-page transaction report, processing evidence, and grey-area comparison graph.
- Prediction Logs: open previous test transactions and inspect their brief details.

## 12. Test the Prediction Flow

With both the API and Vite terminals running:

1. Open the React URL.
2. Go to Simulation.
3. Select a preset or enter a transaction manually.
4. Select Run Test.
5. Review the separate supervised fraud probability and unsupervised anomaly score.
6. Review the final fusion score and research resolution in Model Outputs.
7. Open Finance Research Report for the complete explanation.
8. Open Prediction Logs and click a previous transaction to inspect it.

The project is intentionally post-transaction and research-oriented. It does not implement real-time payment processing, banking integration, authentication, or production deployment.

## 13. Troubleshooting

Missing dataset error:

```powershell
Get-ChildItem ./data/raw
```

If a required file is missing, copy it into data/raw and rerun:

```powershell
python main.py --all
```

Missing model files or HTTP 500 from the prediction endpoint:

```powershell
python main.py --all --chunk-size 100000 --fit-rows 100000 --supervised-rows 500000 --legitimate-ratio 3 --anomaly-rows 200000
```

API connection error in React:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Start the API with uvicorn api.main:app --reload and leave it running while using Vite.

Slow analytics page:

The backend scans Parquet in bounded chunks to calculate population counts and uses a bounded sample for chart points. The first analytics request can therefore take longer than a cached request.

## 14. More Pipeline Details

See Pipeline.md for the Parquet architecture, chunk-processing behavior, memory notes, reproducibility guidance, and detailed command-line options.
