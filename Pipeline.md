# Parquet and Chunked Pipeline

The batch pipeline now converts the mapped transaction data to compressed Apache
Parquet before preprocessing. It processes row groups in bounded chunks, so the
7.2M-row dataset is not concatenated into one in-memory dataframe.

## Run the pipeline

From the project root:

```powershell
\.venv\Scripts\activate
python main.py --all
```

The stages are:

1. Raw CSV/Parquet files are read in chunks and mapped to the existing common schema.
2. The mapped rows are written to `data/merged/mapped_common_schema.parquet` with Snappy compression.
3. A bounded sample is used to fit preprocessing state once.
4. Every Parquet row group is feature-engineered and transformed using that same fitted state.
5. The existing batch-only ML models train once on bounded samples.

The transformed output is written to:

```text
data/processed/processed_features.parquet
```

## Adjust the chunk size

The default is `100000` rows per chunk. Change it from the command line:

```powershell
python main.py --all --chunk-size 50000
```

Use a smaller value when RAM is limited. Use a larger value when the machine has
more memory and Parquet conversion is I/O-bound. A practical starting range is
`50000` to `250000`.

The number of rows used to fit preprocessing state is independently configurable:

```powershell
python main.py --all --chunk-size 100000 --fit-rows 200000
```

`--fit-rows` does not discard rows from the transformed Parquet output. It only
controls the bounded sample used to learn imputer, scaler, encoder, and outlier
clipping state. Unknown categories are handled by the existing encoder behavior.

## Model training limits

The current algorithms are batch estimators:

- Random Forest does not implement `partial_fit`.
- XGBoost is trained with one `fit` call in the existing module.
- Isolation Forest is trained with one `fit` call.
- LOF with `novelty=True` is trained with one `fit` call.

Therefore the pipeline does not call `fit` once per chunk. That would overwrite
previous learning or create inconsistent models. Instead, Parquet is scanned in
chunks and a reproducible bounded sample is passed to each existing batch model.
Adjust those sample limits when needed:

```powershell
python main.py --all --supervised-rows 700000 --anomaly-rows 200000
```

Supervised sampling is class-balanced when both labels exist. Anomaly sampling is
also bounded and preserves the existing anomaly training function and algorithms.

## Parquet behavior

- `pyarrow.parquet.ParquetFile.iter_batches` is used for Parquet reads.
- CSV inputs use `pandas.read_csv(..., chunksize=...)` and only columns needed for
  schema mapping are read.
- Parquet writes use Snappy compression and one row group per processing chunk.
- The API analytics endpoint reads only the columns required for its reports.
- Generated transaction IDs use a source offset, so chunk boundaries do not create
  duplicate IDs for datasets without an ID column.

## Correctness and memory checks

The conversion logs each mapped chunk and cumulative row count. The preprocessing
logs each tenth processed chunk and cumulative row count. The final Parquet metadata
can be checked with:

```powershell
python -c "import pyarrow.parquet as pq; p=pq.ParquetFile('data/merged/mapped_common_schema.parquet'); print(p.metadata.num_rows, p.num_row_groups, p.schema.names)"
```

The sum of row-group counts should equal the mapped row count in
`reports/schema_reports.json`. The processed Parquet file should have the same row
count after the existing duplicate-removal behavior is applied within each processing chunk.

For a quick read test:

```powershell
python -c "import pyarrow.parquet as pq; t=pq.read_table('data/merged/mapped_common_schema.parquet', columns=['transaction_id','amount','fraud_label']); print(t.num_rows, t.column_names)"
```

## Important limitation

Feature engineering is invoked once for each bounded chunk, while the fitted
preprocessor is shared across all chunks. This keeps the existing feature formulas
and avoids fitting scalers/encoders independently. The non-incremental model
algorithms remain batch-trained by design; chunking is used for ingestion,
Parquet conversion, preprocessing, analytics, and bounded model sampling.
