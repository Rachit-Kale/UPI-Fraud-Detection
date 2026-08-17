"""Offline batch pipeline entry point for UPI fraud and anomaly research."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.anomaly_detection import train_anomaly_models
from src.parquet_pipeline import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FIT_ROWS,
    MAPPED_PARQUET_PATH,
    PROCESSED_FEATURES_PARQUET_PATH,
    ParquetPipelineConfig,
    sample_parquet_rows,
    stream_preprocess_to_parquet,
    write_mapped_parquet,
)
from src.supervised_model import train_supervised_models
from src.utils import (
    PROJECT_ROOT,
    REPORTS_DIR,
    ensure_project_dirs,
    save_joblib,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for batch pipeline stages."""
    parser = argparse.ArgumentParser(description="Offline UPI fraud detection pipeline")
    parser.add_argument("--all", action="store_true", help="Run every implemented stage")
    parser.add_argument("--load", action="store_true", help="Load and map raw datasets")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing")
    parser.add_argument("--features", action="store_true", help="Run feature engineering")
    parser.add_argument("--train-supervised", action="store_true", help="Train supervised models")
    parser.add_argument("--train-anomaly", action="store_true", help="Train anomaly models")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Rows per Parquet/CSV processing chunk (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--fit-rows",
        type=int,
        default=DEFAULT_FIT_ROWS,
        help=f"Bounded rows used to fit preprocessing state (default: {DEFAULT_FIT_ROWS})",
    )
    parser.add_argument(
        "--supervised-rows",
        type=int,
        default=500_000,
        help="Bounded balanced rows for supervised batch training",
    )
    parser.add_argument(
        "--anomaly-rows",
        type=int,
        default=200_000,
        help="Bounded stratified rows for anomaly batch training",
    )
    parser.add_argument(
        "--mapped-parquet",
        type=Path,
        default=MAPPED_PARQUET_PATH,
        help="Output path for the unified mapped Parquet file",
    )
    return parser.parse_args()


def main() -> int:
    """Run selected offline batch stages."""
    ensure_project_dirs()
    args = parse_args()

    run_all = args.all or not any(
        [
            args.load,
            args.preprocess,
            args.features,
            args.train_supervised,
            args.train_anomaly,
        ]
    )

    config = ParquetPipelineConfig(
        chunk_size=args.chunk_size,
        fit_rows=args.fit_rows,
        mapped_path=args.mapped_parquet,
        processed_path=PROCESSED_FEATURES_PARQUET_PATH,
    )

    mapped_path = config.mapped_path
    preprocessor = None

    def ensure_mapped_parquet() -> Path:
        if not mapped_path.exists():
            print("Creating unified compressed Parquet dataset...")
            stats = write_mapped_parquet(
                raw_dir=PROJECT_ROOT / "data" / "raw",
                output_path=mapped_path,
                chunk_size=config.chunk_size,
                compression=config.compression,
            )
            print(f"Mapped Parquet: {stats['rows']} rows in {stats['chunks']} chunks")
        return mapped_path

    def ensure_processed_parquet() -> Path:
        nonlocal preprocessor
        ensure_mapped_parquet()
        if preprocessor is None or not config.processed_path.exists():
            preprocessor, stats = stream_preprocess_to_parquet(
                mapped_parquet_path=mapped_path,
                output_path=config.processed_path,
                preprocessor=preprocessor,
                fit_rows=config.fit_rows,
                chunk_size=config.chunk_size,
            )
            save_joblib(preprocessor, PROJECT_ROOT / "models" / "preprocessor.pkl")
            save_joblib(preprocessor, PROJECT_ROOT / "models" / "scaler.pkl")
            print(f"Processed Parquet: {stats['rows']} rows in {stats['chunks']} chunks")
        return config.processed_path

    if run_all or args.load:
        print("Stage 1/5: Converting raw datasets to compressed Parquet chunks...")
        stats = write_mapped_parquet(
            raw_dir=PROJECT_ROOT / "data" / "raw",
            output_path=mapped_path,
            chunk_size=config.chunk_size,
            compression=config.compression,
        )
        reports = {
            "mapped_parquet": {
                "path": str(mapped_path),
                "row_count": stats["rows"],
                "chunk_count": stats["chunks"],
                "columns": [
                    "transaction_id",
                    "timestamp",
                    "amount",
                    "sender_id",
                    "receiver_id",
                    "device_type",
                    "merchant_category",
                    "location",
                    "transaction_type",
                    "fraud_label",
                ],
                "compression": config.compression,
                "chunk_size": config.chunk_size,
            }
        }
        schema_report_path = REPORTS_DIR / "schema_reports.json"
        schema_report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")

    if run_all or args.preprocess or args.features:
        print("Stage 2/5: Fitting once and preprocessing Parquet chunks...")
        ensure_processed_parquet()

    if run_all or args.features:
        print("Stage 3/5: Feature engineering was applied during streaming preprocessing.")
        ensure_mapped_parquet()

    if run_all or args.train_supervised:
        print("Stage 4/5: Training supervised models...")
        processed_path = ensure_processed_parquet()
        supervised_sample = sample_parquet_rows(
            processed_path,
            max_rows=args.supervised_rows,
            chunk_size=config.chunk_size,
            target_column="fraud_label",
            random_state=42,
            columns=None,
        )
        print(f"Using bounded supervised sample: {supervised_sample.shape}")
        supervised_results = train_supervised_models(
            supervised_sample,
            max_rows=len(supervised_sample),
            preprocessor=preprocessor,
            preprocessed=True,
        )
        results_path = REPORTS_DIR / "supervised_metrics.json"
        results_path.write_text(json.dumps(supervised_results, indent=2, default=str), encoding="utf-8")
        del supervised_sample
        print("Supervised model training complete.")

    if run_all or args.train_anomaly:
        print("Stage 5/5: Training anomaly models...")
        processed_path = ensure_processed_parquet()
        anomaly_sample = sample_parquet_rows(
            processed_path,
            max_rows=args.anomaly_rows,
            chunk_size=config.chunk_size,
            target_column="fraud_label",
            random_state=42,
            columns=None,
        )
        print(f"Using bounded anomaly sample: {anomaly_sample.shape}")
        anomaly_results = train_anomaly_models(
            anomaly_sample,
            max_rows=len(anomaly_sample),
            preprocessor=preprocessor,
            preprocessed=True,
        )
        for name, frame in anomaly_results.items():
            frame.to_csv(PROJECT_ROOT / "data" / "processed" / f"{name}_anomaly_scores.csv", index=False)
        del anomaly_sample
        print("Anomaly model training complete.")

    print("Pipeline completed with Parquet-backed chunk processing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
