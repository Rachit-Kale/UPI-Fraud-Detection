"""Offline batch pipeline entry point for UPI fraud and anomaly research."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.anomaly_detection import train_anomaly_models
from src.data_loader import generate_schema_reports, load_all_datasets, load_and_map_all
from src.data_preprocessing import PreprocessingConfig, UPITransactionPreprocessor
from src.feature_engineering import engineer_features
from src.sampling import balanced_binary_sample
from src.supervised_model import train_supervised_models
from src.utils import (
    MERGED_DATA_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    ensure_project_dirs,
    save_dataframe,
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

    mapped = None
    engineered = None

    if run_all or args.load:
        print("Stage 1/5: Loading datasets and generating schema reports...")
        datasets = load_all_datasets()
        reports = generate_schema_reports(datasets)
        schema_report_path = REPORTS_DIR / "schema_reports.json"
        schema_report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        mapped = load_and_map_all()
        save_dataframe(mapped, MERGED_DATA_DIR / "merged_common_schema.csv")

    if run_all or args.preprocess:
        print("Stage 2/5: Preprocessing mapped dataset...")
        mapped = mapped if mapped is not None else load_and_map_all()
        preprocessor = UPITransactionPreprocessor(PreprocessingConfig())
        x, y = preprocessor.fit_transform(mapped)
        save_joblib(preprocessor, PROJECT_ROOT / "models" / "preprocessor.pkl")
        save_dataframe(mapped, PROCESSED_DATA_DIR / "preprocessed_common_schema.csv")
        print(f"Preprocessed shape: X={x.shape}, y={None if y is None else y.shape}")

    if run_all or args.features:
        print("Stage 3/5: Engineering features...")
        mapped = mapped if mapped is not None else load_and_map_all()
        feature_df = balanced_binary_sample(
            mapped,
            target_column="fraud_label",
            max_rows=300_000,
            random_state=42,
        )
        print(f"Using balanced feature-engineering sample: {feature_df.shape}")
        engineered = engineer_features(feature_df)
        save_dataframe(engineered, PROCESSED_DATA_DIR / "engineered_features.csv")
        print(f"Engineered feature dataset shape: {engineered.shape}")

    if run_all or args.train_supervised:
        print("Stage 4/5: Training supervised models...")
        mapped = mapped if mapped is not None else load_and_map_all()
        supervised_results = train_supervised_models(mapped)
        results_path = REPORTS_DIR / "supervised_metrics.json"
        results_path.write_text(json.dumps(supervised_results, indent=2, default=str), encoding="utf-8")
        print("Supervised model training complete.")

    if run_all or args.train_anomaly:
        print("Stage 5/5: Training anomaly models...")
        mapped = mapped if mapped is not None else load_and_map_all()
        anomaly_results = train_anomaly_models(mapped)
        for name, frame in anomaly_results.items():
            save_dataframe(frame, PROCESSED_DATA_DIR / f"{name}_anomaly_scores.csv")
        print("Anomaly model training complete.")

    print("Pipeline completed up to supervised and unsupervised anomaly detection.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
