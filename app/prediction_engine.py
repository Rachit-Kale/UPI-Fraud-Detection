"""Prediction helpers for the Streamlit testing dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.anomaly_detection import predict_anomaly
from src.schema_mapping import COMMON_SCHEMA
from src.supervised_model import predict_fraud_probability
from src.utils import MODELS_DIR, load_joblib


class PredictionEngine:
    """Load trained models and produce separate supervised/anomaly outputs."""

    def __init__(self, model_dir: Path | str = MODELS_DIR) -> None:
        self.model_dir = Path(model_dir)
        self.supervised_model = self._load_first_available(
            ["xgboost_model.pkl", "random_forest.pkl"]
        )
        self.supervised_preprocessor = self._load_optional("preprocessor.pkl") or self._load_optional("scaler.pkl")
        self.anomaly_model = self._load_optional("isolation_forest.pkl")
        self.anomaly_preprocessor = self._load_optional("anomaly_preprocessor.pkl") or self.supervised_preprocessor

    @property
    def is_ready(self) -> bool:
        """Return True when all dashboard predictions can run."""
        return all(
            [
                self.supervised_model is not None,
                self.supervised_preprocessor is not None,
                self.anomaly_model is not None,
                self.anomaly_preprocessor is not None,
            ]
        )

    def predict(self, transaction: dict[str, Any]) -> dict[str, pd.DataFrame]:
        """Run supervised fraud and unsupervised anomaly prediction separately."""
        if not self.is_ready:
            raise FileNotFoundError(
                "Trained models are missing. Run `python main.py --all` after placing datasets in data/raw/."
            )

        frame = transaction_to_common_schema(transaction)
        supervised = predict_fraud_probability(
            self.supervised_model,
            self.supervised_preprocessor,
            frame,
        )
        anomaly = predict_anomaly(self.anomaly_model, self.anomaly_preprocessor, frame)
        return {"supervised": supervised, "anomaly": anomaly}

    def _load_optional(self, filename: str) -> Any | None:
        path = self.model_dir / filename
        if not path.exists():
            return None
        return load_joblib(path)

    def _load_first_available(self, filenames: list[str]) -> Any | None:
        for filename in filenames:
            model = self._load_optional(filename)
            if model is not None:
                return model
        return None


def transaction_to_common_schema(transaction: dict[str, Any]) -> pd.DataFrame:
    """Convert a dashboard input dictionary into the common schema dataframe."""
    row = {
        "transaction_id": transaction.get("transaction_id", "manual_test_0001"),
        "timestamp": pd.to_datetime(transaction.get("timestamp"), errors="coerce"),
        "amount": float(transaction.get("amount", 0.0)),
        "sender_id": str(transaction.get("sender_id", "Unknown")),
        "receiver_id": str(transaction.get("receiver_id", "Unknown")),
        "device_type": str(transaction.get("device_type", "Unknown")),
        "merchant_category": str(transaction.get("merchant_category", "Unknown")),
        "location": str(transaction.get("location", "Unknown")),
        "transaction_type": str(transaction.get("transaction_type", "Unknown")),
        "fraud_label": int(transaction.get("fraud_label", 0)),
    }
    frame = pd.DataFrame([row], columns=COMMON_SCHEMA)
    frame["timestamp"] = frame["timestamp"].fillna(pd.Timestamp.now())
    frame["amount"] = frame["amount"].replace([np.inf, -np.inf], 0).fillna(0)
    return frame
