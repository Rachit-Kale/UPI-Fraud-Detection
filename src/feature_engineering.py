"""Feature engineering for offline transaction fraud and anomaly analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema_mapping import validate_common_schema
from src.utils import get_logger, safe_datetime


RAPID_TRANSACTION_WINDOW_MINUTES = 5
LOGGER = get_logger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate behavioral, velocity, risk, and temporal features."""
    LOGGER.info("Starting feature engineering on %s rows", len(df))
    validate_common_schema(df)
    features = df.copy()
    features["timestamp"] = safe_datetime(features["timestamp"])
    features["timestamp"] = features["timestamp"].fillna(pd.Timestamp("2024-01-01"))
    features["amount"] = pd.to_numeric(features["amount"], errors="coerce").fillna(0.0)
    features["sender_id"] = features["sender_id"].astype("category")
    features["receiver_id"] = features["receiver_id"].astype("category")
    features["merchant_category"] = features["merchant_category"].astype("category")
    features["device_type"] = features["device_type"].astype("category")
    features["location"] = features["location"].astype("category")

    LOGGER.info("Adding temporal features")
    features = add_temporal_features(features)
    LOGGER.info("Adding behavioral features")
    features = add_behavioral_features(features)
    LOGGER.info("Adding velocity features")
    features = add_velocity_features(features)
    LOGGER.info("Adding risk features")
    features = add_risk_features(features)
    LOGGER.info("Feature engineering complete with %s columns", len(features.columns))
    return features


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day-of-week, and weekend indicators."""
    df["hour_of_day"] = df["timestamp"].dt.hour.astype("int8")
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int8")
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int8")
    return df


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sender-level behavioral aggregates."""
    sender_group = df.groupby("sender_id", dropna=False, sort=False, observed=True)

    df["avg_transaction_amount"] = sender_group["amount"].transform("mean").astype("float32")
    df["transaction_frequency"] = sender_group["transaction_id"].transform("count").astype("int32")
    df["merchant_diversity"] = sender_group["merchant_category"].transform("nunique").astype("int16")
    df["device_switching_frequency"] = sender_group["device_type"].transform("nunique").astype("int16")

    sender_hour_group = df.groupby(["sender_id", "hour_of_day"], dropna=False, sort=False, observed=True)
    df["transactions_per_hour"] = sender_hour_group["transaction_id"].transform("count").astype("int16")
    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rapid transaction, amount spike, and high-frequency payment features."""
    df = df.sort_values(["sender_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    df["minutes_since_previous_sender_txn"] = (
        df.groupby("sender_id", sort=False, observed=True)["timestamp"].diff().dt.total_seconds().div(60)
    ).astype("float32")
    df["rapid_transactions"] = (
        df["minutes_since_previous_sender_txn"].fillna(np.inf)
        <= RAPID_TRANSACTION_WINDOW_MINUTES
    ).astype("int8")

    sender_group = df.groupby("sender_id", dropna=False, sort=False, observed=True)["amount"]
    sender_mean = sender_group.transform("mean")
    sender_std = sender_group.transform("std").fillna(0)
    df["amount_spike"] = (df["amount"] > sender_mean + (3 * sender_std)).astype("int8")

    hourly_threshold = df["transactions_per_hour"].quantile(0.95)
    if pd.isna(hourly_threshold):
        hourly_threshold = 1
    df["high_frequency_payments"] = (
        df["transactions_per_hour"] >= max(2, hourly_threshold)
    ).astype("int8")
    return df


def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add unusual timing, new payee, and unusual location flags."""
    df["unusual_transaction_timing"] = df["hour_of_day"].between(0, 5).astype("int8")
    df["new_payee_flag"] = (
        ~df.duplicated(subset=["sender_id", "receiver_id"], keep="first")
    ).astype("int8")

    LOGGER.info("Calculating usual location per sender")
    usual_location = _most_frequent_value_by_group(df, group_column="sender_id", value_column="location")
    mapped_location = df["sender_id"].astype(str).map(usual_location.astype(str).to_dict())
    df["usual_location"] = mapped_location.fillna("Unknown")
    df["unusual_location_flag"] = (df["location"].astype(str) != df["usual_location"].astype(str)).astype("int8")
    df = df.drop(columns=["usual_location"])
    return df


def feature_columns_for_model(df: pd.DataFrame, target_column: str = "fraud_label") -> list[str]:
    """Return model feature columns after excluding identifiers and labels."""
    excluded = {
        target_column,
        "transaction_id",
        "timestamp",
    }
    return [column for column in df.columns if column not in excluded]


def _most_frequent_value_by_group(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> pd.Series:
    """Return the most frequent value for each group without per-group Python mode calls."""
    counts = (
        df.groupby([group_column, value_column], sort=False, observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    if counts.empty:
        return pd.Series(dtype="object")

    winners = counts.loc[counts.groupby(group_column, sort=False, observed=True)["count"].idxmax()]
    return winners.set_index(group_column)[value_column]
