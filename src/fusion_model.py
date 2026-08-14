"""Explainable research fusion for ambiguous transaction analysis.

This module combines already-trained supervised and unsupervised outputs. It is
not a third classifier and must not be treated as a production risk decision.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np


FUSION_WEIGHTS = {
    "supervised_fraud_probability": 0.60,
    "unsupervised_unusualness_percentile": 0.40,
}


def fuse_signals(
    supervised: dict[str, Any],
    anomaly: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a transparent score and an ambiguity-oriented resolution.

    The fusion score measures combined concern. The ambiguity score intentionally
    measures a different idea: disagreement between the model families and
    uncertainty around the supervised decision boundary. A high ambiguity score
    sends the transaction to a research-review band instead of calling it an
    anomaly or making a banking decision.
    """
    fraud_probability = _unit_interval(supervised.get("fraud_probability", 0.0))
    unusualness_percentile = _unit_interval(
        anomaly.get("anomaly_percentile", anomaly.get("anomaly_confidence", 0.0))
    )

    disagreement = abs(fraud_probability - unusualness_percentile)
    supervised_uncertainty = 1.0 - abs((2.0 * fraud_probability) - 1.0)
    repeatability_penalty = _repeatability_penalty(diagnostics)
    fusion_score = (
        FUSION_WEIGHTS["supervised_fraud_probability"] * fraud_probability
        + FUSION_WEIGHTS["unsupervised_unusualness_percentile"] * unusualness_percentile
    )
    ambiguity_score = (
        0.50 * disagreement
        + 0.35 * supervised_uncertainty
        + 0.15 * repeatability_penalty
    )

    if ambiguity_score >= 0.38 or disagreement >= 0.45:
        resolution = "AMBIGUOUS_REVIEW"
        resolution_text = "Falls within the ambiguous research-review band"
    elif fusion_score >= 0.60:
        resolution = "FRAUD_LIKELY"
        resolution_text = "Fraud signal is elevated across the combined evidence"
    else:
        resolution = "LIKELY_LEGITIMATE"
        resolution_text = "Combined evidence is currently closer to legitimate behaviour"

    return {
        "fusion_score": round(float(fusion_score), 6),
        "ambiguity_score": round(float(ambiguity_score), 6),
        "resolution": resolution,
        "resolution_text": resolution_text,
        "signal_disagreement": round(float(disagreement), 6),
        "supervised_uncertainty": round(float(supervised_uncertainty), 6),
        "repeatability_penalty": round(float(repeatability_penalty), 6),
        "weights": FUSION_WEIGHTS.copy(),
        "method": "Weighted dual-signal score with disagreement and uncertainty review band",
    }


def build_transaction_report(
    transaction: dict[str, Any],
    supervised: dict[str, Any],
    anomaly: dict[str, Any],
    fusion: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe, downloadable post-transaction research report."""
    return {
        "report_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": "Offline post-transaction research analysis only",
        "transaction": _json_safe(transaction),
        "supervised_model_output": _json_safe(supervised),
        "unsupervised_model_output": _json_safe(anomaly),
        "fusion_resolution": _json_safe(fusion),
        "model_evidence": _json_safe(diagnostics or {}),
        "interpretation": (
            "The resolution is a research score for comparing supervised fraud "
            "probability and unsupervised unusualness. It is not an approval, "
            "decline, block, or proof of fraud."
        ),
    }


def _repeatability_penalty(diagnostics: dict[str, Any] | None) -> float:
    """Map same-input repeat deltas to a bounded numerical-stability penalty."""
    deterministic = (diagnostics or {}).get("deterministic", {})
    fraud_delta = abs(float(deterministic.get("repeat_fraud_delta", 0.0)))
    anomaly_delta = abs(float(deterministic.get("repeat_anomaly_delta", 0.0)))
    return _unit_interval((fraud_delta + anomaly_delta) / 2.0)


def _unit_interval(value: Any) -> float:
    """Coerce a scalar into the inclusive 0-1 interval."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(numeric_value):
        return 0.0
    return float(np.clip(numeric_value, 0.0, 1.0))


def _json_safe(value: Any) -> Any:
    """Convert common numpy/pandas scalar values into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value
