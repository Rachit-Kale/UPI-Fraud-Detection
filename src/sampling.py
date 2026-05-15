"""Sampling helpers for large offline modeling datasets."""

from __future__ import annotations

import pandas as pd


def balanced_binary_sample(
    df: pd.DataFrame,
    target_column: str,
    max_rows: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a bounded, class-balanced sample for binary classification."""
    if len(df) <= max_rows or target_column not in df.columns:
        return df.copy()

    label_counts = df[target_column].value_counts(dropna=False)
    if set(label_counts.index) != {0, 1}:
        return df.sample(n=max_rows, random_state=random_state, replace=False).reset_index(drop=True)

    minority_count = int(label_counts.min())
    per_class = min(minority_count, max_rows // 2)
    if per_class == 0:
        return df.sample(n=min(max_rows, len(df)), random_state=random_state, replace=False).reset_index(drop=True)

    sampled_frames = []
    for label in [0, 1]:
        subset = df[df[target_column] == label]
        if len(subset) <= per_class:
            sampled_frames.append(subset.copy())
        else:
            sampled_frames.append(
                subset.sample(n=per_class, random_state=random_state, replace=False)
            )

    sampled = pd.concat(sampled_frames, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return sampled


def stratified_sample(
    df: pd.DataFrame,
    target_column: str,
    max_rows: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a bounded sample that roughly preserves class proportions."""
    if len(df) <= max_rows or target_column not in df.columns:
        return df.copy()

    grouped = []
    total_rows = len(df)
    for label, subset in df.groupby(target_column, dropna=False, sort=False):
        share = len(subset) / total_rows
        n_rows = max(1, int(round(max_rows * share)))
        n_rows = min(n_rows, len(subset))
        grouped.append(subset.sample(n=n_rows, random_state=random_state, replace=False))

    sampled = pd.concat(grouped, ignore_index=True)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=random_state, replace=False)
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
