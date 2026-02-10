# -*- coding: utf-8 -*-
"""Data augmentation functions for BCG signal classification."""

from typing import Dict, Tuple

import numpy as np


def augment_capped_option_b(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    target_len: int,
    max_oversample_factor: float,
    noise_std: float,
    scale_min: float,
    scale_max: float,
    max_roll_frac: float,
    downsample_majority: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[str, int]]:
    """Apply capped augmentation with optional majority downsampling.

    Augments the minority class with noise, scaling, and roll transforms,
    optionally downsamples the majority class, and returns balanced data.

    Args:
        X: Input features, shape (n_samples, signal_length).
        y: Labels, shape (n_samples,).
        seed: Random seed for reproducibility.
        target_len: Length of each signal (for roll calculation).
        max_oversample_factor: Maximum ratio of minority samples after augmentation.
        noise_std: Standard deviation for Gaussian noise augmentation.
        scale_min: Minimum scale factor for multiplicative scaling.
        scale_max: Maximum scale factor for multiplicative scaling.
        max_roll_frac: Maximum fraction of signal length to roll.
        downsample_majority: Whether to downsample majority class.

    Returns:
        Tuple of (X_out, y_out, before_counts, after_counts) where:
            - X_out: Augmented features.
            - y_out: Augmented labels.
            - before_counts: Class counts before augmentation.
            - after_counts: Class counts after augmentation.
    """
    rng = np.random.default_rng(seed)
    before = {"NonBCG": int(np.sum(y == 0)), "BCG": int(np.sum(y == 1))}

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X.copy(), y.copy(), before, dict(before)

    maj = int(classes[np.argmax(counts)])
    minc = int(classes[np.argmin(counts)])
    n_maj = int(np.max(counts))
    n_min = int(np.min(counts))

    target = int(min(n_maj, int(np.ceil(max_oversample_factor * n_min))))

    idx_min = np.where(y == minc)[0]
    idx_maj = np.where(y == maj)[0]

    X_min = X[idx_min]
    X_maj = X[idx_maj]

    need = max(0, target - X_min.shape[0])
    max_roll = max(1, int(max_roll_frac * target_len))

    aug = []
    for i in range(need):
        base = X_min[i % X_min.shape[0]].copy()
        choice = int(rng.integers(0, 3))
        if choice == 0:
            out = base + noise_std * rng.standard_normal(base.shape[0]).astype(np.float32)
        elif choice == 1:
            s = float(rng.uniform(scale_min, scale_max))
            out = (base * s).astype(np.float32)
        else:
            shift = int(rng.integers(-max_roll, max_roll + 1))
            out = np.roll(base, shift).astype(np.float32)
        aug.append(out)

    X_min2 = np.vstack([X_min, np.stack(aug, axis=0)]) if aug else X_min

    if downsample_majority and X_maj.shape[0] > target:
        sel = rng.choice(X_maj.shape[0], size=target, replace=False)
        X_maj2 = X_maj[sel]
    else:
        X_maj2 = X_maj

    X_min2 = X_min2[:target]
    X_maj2 = X_maj2[:target]

    y_min2 = np.full((X_min2.shape[0],), minc, dtype=np.int64)
    y_maj2 = np.full((X_maj2.shape[0],), maj, dtype=np.int64)

    X_out = np.vstack([X_min2, X_maj2]).astype(np.float32)
    y_out = np.concatenate([y_min2, y_maj2]).astype(np.int64)

    perm = rng.permutation(X_out.shape[0])
    X_out = X_out[perm]
    y_out = y_out[perm]

    after = {"NonBCG": int(np.sum(y_out == 0)), "BCG": int(np.sum(y_out == 1))}
    return X_out, y_out, before, after
