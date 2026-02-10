# -*- coding: utf-8 -*-
"""Tests for augmentation module."""

import numpy as np

from bcg_signal_classifier.augmentation import augment_capped_option_b


def test_augment_preserves_feature_shape():
    """Test that augmentation preserves feature dimension."""
    X = np.random.randn(20, 50).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10, dtype=np.int64)

    X_aug, y_aug, before, after = augment_capped_option_b(
        X, y, seed=42, target_len=50, max_oversample_factor=2.0,
        noise_std=0.05, scale_min=0.9, scale_max=1.1,
        max_roll_frac=0.1, downsample_majority=True
    )

    assert X_aug.shape[1] == 50, "Feature dimension should be preserved"


def test_augment_returns_correct_counts():
    """Test that augmentation returns count dictionaries."""
    X = np.random.randn(20, 50).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10, dtype=np.int64)

    X_aug, y_aug, before, after = augment_capped_option_b(
        X, y, seed=42, target_len=50, max_oversample_factor=2.0,
        noise_std=0.05, scale_min=0.9, scale_max=1.1,
        max_roll_frac=0.1, downsample_majority=True
    )

    assert "NonBCG" in before and "BCG" in before
    assert "NonBCG" in after and "BCG" in after
    assert before["NonBCG"] == 10
    assert before["BCG"] == 10


def test_augment_handles_single_class():
    """Test that augmentation handles single-class input gracefully."""
    X = np.random.randn(10, 50).astype(np.float32)
    y = np.array([1] * 10, dtype=np.int64)

    X_aug, y_aug, before, after = augment_capped_option_b(
        X, y, seed=42, target_len=50, max_oversample_factor=2.0,
        noise_std=0.05, scale_min=0.9, scale_max=1.1,
        max_roll_frac=0.1, downsample_majority=True
    )

    # Should return copy when only one class
    assert X_aug.shape[0] == X.shape[0]
    assert y_aug.shape[0] == y.shape[0]


def test_augment_balanced_output():
    """Test that augmentation balances classes."""
    X = np.random.randn(30, 50).astype(np.float32)
    y = np.array([0] * 5 + [1] * 25, dtype=np.int64)  # Imbalanced

    X_aug, y_aug, before, after = augment_capped_option_b(
        X, y, seed=42, target_len=50, max_oversample_factor=3.0,
        noise_std=0.05, scale_min=0.9, scale_max=1.1,
        max_roll_frac=0.1, downsample_majority=True
    )

    # After augmentation, classes should be more balanced
    assert after["NonBCG"] > before["NonBCG"], "Minority class should be augmented"
