# -*- coding: utf-8 -*-
"""Tests for calibration module."""

import numpy as np

from bcg_signal_classifier.calibration import (
    softmax_np,
    brier_score_binary,
    expected_calibration_error_binary,
    TemperatureScaler,
    PlattScalerBinary,
)


def test_softmax_sums_to_one():
    """Test that softmax outputs sum to 1."""
    logits = np.random.randn(10, 2).astype(np.float32)
    probs = softmax_np(logits)
    sums = np.sum(probs, axis=1)
    np.testing.assert_array_almost_equal(sums, np.ones(10), decimal=5)


def test_softmax_handles_large_values():
    """Test that softmax handles large values without overflow."""
    logits = np.array([[1000.0, 0.0], [0.0, 1000.0]], dtype=np.float32)
    probs = softmax_np(logits)
    assert np.all(np.isfinite(probs)), "Softmax should handle large values"
    np.testing.assert_array_almost_equal(probs[0], [1.0, 0.0], decimal=5)
    np.testing.assert_array_almost_equal(probs[1], [0.0, 1.0], decimal=5)


def test_brier_score_perfect_predictions():
    """Test that Brier score is 0 for perfect predictions."""
    y_true = np.array([0, 0, 1, 1])
    p_pos = np.array([0.0, 0.0, 1.0, 1.0])
    score = brier_score_binary(y_true, p_pos)
    assert score == 0.0, "Brier score should be 0 for perfect predictions"


def test_brier_score_worst_predictions():
    """Test that Brier score is high for worst predictions."""
    y_true = np.array([0, 0, 1, 1])
    p_pos = np.array([1.0, 1.0, 0.0, 0.0])
    score = brier_score_binary(y_true, p_pos)
    assert score == 1.0, "Brier score should be 1 for worst predictions"


def test_ece_perfect_calibration():
    """Test that ECE is 0 for perfectly calibrated predictions."""
    y_true = np.array([0, 0, 1, 1])
    p_pos = np.array([0.0, 0.0, 1.0, 1.0])
    ece = expected_calibration_error_binary(y_true, p_pos, n_bins=10)
    assert ece < 0.01, "ECE should be near 0 for perfectly calibrated predictions"


def test_temperature_scaler_fit_predict():
    """Test TemperatureScaler basic fit and predict."""
    logits = np.random.randn(50, 2).astype(np.float32)
    y_true = np.random.randint(0, 2, size=50)

    scaler = TemperatureScaler(opt_steps=10, opt_lr=0.05)
    scaler.fit(logits, y_true)

    # Check that temperature was fitted
    assert hasattr(scaler, 'temperature_')
    assert 0.05 <= scaler.temperature_ <= 50.0

    # Check that predict_proba works
    probs = scaler.predict_proba(logits)
    assert probs.shape == (50, 2)
    np.testing.assert_array_almost_equal(np.sum(probs, axis=1), np.ones(50), decimal=5)


def test_platt_scaler_fit_predict():
    """Test PlattScalerBinary basic fit and predict."""
    logits = np.random.randn(50, 2).astype(np.float32)
    y_true = np.random.randint(0, 2, size=50)

    scaler = PlattScalerBinary()
    scaler.fit(logits, y_true)

    # Check that predict_proba works
    probs = scaler.predict_proba(logits)
    assert probs.shape == (50, 2)
    np.testing.assert_array_almost_equal(np.sum(probs, axis=1), np.ones(50), decimal=5)
