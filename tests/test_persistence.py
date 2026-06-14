# -*- coding: utf-8 -*-
"""Tests for persistence (calibrator (de)serialization round-trips)."""

import numpy as np

from bcg_signal_classifier.calibration import PlattScalerBinary, TemperatureScaler
from bcg_signal_classifier.persistence import calibrator_from_dict, calibrator_to_dict


def test_none_calibrator_round_trip():
    """A disabled calibrator serializes to type 'none' and back to None."""
    d = calibrator_to_dict(None)
    assert d == {"type": "none"}
    assert calibrator_from_dict(d) is None


def test_temperature_calibrator_round_trip():
    """Temperature scaling produces identical probabilities after reload."""
    logits = np.random.randn(40, 2).astype(np.float32)
    y_true = np.random.randint(0, 2, size=40)

    scaler = TemperatureScaler(opt_steps=10, opt_lr=0.05).fit(logits, y_true)
    restored = calibrator_from_dict(calibrator_to_dict(scaler))

    assert isinstance(restored, TemperatureScaler)
    assert restored.temperature_ == scaler.temperature_
    np.testing.assert_array_almost_equal(
        restored.predict_proba(logits), scaler.predict_proba(logits), decimal=6
    )


def test_platt_calibrator_round_trip():
    """Platt scaling produces identical probabilities after reload (no refit)."""
    logits = np.random.randn(60, 2).astype(np.float32)
    y_true = np.random.randint(0, 2, size=60)

    scaler = PlattScalerBinary().fit(logits, y_true)
    restored = calibrator_from_dict(calibrator_to_dict(scaler))

    assert isinstance(restored, PlattScalerBinary)
    np.testing.assert_array_almost_equal(
        restored.predict_proba(logits), scaler.predict_proba(logits), decimal=6
    )
