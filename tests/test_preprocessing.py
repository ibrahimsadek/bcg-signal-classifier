# -*- coding: utf-8 -*-
"""Tests for preprocessing module."""

import numpy as np
import pytest

from bcg_signal_classifier.preprocessing import zscore, resample_1d


def test_zscore_produces_zero_mean():
    """Test that zscore produces zero mean."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    result = zscore(x)
    assert np.abs(np.mean(result)) < 1e-6, "Mean should be approximately zero"


def test_zscore_produces_unit_std():
    """Test that zscore produces unit standard deviation."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    result = zscore(x)
    assert np.abs(np.std(result) - 1.0) < 1e-6, "Std should be approximately 1.0"


def test_resample_1d_identity():
    """Test that resampling to same length returns similar array."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    result = resample_1d(x, target_len=5)
    np.testing.assert_array_almost_equal(result, x, decimal=5)


def test_resample_1d_changes_length():
    """Test that resampling changes length correctly."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    result = resample_1d(x, target_len=10)
    assert result.shape[0] == 10, "Resampled array should have target length"


def test_resample_1d_downsampling():
    """Test downsampling."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    result = resample_1d(x, target_len=3)
    assert result.shape[0] == 3, "Downsampled array should have target length"
