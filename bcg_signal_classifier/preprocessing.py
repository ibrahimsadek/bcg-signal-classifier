# -*- coding: utf-8 -*-
"""Signal preprocessing functions for BCG signal classification."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import cheby1, filtfilt

from bcg_signal_classifier.config import Config


def build_filters(cfg: Config) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Build high-pass and low-pass Chebyshev filters.

    Args:
        cfg: Configuration object containing filter parameters.

    Returns:
        A tuple of ((b_hp, a_hp), (b_lp, a_lp)) filter coefficients.
    """
    nyq = cfg.fs / 2.0
    b_hp, a_hp = cheby1(cfg.hp_order, cfg.hp_ripple, cfg.hp_cutoff_hz / nyq, btype="high")
    b_lp, a_lp = cheby1(cfg.lp_order, cfg.lp_ripple, cfg.lp_cutoff_hz / nyq, btype="low")
    return (b_hp, a_hp), (b_lp, a_lp)


def min_padlen(b: np.ndarray, a: np.ndarray) -> int:
    """Calculate minimum pad length for filtfilt.

    Args:
        b: Filter numerator coefficients.
        a: Filter denominator coefficients.

    Returns:
        Minimum pad length required for filtfilt.
    """
    return 3 * max(len(b), len(a))


def filter_signal(x: np.ndarray, hp: Tuple[np.ndarray, np.ndarray], lp: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Apply high-pass and low-pass filters to signal.

    Args:
        x: Input signal.
        hp: High-pass filter coefficients (b, a).
        lp: Low-pass filter coefficients (b, a).

    Returns:
        Filtered signal.
    """
    b_hp, a_hp = hp
    b_lp, a_lp = lp
    y = filtfilt(b_hp, a_hp, x)
    y = filtfilt(b_lp, a_lp, y)
    return y


def extract_interval(df: pd.DataFrame, col: str, start_ms: int, end_ms: int) -> Optional[np.ndarray]:
    """Extract signal segment from a time interval.

    Args:
        df: DataFrame containing epoch timestamps and signal data.
        col: Column name containing signal values.
        start_ms: Start time in milliseconds.
        end_ms: End time in milliseconds.

    Returns:
        Signal segment as float32 array, or None if no samples in interval.
    """
    seg = df.loc[(df["epoch"] >= start_ms) & (df["epoch"] < end_ms), col].to_numpy()
    if seg.size == 0:
        return None
    return seg.astype(np.float32)


def zscore(x: np.ndarray) -> np.ndarray:
    """Z-score normalize a signal.

    Args:
        x: Input signal.

    Returns:
        Z-score normalized signal (zero mean, unit std).
    """
    mu = float(np.mean(x))
    sd = float(np.std(x)) + 1e-8
    return (x - mu) / sd


def resample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    """Resample 1D signal to target length using linear interpolation.

    Args:
        x: Input signal.
        target_len: Target length for resampled signal.

    Returns:
        Resampled signal of length target_len.
    """
    if x.size == target_len:
        return x.astype(np.float32)
    src = np.linspace(0.0, 1.0, num=x.size, dtype=np.float32)
    tgt = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(tgt, src, x).astype(np.float32)
