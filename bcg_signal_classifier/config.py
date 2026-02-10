# -*- coding: utf-8 -*-
"""Configuration dataclass for BCG signal classification pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class Config:
    """Configuration for BCG signal classification pipeline.

    Args:
        project_dir: Root directory of the project.
        data_dir: Directory containing patient CSV files.
        ann_dir: Directory containing annotation files.
        out_dir: Output directory for results.
        fs: Sampling rate in Hz.
        hp_order: High-pass filter order.
        hp_ripple: High-pass filter ripple in dB.
        hp_cutoff_hz: High-pass filter cutoff frequency in Hz.
        lp_order: Low-pass filter order.
        lp_ripple: Low-pass filter ripple in dB.
        lp_cutoff_hz: Low-pass filter cutoff frequency in Hz.
        target_len: Target length for resampled signals.
        outer_splits: Number of outer CV folds.
        inner_splits: Number of inner CV folds.
        seed: Random seed for reproducibility.
        epochs: Number of training epochs for outer CV.
        inner_epochs: Number of training epochs for inner CV.
        batch_size: Batch size for training.
        lrs: Learning rate candidates for hyperparameter search.
        dropouts: Dropout rate candidates for hyperparameter search.
        max_oversample_factors: Oversampling factor candidates.
        noise_stds: Noise standard deviation candidates for augmentation.
        hp_max_trials: Maximum number of hyperparameter trials.
        downsample_majority: Whether to downsample majority class in augmentation.
        scale_min: Minimum scale factor for augmentation.
        scale_max: Maximum scale factor for augmentation.
        max_roll_frac: Maximum roll fraction for augmentation.
        calibration: Calibration method (temperature, platt, or none).
        calib_frac: Fraction of training data to use for calibration.
        calib_opt_steps: Optimization steps for temperature scaling.
        calib_opt_lr: Learning rate for temperature scaling optimization.
        disable_xai: Whether to disable explainability analysis.
        xai_examples_per_fold: Number of examples for XAI per fold.
        ig_steps: Number of steps for Integrated Gradients.
    """

    project_dir: Path
    data_dir: Path
    ann_dir: Path
    out_dir: Path

    # Signal filtering
    fs: float = 50.0
    hp_order: int = 2
    hp_ripple: float = 0.5
    hp_cutoff_hz: float = 2.5
    lp_order: int = 4
    lp_ripple: float = 0.5
    lp_cutoff_hz: float = 5.0

    target_len: int = 50

    # CV
    outer_splits: int = 10
    inner_splits: int = 3
    seed: int = 42

    # Training
    epochs: int = 20
    inner_epochs: int = 8
    batch_size: int = 128

    # Hyperparameter search space
    lrs: Tuple[float, ...] = (1e-4, 3e-4)
    dropouts: Tuple[float, ...] = (0.1, 0.2)
    max_oversample_factors: Tuple[float, ...] = (2.0, 3.0)
    noise_stds: Tuple[float, ...] = (0.03, 0.05)
    hp_max_trials: int = 12

    # Augmentation transforms
    downsample_majority: bool = True
    scale_min: float = 0.90
    scale_max: float = 1.10
    max_roll_frac: float = 0.10

    # Calibration
    calibration: str = "temperature"  # temperature | platt | none
    calib_frac: float = 0.2
    calib_opt_steps: int = 250
    calib_opt_lr: float = 0.05

    # XAI
    disable_xai: bool = False
    xai_examples_per_fold: int = 32
    ig_steps: int = 64
