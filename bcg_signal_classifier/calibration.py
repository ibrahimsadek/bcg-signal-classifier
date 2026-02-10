# -*- coding: utf-8 -*-
"""Probability calibration functions and classes for BCG signal classification."""

import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

from bcg_signal_classifier.config import Config


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities from logits (numerically stable).

    Args:
        logits: Logits array of shape (n_samples, n_classes).

    Returns:
        Softmax probabilities of shape (n_samples, n_classes).
    """
    z = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def brier_score_binary(y_true: np.ndarray, p_pos: np.ndarray) -> float:
    """Compute Brier score for binary classification.

    Args:
        y_true: True binary labels (0 or 1).
        p_pos: Predicted probabilities for positive class.

    Returns:
        Brier score (lower is better).
    """
    y_true = y_true.astype(np.float32)
    p_pos = np.clip(p_pos.astype(np.float32), 0.0, 1.0)
    return float(np.mean((p_pos - y_true) ** 2))


def expected_calibration_error_binary(y_true: np.ndarray, p_pos: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) for binary classification.

    Args:
        y_true: True binary labels (0 or 1).
        p_pos: Predicted probabilities for positive class.
        n_bins: Number of bins for calibration.

    Returns:
        ECE value (lower is better).
    """
    y_true = y_true.astype(np.int64)
    p_pos = np.clip(p_pos.astype(np.float64), 0.0, 1.0)

    pred = (p_pos >= 0.5).astype(np.int64)
    conf = np.where(pred == 1, p_pos, 1.0 - p_pos)
    correct = (pred == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = conf.shape[0]
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        acc_bin = float(np.mean(correct[mask]))
        conf_bin = float(np.mean(conf[mask]))
        ece += (np.sum(mask) / n) * abs(acc_bin - conf_bin)

    return float(ece)


def reliability_plot(y_true: np.ndarray, p_pos: np.ndarray, out_path: Path, n_bins: int = 10) -> None:
    """Generate and save a reliability diagram.

    Args:
        y_true: True binary labels (0 or 1).
        p_pos: Predicted probabilities for positive class.
        out_path: Path to save the plot.
        n_bins: Number of bins for calibration.
    """
    y_true = y_true.astype(np.int64)
    p_pos = np.clip(p_pos.astype(np.float64), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    confs, accs = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p_pos >= lo) & (p_pos < hi) if i < n_bins - 1 else (p_pos >= lo) & (p_pos <= hi)
        if not np.any(mask):
            continue
        pred = (p_pos[mask] >= 0.5).astype(np.int64)
        acc = float(np.mean(pred == y_true[mask]))
        conf = float(np.mean(np.where(pred == 1, p_pos[mask], 1.0 - p_pos[mask])))
        confs.append(conf)
        accs.append(acc)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.scatter(confs, accs)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


class TemperatureScaler:
    """Temperature scaling calibrator.

    Learns a single temperature parameter to scale logits for better
    probability calibration.

    Args:
        opt_steps: Number of optimization steps.
        opt_lr: Learning rate for optimization.
    """

    def __init__(self, opt_steps: int = 250, opt_lr: float = 0.05):
        """Initialize temperature scaler.

        Args:
            opt_steps: Number of optimization steps.
            opt_lr: Learning rate for optimization.
        """
        self.opt_steps = int(opt_steps)
        self.opt_lr = float(opt_lr)
        self.temperature_: float = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        """Fit temperature parameter.

        Args:
            logits: Model logits of shape (n_samples, n_classes).
            y_true: True labels of shape (n_samples,).

        Returns:
            Self.
        """
        logits_tf = tf.constant(logits, dtype=tf.float32)
        y_tf = tf.constant(y_true.astype(np.int64), dtype=tf.int64)

        log_T = tf.Variable(0.0, dtype=tf.float32)  # T = exp(log_T)
        opt = tf.keras.optimizers.Adam(learning_rate=self.opt_lr)

        for _ in range(self.opt_steps):
            with tf.GradientTape() as tape:
                T = tf.exp(log_T)
                scaled = logits_tf / T
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_tf, scaled, from_logits=True))
            grads = tape.gradient(loss, [log_T])
            opt.apply_gradients(zip(grads, [log_T]))

        self.temperature_ = float(tf.exp(log_T).numpy())
        self.temperature_ = float(np.clip(self.temperature_, 0.05, 50.0))
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities.

        Args:
            logits: Model logits of shape (n_samples, n_classes).

        Returns:
            Calibrated probabilities of shape (n_samples, n_classes).
        """
        return softmax_np(logits / float(self.temperature_))


class PlattScalerBinary:
    """Platt scaling calibrator for binary classification.

    Fits a logistic regression on the logit margin.
    """

    def __init__(self):
        """Initialize Platt scaler."""
        self.clf = LogisticRegression(solver="lbfgs")

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "PlattScalerBinary":
        """Fit Platt scaler.

        Args:
            logits: Model logits of shape (n_samples, 2).
            y_true: True labels of shape (n_samples,).

        Returns:
            Self.
        """
        s = (logits[:, 1] - logits[:, 0]).reshape(-1, 1)
        self.clf.fit(s, y_true.astype(np.int64))
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities.

        Args:
            logits: Model logits of shape (n_samples, 2).

        Returns:
            Calibrated probabilities of shape (n_samples, 2).
        """
        s = (logits[:, 1] - logits[:, 0]).reshape(-1, 1)
        p1 = self.clf.predict_proba(s)[:, 1]
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)


def fit_calibrator(
    cfg: Config, logits_cal: np.ndarray, y_cal: np.ndarray
) -> Optional[Union[TemperatureScaler, PlattScalerBinary]]:
    """Fit a calibrator based on configuration.

    Args:
        cfg: Configuration object.
        logits_cal: Calibration logits of shape (n_samples, n_classes).
        y_cal: Calibration labels of shape (n_samples,).

    Returns:
        Fitted calibrator or None if calibration is disabled.

    Raises:
        ValueError: If calibration method is unknown.
    """
    if cfg.calibration == "none":
        return None
    if cfg.calibration == "temperature":
        return TemperatureScaler(opt_steps=cfg.calib_opt_steps, opt_lr=cfg.calib_opt_lr).fit(logits_cal, y_cal)
    if cfg.calibration == "platt":
        return PlattScalerBinary().fit(logits_cal, y_cal)
    raise ValueError(f"Unknown calibration method: {cfg.calibration}")


def apply_calibrator(calibrator: Optional[Union[TemperatureScaler, PlattScalerBinary]], logits: np.ndarray) -> np.ndarray:
    """Apply calibrator to logits.

    Args:
        calibrator: Fitted calibrator or None.
        logits: Model logits of shape (n_samples, n_classes).

    Returns:
        Calibrated probabilities of shape (n_samples, n_classes).
    """
    if calibrator is None:
        return softmax_np(logits)
    return calibrator.predict_proba(logits)
