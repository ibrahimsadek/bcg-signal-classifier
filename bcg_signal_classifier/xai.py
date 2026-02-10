# -*- coding: utf-8 -*-
"""Explainable AI (XAI) functions using Integrated Gradients."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def ig_grad(model: tf.keras.Model, x: tf.Tensor, target_class: int) -> np.ndarray:
    """Compute gradient of model output w.r.t. input for a target class.

    Args:
        model: Keras model outputting logits.
        x: Input tensor of shape (1, signal_length, 1).
        target_class: Target class index.

    Returns:
        Gradient array of shape (signal_length,).
    """
    x = tf.cast(x, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        logits = model(x, training=False)
        probs = tf.nn.softmax(logits, axis=-1)
        prob = probs[0, target_class]
    g = tape.gradient(prob, x)
    if g is None:
        return np.zeros((x.shape[1],), dtype=np.float32)
    return g.numpy().reshape(-1).astype(np.float32)


def integrated_gradients_1d(model: tf.keras.Model, x: np.ndarray, target_class: int, steps: int) -> np.ndarray:
    """Compute Integrated Gradients for a 1D signal.

    Args:
        model: Keras model outputting logits.
        x: Input signal of shape (signal_length,).
        target_class: Target class index.
        steps: Number of integration steps.

    Returns:
        Integrated Gradients attribution of shape (signal_length,).
    """
    x0 = np.zeros_like(x, dtype=np.float32)
    alphas = np.linspace(0.0, 1.0, num=steps, dtype=np.float32)

    total = np.zeros_like(x, dtype=np.float32)
    for a in alphas:
        xi = x0 + a * (x - x0)
        total += ig_grad(model, tf.constant(xi.reshape(1, -1, 1)), target_class)

    avg = total / float(steps)
    ig = (x - x0) * avg
    return np.nan_to_num(ig, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def plot_ig_overlay(x: np.ndarray, ig: np.ndarray, title: str, out_path: Path) -> None:
    """Plot signal with Integrated Gradients overlay and save.

    Args:
        x: Input signal of shape (signal_length,).
        ig: Integrated Gradients attribution of shape (signal_length,).
        title: Plot title.
        out_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x, label="signal")
    ign = ig / (np.max(np.abs(ig)) + 1e-8)
    ax.fill_between(np.arange(len(ign)), 0, ign, alpha=0.35, label="IG (norm)")
    ax.set_title(title, fontsize=7)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
