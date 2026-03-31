# -*- coding: utf-8 -*-
"""Visualization functions for BCG signal classification."""

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

HIGH_RES_DPI = 600


def _save_figure_both(fig: plt.Figure, out_stem: Path, png_dpi: int = HIGH_RES_DPI) -> None:
    """Save a figure as both high-resolution PNG and SVG."""
    out_stem = Path(out_stem)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _save_unavailable_plot(title: str, message: str, out_stem: Path) -> None:
    """Save a placeholder figure when a metric curve cannot be computed."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    _save_figure_both(fig, out_stem)


def plot_counts(counts: Dict[str, int], title: str, out_path: Path) -> None:
    """Plot and save class counts as a bar chart."""
    fig, ax = plt.subplots()
    ax.bar(list(counts.keys()), list(counts.values()))
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Number of chunks")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_stem: Path,
    class_names: Sequence[str] = ("NonBCG", "BCG"),
) -> None:
    """Plot and save confusion matrix as PNG and SVG."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names))
    disp.plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")
    _save_figure_both(fig, out_stem)


def plot_roc_curve_binary(
    y_true: np.ndarray,
    p_uncal: np.ndarray,
    out_stem: Path,
    p_cal: Optional[np.ndarray] = None,
) -> None:
    """Plot and save ROC curve; optionally overlay calibrated probabilities."""
    if np.unique(y_true).size < 2:
        _save_unavailable_plot(
            "ROC Curve",
            "ROC curve is unavailable because this fold test set contains a single class.",
            out_stem,
        )
        return

    fig, ax = plt.subplots(figsize=(6.2, 5.2))

    fpr_u, tpr_u, _ = roc_curve(y_true, p_uncal)
    auc_u = roc_auc_score(y_true, p_uncal)
    ax.plot(fpr_u, tpr_u, linewidth=2, label=f"Uncalibrated (AUC={auc_u:.3f})")

    if p_cal is not None:
        fpr_c, tpr_c, _ = roc_curve(y_true, p_cal)
        auc_c = roc_auc_score(y_true, p_cal)
        ax.plot(fpr_c, tpr_c, linewidth=2, label=f"Calibrated (AUC={auc_c:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    _save_figure_both(fig, out_stem)


def plot_precision_recall_curve_binary(
    y_true: np.ndarray,
    p_uncal: np.ndarray,
    out_stem: Path,
    p_cal: Optional[np.ndarray] = None,
) -> None:
    """Plot and save Precision-Recall curve; optionally overlay calibrated probabilities."""
    if np.unique(y_true).size < 2:
        _save_unavailable_plot(
            "Precision-Recall Curve",
            "Precision-Recall curve is unavailable because this fold test set contains a single class.",
            out_stem,
        )
        return

    prevalence = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(6.2, 5.2))

    prec_u, rec_u, _ = precision_recall_curve(y_true, p_uncal)
    ap_u = average_precision_score(y_true, p_uncal)
    ax.plot(rec_u, prec_u, linewidth=2, label=f"Uncalibrated (AP={ap_u:.3f})")

    if p_cal is not None:
        prec_c, rec_c, _ = precision_recall_curve(y_true, p_cal)
        ap_c = average_precision_score(y_true, p_cal)
        ax.plot(rec_c, prec_c, linewidth=2, label=f"Calibrated (AP={ap_c:.3f})")

    ax.hlines(
        y=prevalence,
        xmin=0.0,
        xmax=1.0,
        linestyles="--",
        linewidth=1,
        label=f"Baseline={prevalence:.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    _save_figure_both(fig, out_stem)
