# -*- coding: utf-8 -*-
"""Baseline / ablation experiments for the BCG classification pipeline.

This module runs *one* configurable evaluation per invocation so the proposed
design choices can be contrasted against simpler baselines requested by
reviewers:

* **Evaluation split** (``--split_mode``): ``patient`` (leakage-safe, GroupKFold)
  vs ``record`` (random StratifiedKFold that lets a patient's segments leak
  across train/test).
* **Augmentation** (``--aug_mode``): ``capped`` (the proposed strategy),
  ``plain`` (uncapped random minority oversampling), or ``none``.
* **Calibration** (``--calibration``): ``temperature`` / ``platt`` / ``none``;
  metrics are reported both uncalibrated and calibrated.

This is a deliberately flat (non-nested) cross-validation with fixed
hyperparameters, i.e. the "standard CV" baseline. The validated nested pipeline
in ``pipeline.py`` is untouched, so the manuscript's main results are unaffected;
each ablation run only appends one summary row to ``ablation_results.tsv``.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split

from bcg_signal_classifier.augmentation import augment_capped_option_b
from bcg_signal_classifier.calibration import (
    apply_calibrator,
    brier_score_binary,
    expected_calibration_error_binary,
    fit_calibrator,
    softmax_np,
)
from bcg_signal_classifier.config import Config
from bcg_signal_classifier.dataset import build_dataset
from bcg_signal_classifier.models import build_model
from bcg_signal_classifier.pipeline import set_seeds


def plain_oversample(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Random minority oversampling to match the majority count (no transforms).

    This is the common "standard oversampling" baseline: minority samples are
    duplicated (sampled with replacement) until the classes are balanced. No
    noise/scale/roll transforms and no cap are applied.

    Args:
        X: Features of shape (n_samples, signal_length).
        y: Labels of shape (n_samples,).
        seed: Random seed.

    Returns:
        Tuple ``(X_out, y_out)`` with balanced classes (shuffled).
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X.copy(), y.copy()

    n_max = int(np.max(counts))
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    for c in classes:
        idx = np.where(y == c)[0]
        if idx.size < n_max:
            extra = rng.choice(idx, size=n_max - idx.size, replace=True)
            idx = np.concatenate([idx, extra])
        X_parts.append(X[idx])
        y_parts.append(np.full((idx.size,), c, dtype=np.int64))

    X_out = np.vstack(X_parts).astype(np.float32)
    y_out = np.concatenate(y_parts).astype(np.int64)
    perm = rng.permutation(X_out.shape[0])
    return X_out[perm], y_out[perm]


def apply_augmentation(
    aug_mode: str, X: np.ndarray, y: np.ndarray, cfg: Config, hp: Dict, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Augment a training subset according to the selected strategy.

    Args:
        aug_mode: One of 'capped', 'plain', 'none'.
        X: Training features.
        y: Training labels.
        cfg: Configuration (augmentation transform parameters).
        hp: Hyperparameters (uses ``k`` and ``noise_std`` for capped).
        seed: Random seed.

    Returns:
        Tuple ``(X_aug, y_aug)``.

    Raises:
        ValueError: If ``aug_mode`` is unknown.
    """
    if aug_mode == "none":
        return X, y
    if aug_mode == "plain":
        return plain_oversample(X, y, seed)
    if aug_mode == "capped":
        X_a, y_a, _, _ = augment_capped_option_b(
            X,
            y,
            seed=seed,
            target_len=cfg.target_len,
            max_oversample_factor=hp["k"],
            noise_std=hp["noise_std"],
            scale_min=cfg.scale_min,
            scale_max=cfg.scale_max,
            max_roll_frac=cfg.max_roll_frac,
            downsample_majority=cfg.downsample_majority,
        )
        return X_a, y_a
    raise ValueError(f"Unknown aug_mode: {aug_mode}")


def _binary_metrics(y_true: np.ndarray, p_pos: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute accuracy, F1, AUC, Brier, ECE, and NLL for a binary problem."""
    p_pos = np.clip(p_pos, 1e-7, 1.0 - 1e-7)
    try:
        auc = float(roc_auc_score(y_true, p_pos))
    except ValueError:
        auc = float("nan")
    return {
        "acc": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": auc,
        "brier": float(brier_score_binary(y_true, p_pos)),
        "ece": float(expected_calibration_error_binary(y_true, p_pos)),
        "nll": float(log_loss(y_true, p_pos, labels=[0, 1])),
    }


def _make_calibration_split(
    X_tr: np.ndarray, y_tr: np.ndarray, g_tr: np.ndarray, split_mode: str, calib_frac: float, seed: int
):
    """Carve a fit/calibration split from the training fold.

    Patient mode keeps the split patient-wise (leakage-safe); record mode uses a
    stratified random split.

    Returns:
        Tuple of index arrays ``(fit_idx, cal_idx)`` relative to the training fold.
    """
    if split_mode == "patient":
        gss = GroupShuffleSplit(n_splits=1, test_size=calib_frac, random_state=seed)
        return next(gss.split(np.zeros_like(y_tr), y_tr, groups=g_tr))
    fit_idx, cal_idx = train_test_split(
        np.arange(y_tr.shape[0]), test_size=calib_frac, random_state=seed, stratify=y_tr
    )
    return fit_idx, cal_idx


def run_fold(
    model_name: str,
    cfg: Config,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    g_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    hp: Dict,
    aug_mode: str,
    split_mode: str,
    seed: int,
) -> Dict[str, float]:
    """Train on one fold and return uncalibrated (and calibrated) test metrics.

    Args:
        model_name: 'cnn' or 'transformer'.
        cfg: Configuration (carries the chosen ``calibration`` mode).
        X_tr, y_tr, g_tr: Training features, labels, groups.
        X_te, y_te: Test features and labels.
        hp: Fixed hyperparameters (lr, dropout, k, noise_std).
        aug_mode: Augmentation strategy.
        split_mode: 'patient' or 'record' (controls calibration sub-split).
        seed: Random seed.

    Returns:
        Per-fold metrics dictionary (calibrated metrics prefixed with ``cal_``).
    """
    calibrate = cfg.calibration != "none"

    if calibrate:
        fit_idx, cal_idx = _make_calibration_split(X_tr, y_tr, g_tr, split_mode, cfg.calib_frac, seed)
        X_fit, y_fit = X_tr[fit_idx], y_tr[fit_idx]
        X_cal, y_cal = X_tr[cal_idx], y_tr[cal_idx]
    else:
        X_fit, y_fit = X_tr, y_tr
        X_cal = y_cal = None

    X_fit_a, y_fit_a = apply_augmentation(aug_mode, X_fit, y_fit, cfg, hp, seed + 7)

    import tensorflow as tf

    tf.keras.backend.clear_session()
    set_seeds(seed + 13)
    model = build_model(model_name, cfg.target_len, lr=hp["lr"], dropout=hp["dropout"])
    model.fit(X_fit_a[..., None], y_fit_a, epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=0)

    logits_te = model.predict(X_te[..., None], verbose=0)
    prob_uncal = softmax_np(logits_te)
    pred = np.argmax(logits_te, axis=1)
    out = {f"uncal_{k}": v for k, v in _binary_metrics(y_te, prob_uncal[:, 1], pred).items()}

    if calibrate:
        calibrator = fit_calibrator(cfg, model.predict(X_cal[..., None], verbose=0), y_cal)
        prob_cal = apply_calibrator(calibrator, logits_te)
        cal_metrics = _binary_metrics(y_te, prob_cal[:, 1], np.argmax(prob_cal, axis=1))
        out.update({f"cal_{k}": v for k, v in cal_metrics.items()})

    return out


def run_ablation(args: argparse.Namespace) -> Dict[str, float]:
    """Run one flat-CV ablation configuration and append a summary row.

    Args:
        args: Parsed CLI arguments (see :func:`build_arg_parser`).

    Returns:
        Dictionary of mean metrics across folds.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = Config(
        project_dir=Path(args.project_dir),
        data_dir=Path(args.project_dir) / "data",
        ann_dir=Path(args.project_dir) / "annotations",
        out_dir=Path(args.out_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        calibration=args.calibration,
        seed=args.seed,
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    set_seeds(cfg.seed)
    X_all, y_all, g_all, counts = build_dataset(cfg)
    logging.info("Dataset: %d segments | counts=%s", len(y_all), counts)

    hp = {"lr": args.lr, "dropout": args.dropout, "k": args.max_oversample_factor, "noise_std": args.noise_std}

    if args.split_mode == "patient":
        splitter = GroupKFold(n_splits=args.splits)
        split_iter = splitter.split(np.zeros_like(y_all), y_all, groups=g_all)
    else:
        splitter = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=cfg.seed)
        split_iter = splitter.split(np.zeros_like(y_all), y_all)

    fold_metrics: List[Dict[str, float]] = []
    for fold, (tr_idx, te_idx) in enumerate(split_iter, start=1):
        if args.split_mode == "patient":
            train_pat, test_pat = set(g_all[tr_idx]), set(g_all[te_idx])
            if not train_pat.isdisjoint(test_pat):
                raise RuntimeError(f"LEAKAGE in ablation fold {fold}")
        m = run_fold(
            args.model,
            cfg,
            X_all[tr_idx],
            y_all[tr_idx],
            g_all[tr_idx],
            X_all[te_idx],
            y_all[te_idx],
            hp,
            args.aug_mode,
            args.split_mode,
            seed=cfg.seed + fold * 100,
        )
        fold_metrics.append(m)
        logging.info("Fold %d/%d | %s", fold, args.splits, {k: round(v, 4) for k, v in m.items()})

    keys = sorted(fold_metrics[0].keys())
    means = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in keys}
    stds = {k: float(np.std([fm[k] for fm in fold_metrics])) for k in keys}

    label = args.label or f"{args.model}_{args.split_mode}_aug-{args.aug_mode}_cal-{args.calibration}"
    _append_summary(cfg.out_dir / "ablation_results.tsv", label, args, keys, means, stds)

    logging.info("==== ABLATION '%s' (mean over %d folds) ====", label, args.splits)
    for k in keys:
        logging.info("  %-12s %.4f +/- %.4f", k, means[k], stds[k])
    return means


def _append_summary(path: Path, label: str, args, keys, means, stds) -> None:
    """Append a labeled result row (mean +/- std per metric) to a TSV file."""
    header = ["label", "model", "split_mode", "aug_mode", "calibration", "splits"] + [
        f"{k}_mean" for k in keys
    ] + [f"{k}_std" for k in keys]
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        if write_header:
            w.writerow(header)
        row = [label, args.model, args.split_mode, args.aug_mode, args.calibration, args.splits]
        row += [f"{means[k]:.6f}" for k in keys] + [f"{stds[k]:.6f}" for k in keys]
        w.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ablation CLI."""
    p = argparse.ArgumentParser(description="Run one baseline/ablation configuration (flat CV).")
    p.add_argument("--ablation", action="store_true", help="Run ablation (dispatch flag from main.py).")
    p.add_argument("--project_dir", type=str, default=".")
    p.add_argument("--out_dir", type=str, default="./ablation_output")
    p.add_argument("--model", type=str, choices=["cnn", "transformer"], default="cnn")
    p.add_argument("--splits", type=int, default=5, help="Number of (flat) CV folds.")
    p.add_argument("--split_mode", type=str, choices=["patient", "record"], default="patient")
    p.add_argument("--aug_mode", type=str, choices=["capped", "plain", "none"], default="capped")
    p.add_argument("--calibration", type=str, choices=["temperature", "platt", "none"], default="temperature")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4, help="Fixed learning rate (non-nested).")
    p.add_argument("--dropout", type=float, default=0.1, help="Fixed dropout (non-nested).")
    p.add_argument("--max_oversample_factor", type=float, default=2.0, help="Cap factor for capped augmentation.")
    p.add_argument("--noise_std", type=float, default=0.03, help="Noise std for capped augmentation.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", type=str, default="", help="Optional row label for the results TSV.")
    return p


def ablation_main() -> None:
    """CLI entry point for ablation experiments."""
    args = build_arg_parser().parse_args()
    run_ablation(args)


if __name__ == "__main__":
    ablation_main()
