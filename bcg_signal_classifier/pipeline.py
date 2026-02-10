# -*- coding: utf-8 -*-
"""
BCG vs NonBCG: Nested (patient-wise) Cross-Validation + Probability Calibration
==============================================================================

Implements (no leakage):
- Per-patient preprocessing + chunk extraction using ALL annotated intervals.
- Outer CV: GroupKFold(10) at patient level.
- Inner CV: GroupKFold(inner_splits) on OUTER-TRAIN patients for hyperparameter selection.
- Train-only capped augmentation (Option-B) inside each (inner/outer) training split only.
- Post-hoc calibration (Temperature scaling or Platt scaling) fit on a patient-wise calibration split
  drawn from OUTER-TRAIN patients only; evaluated on OUTER-TEST patients only.

Run:
  python main.py --model transformer

Notes:
- Nested CV is expensive. Start small:
  python main.py --model cnn --inner_splits 3 --inner_epochs 6 --epochs 12 --hp_max_trials 6 --disable_xai
- Temperature scaling improves probability reliability; it does NOT usually change argmax-based accuracy/F1.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
matplotlib.use("Agg")

from bcg_signal_classifier.augmentation import augment_capped_option_b  # noqa: E402
from bcg_signal_classifier.calibration import (  # noqa: E402
    apply_calibrator,
    brier_score_binary,
    expected_calibration_error_binary,
    fit_calibrator,
    reliability_plot,
    softmax_np,
)
from bcg_signal_classifier.config import Config  # noqa: E402
from bcg_signal_classifier.dataset import build_dataset  # noqa: E402
from bcg_signal_classifier.models import build_model  # noqa: E402
from bcg_signal_classifier.visualization import plot_counts  # noqa: E402
from bcg_signal_classifier.xai import integrated_gradients_1d, plot_ig_overlay  # noqa: E402


def setup_logging() -> None:
    """Configure logging with INFO level and timestamp format."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(p: Path) -> None:
    """Ensure directory exists, creating if necessary.

    Args:
        p: Directory path.
    """
    p.mkdir(parents=True, exist_ok=True)


def configure_gpu_directml_safe() -> None:
    """Configure GPU settings safely for DirectML.

    DirectML-safe config:
    - memory growth best-effort
    - float32 policy (avoid mixed_float16 warnings on DirectML)
    - optionally restrict visible device to GPU:0
    """
    gpus = tf.config.list_physical_devices("GPU")
    logging.info("TensorFlow GPUs detected: %s", gpus)

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            logging.debug("Could not set memory growth for %s: %s", gpu, e)

    try:
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("float32")
        logging.info("Mixed precision disabled (float32).")
    except Exception as e:
        logging.debug("Could not set mixed precision policy: %s", e)

    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], "GPU")
            logging.info("Using single GPU device: %s", gpus[0])
        except Exception as e:
            logging.debug("Could not set visible GPU device: %s", e)


def iter_hyperparam_grid(cfg: Config) -> List[Dict]:
    """Generate hyperparameter grid from config.

    Args:
        cfg: Configuration object.

    Returns:
        List of hyperparameter dictionaries, truncated to hp_max_trials.
    """
    grid = list(itertools.product(cfg.lrs, cfg.dropouts, cfg.max_oversample_factors, cfg.noise_stds))
    grid = grid[: cfg.hp_max_trials]
    return [{"lr": float(lr), "dropout": float(dr), "k": float(k), "noise_std": float(ns)} for lr, dr, k, ns in grid]


def train_eval_inner_fold(
    model_name: str,
    cfg: Config,
    hp: Dict,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    seed: int,
) -> float:
    """Train and evaluate model on one inner fold.

    Args:
        model_name: Model type ('cnn' or 'transformer').
        cfg: Configuration object.
        hp: Hyperparameters dictionary.
        X_tr: Training features.
        y_tr: Training labels.
        X_va: Validation features.
        y_va: Validation labels.
        seed: Random seed for this fold.

    Returns:
        F1 score on validation set.
    """
    X_aug, y_aug, _, _ = augment_capped_option_b(
        X_tr,
        y_tr,
        seed=seed,
        target_len=cfg.target_len,
        max_oversample_factor=hp["k"],
        noise_std=hp["noise_std"],
        scale_min=cfg.scale_min,
        scale_max=cfg.scale_max,
        max_roll_frac=cfg.max_roll_frac,
        downsample_majority=cfg.downsample_majority,
    )

    tf.keras.backend.clear_session()
    set_seeds(seed)

    model = build_model(model_name, cfg.target_len, lr=hp["lr"], dropout=hp["dropout"])
    model.fit(X_aug[..., None], y_aug, epochs=cfg.inner_epochs, batch_size=cfg.batch_size, verbose=0)

    logits_va = model.predict(X_va[..., None], verbose=0)
    y_pred = np.argmax(logits_va, axis=1)
    return float(f1_score(y_va, y_pred, zero_division=0))


def select_hyperparams_nested(
    model_name: str,
    cfg: Config,
    X_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    g_outer_train: np.ndarray,
    fold_seed: int,
) -> Dict:
    """Select best hyperparameters using nested cross-validation.

    Args:
        model_name: Model type ('cnn' or 'transformer').
        cfg: Configuration object.
        X_outer_train: Features from outer training fold.
        y_outer_train: Labels from outer training fold.
        g_outer_train: Patient groups from outer training fold.
        fold_seed: Random seed for this outer fold.

    Returns:
        Best hyperparameter dictionary based on mean inner F1 score.
    """
    inner = GroupKFold(n_splits=cfg.inner_splits)
    hp_grid = iter_hyperparam_grid(cfg)

    best_hp = hp_grid[0]
    best_score = -1.0

    for t, hp in enumerate(hp_grid, start=1):
        scores = []
        for inner_i, (tr_i, va_i) in enumerate(
            inner.split(np.zeros_like(y_outer_train), y_outer_train, groups=g_outer_train), start=1
        ):
            tr_pat = set(g_outer_train[tr_i].tolist())
            va_pat = set(g_outer_train[va_i].tolist())
            if not tr_pat.isdisjoint(va_pat):
                raise RuntimeError("LEAKAGE in inner GroupKFold")

            score = train_eval_inner_fold(
                model_name=model_name,
                cfg=cfg,
                hp=hp,
                X_tr=X_outer_train[tr_i],
                y_tr=y_outer_train[tr_i],
                X_va=X_outer_train[va_i],
                y_va=y_outer_train[va_i],
                seed=fold_seed + 1000 * inner_i,
            )
            scores.append(score)

        mean_score = float(np.mean(scores)) if scores else -1.0
        logging.info(
            "OuterFoldSeed=%d | HP %02d/%02d | lr=%.1e dr=%.2f k=%.2f noise=%.3f | inner_F1=%.4f",
            fold_seed,
            t,
            len(hp_grid),
            hp["lr"],
            hp["dropout"],
            hp["k"],
            hp["noise_std"],
            mean_score,
        )
        if mean_score > best_score:
            best_score = mean_score
            best_hp = hp

    logging.info("Selected HP: %s | inner_F1=%.4f", best_hp, best_score)
    return best_hp


def main() -> None:
    """Main pipeline orchestration function."""
    setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--project_dir", type=str, default=".")
    p.add_argument("--out_dir", type=str, default="./cv_output_nested")
    p.add_argument("--model", type=str, choices=["cnn", "transformer"], default="transformer")

    p.add_argument("--outer_splits", type=int, default=10)
    p.add_argument("--inner_splits", type=int, default=3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--inner_epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=128)

    p.add_argument("--lrs", type=str, default="1e-4,3e-4")
    p.add_argument("--dropouts", type=str, default="0.1,0.2")
    p.add_argument("--max_oversample_factors", type=str, default="2.0,3.0")
    p.add_argument("--noise_stds", type=str, default="0.03,0.05")
    p.add_argument("--hp_max_trials", type=int, default=12)

    p.add_argument("--calibration", type=str, choices=["temperature", "platt", "none"], default="temperature")
    p.add_argument("--calib_frac", type=float, default=0.2)
    p.add_argument("--calib_opt_steps", type=int, default=250)
    p.add_argument("--calib_opt_lr", type=float, default=0.05)

    p.add_argument("--disable_xai", action="store_true")
    p.add_argument("--xai_examples_per_fold", type=int, default=32)
    p.add_argument("--ig_steps", type=int, default=64)

    args = p.parse_args()

    def parse_floats(s: str) -> Tuple[float, ...]:
        return tuple(float(x.strip()) for x in s.split(",") if x.strip())

    cfg = Config(
        project_dir=Path(args.project_dir),
        data_dir=Path(args.project_dir) / "data",
        ann_dir=Path(args.project_dir) / "annotations",
        out_dir=Path(args.out_dir),
        outer_splits=int(args.outer_splits),
        inner_splits=int(args.inner_splits),
        epochs=int(args.epochs),
        inner_epochs=int(args.inner_epochs),
        batch_size=int(args.batch_size),
        lrs=parse_floats(args.lrs),
        dropouts=parse_floats(args.dropouts),
        max_oversample_factors=parse_floats(args.max_oversample_factors),
        noise_stds=parse_floats(args.noise_stds),
        hp_max_trials=int(args.hp_max_trials),
        calibration=str(args.calibration),
        calib_frac=float(args.calib_frac),
        calib_opt_steps=int(args.calib_opt_steps),
        calib_opt_lr=float(args.calib_opt_lr),
        disable_xai=bool(args.disable_xai),
        xai_examples_per_fold=int(args.xai_examples_per_fold),
        ig_steps=int(args.ig_steps),
    )

    ensure_dir(cfg.out_dir)
    set_seeds(cfg.seed)
    configure_gpu_directml_safe()

    X_all, y_all, g_all, overall_counts = build_dataset(cfg)
    plot_counts(overall_counts, "All extracted chunks (BEFORE augmentation)", cfg.out_dir / "counts_before_overall.png")

    outer = GroupKFold(n_splits=cfg.outer_splits)

    rows = []
    after_sum = {"NonBCG": 0, "BCG": 0}

    for fold, (train_idx, test_idx) in enumerate(outer.split(np.zeros_like(y_all), y_all, groups=g_all), start=1):
        fold_dir = cfg.out_dir / f"fold_{fold:02d}"
        ensure_dir(fold_dir)
        xai_dir = fold_dir / "xai_ig"
        ensure_dir(xai_dir)

        X_train_outer = X_all[train_idx]
        y_train_outer = y_all[train_idx]
        g_train_outer = g_all[train_idx]

        X_test = X_all[test_idx]
        y_test = y_all[test_idx]
        g_test = g_all[test_idx]

        train_pat = set(g_train_outer.tolist())
        test_pat = set(g_test.tolist())
        if not train_pat.isdisjoint(test_pat):
            raise RuntimeError(f"LEAKAGE in outer fold {fold}: {train_pat & test_pat}")

        logging.info(
            "========== OUTER FOLD %02d/%02d | test_patients=%s ==========", fold, cfg.outer_splits, sorted(test_pat)
        )

        fold_seed = cfg.seed + fold * 100

        # Nested hyperparameter selection on outer-train only
        best_hp = select_hyperparams_nested(args.model, cfg, X_train_outer, y_train_outer, g_train_outer, fold_seed)

        # Patient-wise calibration split within outer-train
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg.calib_frac, random_state=fold_seed)
        fit_rel, cal_rel = next(gss.split(np.zeros_like(y_train_outer), y_train_outer, groups=g_train_outer))

        fit_pat = set(g_train_outer[fit_rel].tolist())
        cal_pat = set(g_train_outer[cal_rel].tolist())
        if not fit_pat.isdisjoint(cal_pat):
            raise RuntimeError("LEAKAGE in calibration split")

        X_fit, y_fit = X_train_outer[fit_rel], y_train_outer[fit_rel]
        X_cal, y_cal = X_train_outer[cal_rel], y_train_outer[cal_rel]

        # Train final model on train_fit only (train-only augmentation)
        X_fit_aug, y_fit_aug, before, after = augment_capped_option_b(
            X_fit,
            y_fit,
            seed=fold_seed + 7,
            target_len=cfg.target_len,
            max_oversample_factor=best_hp["k"],
            noise_std=best_hp["noise_std"],
            scale_min=cfg.scale_min,
            scale_max=cfg.scale_max,
            max_roll_frac=cfg.max_roll_frac,
            downsample_majority=cfg.downsample_majority,
        )

        after_sum["NonBCG"] += after["NonBCG"]
        after_sum["BCG"] += after["BCG"]

        plot_counts(before, f"Fold {fold:02d} TRAIN_FIT (BEFORE aug)", fold_dir / "counts_before.png")
        plot_counts(after, f"Fold {fold:02d} TRAIN_FIT (AFTER Option-B aug)", fold_dir / "counts_after.png")

        tf.keras.backend.clear_session()
        set_seeds(fold_seed + 13)
        model = build_model(args.model, cfg.target_len, lr=best_hp["lr"], dropout=best_hp["dropout"])
        model.fit(X_fit_aug[..., None], y_fit_aug, epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=0)

        # Fit calibrator on calibration-set logits (no augmentation)
        logits_cal = model.predict(X_cal[..., None], verbose=0)
        calibrator = fit_calibrator(cfg, logits_cal, y_cal)

        if calibrator is None:
            logging.info("Calibration disabled.")
        elif cfg.calibration == "temperature":
            logging.info("Fitted temperature: T=%.4f", calibrator.temperature_)
        else:
            logging.info("Fitted Platt calibrator.")

        # Evaluate on outer-test (never used in tuning/calibration)
        logits_test = model.predict(X_test[..., None], verbose=0)
        prob_uncal = softmax_np(logits_test)
        prob_cal = apply_calibrator(calibrator, logits_test)

        y_pred = np.argmax(logits_test, axis=1)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        p_uncal = prob_uncal[:, 1]
        p_cal = prob_cal[:, 1]

        brier_u = brier_score_binary(y_test, p_uncal)
        ece_u = expected_calibration_error_binary(y_test, p_uncal, n_bins=10)
        nll_u = float(log_loss(y_test, prob_uncal, labels=[0, 1]))

        brier_c = brier_score_binary(y_test, p_cal)
        ece_c = expected_calibration_error_binary(y_test, p_cal, n_bins=10)
        nll_c = float(log_loss(y_test, prob_cal, labels=[0, 1]))

        logging.info(
            "Fold %02d | acc=%.4f prec=%.4f rec=%.4f f1=%.4f | "
            "brier_u=%.4f ece_u=%.4f nll_u=%.4f | brier_c=%.4f ece_c=%.4f nll_c=%.4f | "
            "best_hp=%s | test_patients=%s",
            fold,
            acc,
            prec,
            rec,
            f1,
            brier_u,
            ece_u,
            nll_u,
            brier_c,
            ece_c,
            nll_c,
            best_hp,
            sorted(test_pat),
        )

        # Reliability plots (probability quality)
        reliability_plot(y_test, p_uncal, fold_dir / "reliability_uncal.png")
        reliability_plot(y_test, p_cal, fold_dir / "reliability_cal.png")

        # Optional IG on held-out test samples only
        if not cfg.disable_xai and X_test.shape[0] > 0:
            rng = np.random.default_rng(fold_seed + 999)
            k = min(cfg.xai_examples_per_fold, X_test.shape[0])
            idx_sel = rng.choice(X_test.shape[0], size=k, replace=False)
            for j, idx in enumerate(idx_sel, start=1):
                x = X_test[idx]
                pred_class = int(y_pred[idx])
                ig = integrated_gradients_1d(model, x, target_class=pred_class, steps=cfg.ig_steps)
                plot_ig_overlay(
                    x,
                    ig,
                    f"Fold {fold:02d} | sample {idx} | pred={pred_class} | true={int(y_test[idx])}",
                    xai_dir / f"ig_{j:04d}_pred{pred_class}_true{int(y_test[idx])}.png",
                )

        # Save fold report
        with open(fold_dir / "fold_report.txt", "w", encoding="utf-8") as f:
            f.write(f"fold: {fold}\n")
            f.write(f"model: {args.model}\n")
            f.write(f"test_patients: {sorted(test_pat)}\n")
            f.write(f"train_patients: {sorted(train_pat)}\n")
            f.write(f"train_fit_patients: {sorted(fit_pat)}\n")
            f.write(f"calib_patients: {sorted(cal_pat)}\n")
            f.write(f"best_hp: {best_hp}\n")
            f.write(f"calibration: {cfg.calibration}\n")
            if calibrator is not None and cfg.calibration == "temperature":
                f.write(f"temperature_T: {calibrator.temperature_:.6f}\n")
            f.write(f"train_fit_counts_before: {before}\n")
            f.write(f"train_fit_counts_after: {after}\n")
            f.write(f"acc: {acc:.6f}\nprec: {prec:.6f}\nrec: {rec:.6f}\nf1: {f1:.6f}\n")
            f.write(f"confusion_matrix:\n{cm}\n")
            f.write(f"brier_uncal: {brier_u:.6f}\nece_uncal: {ece_u:.6f}\nnll_uncal: {nll_u:.6f}\n")
            f.write(f"brier_cal: {brier_c:.6f}\nece_cal: {ece_c:.6f}\nnll_cal: {nll_c:.6f}\n")

        rows.append(
            {
                "fold": fold,
                "acc": acc,
                "prec": prec,
                "rec": rec,
                "f1": f1,
                "brier_uncal": brier_u,
                "ece_uncal": ece_u,
                "nll_uncal": nll_u,
                "brier_cal": brier_c,
                "ece_cal": ece_c,
                "nll_cal": nll_c,
                "best_lr": best_hp["lr"],
                "best_dropout": best_hp["dropout"],
                "best_k": best_hp["k"],
                "best_noise_std": best_hp["noise_std"],
                "test_patients": ",".join(sorted(test_pat)),
            }
        )

    plot_counts(
        after_sum,
        "TRAIN_FIT counts AFTER augmentation (summed across folds)",
        cfg.out_dir / f"counts_after_overall_{args.model}.png",
    )

    df = pd.DataFrame(rows)
    df.to_csv(cfg.out_dir / f"cv_summary_nested_{args.model}.tsv", sep="\t", index=False)

    cols = ["acc", "prec", "rec", "f1", "brier_uncal", "ece_uncal", "nll_uncal", "brier_cal", "ece_cal", "nll_cal"]
    means = {c: float(df[c].mean()) for c in cols}
    stds = {c: float(df[c].std()) for c in cols}

    with open(cfg.out_dir / f"cv_summary_nested_{args.model}_stats.txt", "w", encoding="utf-8") as f:
        f.write("MEAN:\n")
        for c in cols:
            f.write(f"{c}: {means[c]:.6f}\n")
        f.write("\nSTD:\n")
        for c in cols:
            f.write(f"{c}: {stds[c]:.6f}\n")

    logging.info("Saved summary TSV and stats TXT into: %s", cfg.out_dir)
    logging.info("Done.")
