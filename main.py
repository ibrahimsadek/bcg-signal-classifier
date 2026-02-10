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

Run (Windows):
  cd /d D:\Ibrahim\bcgProject
  set TF_CPP_MIN_LOG_LEVEL=2
  python bcg_cv_pipeline_nested_calibrated.py --model transformer

Notes:
- Nested CV is expensive. Start small:
  python bcg_cv_pipeline_nested_calibrated.py --model cnn --inner_splits 3 --inner_epochs 6 --epochs 12 --hp_max_trials 6 --disable_xai
- Temperature scaling improves probability reliability; it does NOT usually change argmax-based accuracy/F1.
"""

from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import cheby1, filtfilt

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, log_loss
)
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras import layers, models


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
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


# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def configure_gpu_directml_safe() -> None:
    """
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
        except Exception:
            pass

    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("float32")
        logging.info("Mixed precision disabled (float32).")
    except Exception:
        pass

    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], "GPU")
            logging.info("Using single GPU device: %s", gpus[0])
        except Exception:
            pass


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def list_patient_ids(data_dir: Path) -> List[str]:
    pids: List[str] = []
    for f in sorted(data_dir.glob("*.csv")):
        if f.stem.isdigit() and len(f.stem) == 4:
            pids.append(f.stem)
    return pids


def load_data_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"epoch", "raw_data_sleepMat"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df.sort_values("epoch").reset_index(drop=True)


def load_annotations(path: Path) -> pd.DataFrame:
    # auto-detect delimiter
    df = pd.read_csv(path, sep=None, engine="python")
    expected = {"chunk_id", "categories", "start_time", "end_time"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df.sort_values(["start_time", "end_time"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Signal processing
# -----------------------------------------------------------------------------

def build_filters(cfg: Config):
    nyq = cfg.fs / 2.0
    b_hp, a_hp = cheby1(cfg.hp_order, cfg.hp_ripple, cfg.hp_cutoff_hz / nyq, btype="high")
    b_lp, a_lp = cheby1(cfg.lp_order, cfg.lp_ripple, cfg.lp_cutoff_hz / nyq, btype="low")
    return (b_hp, a_hp), (b_lp, a_lp)


def min_padlen(b: np.ndarray, a: np.ndarray) -> int:
    return 3 * max(len(b), len(a))


def filter_signal(x: np.ndarray, hp, lp) -> np.ndarray:
    b_hp, a_hp = hp
    b_lp, a_lp = lp
    y = filtfilt(b_hp, a_hp, x)
    y = filtfilt(b_lp, a_lp, y)
    return y


def extract_interval(df: pd.DataFrame, col: str, start_ms: int, end_ms: int) -> Optional[np.ndarray]:
    seg = df.loc[(df["epoch"] >= start_ms) & (df["epoch"] < end_ms), col].to_numpy()
    if seg.size == 0:
        return None
    return seg.astype(np.float32)


def zscore(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x)) + 1e-8
    return (x - mu) / sd


def resample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.size == target_len:
        return x.astype(np.float32)
    src = np.linspace(0.0, 1.0, num=x.size, dtype=np.float32)
    tgt = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(tgt, src, x).astype(np.float32)


# -----------------------------------------------------------------------------
# Augmentation: Option-B capped augmentation (TRAIN ONLY)
# -----------------------------------------------------------------------------

def augment_capped_option_b(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    target_len: int,
    max_oversample_factor: float,
    noise_std: float,
    scale_min: float,
    scale_max: float,
    max_roll_frac: float,
    downsample_majority: bool,
):
    rng = np.random.default_rng(seed)
    before = {"NonBCG": int(np.sum(y == 0)), "BCG": int(np.sum(y == 1))}

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X.copy(), y.copy(), before, dict(before)

    maj = int(classes[np.argmax(counts)])
    minc = int(classes[np.argmin(counts)])
    n_maj = int(np.max(counts))
    n_min = int(np.min(counts))

    target = int(min(n_maj, int(np.ceil(max_oversample_factor * n_min))))

    idx_min = np.where(y == minc)[0]
    idx_maj = np.where(y == maj)[0]

    X_min = X[idx_min]
    X_maj = X[idx_maj]

    need = max(0, target - X_min.shape[0])
    max_roll = max(1, int(max_roll_frac * target_len))

    aug = []
    for i in range(need):
        base = X_min[i % X_min.shape[0]].copy()
        choice = int(rng.integers(0, 3))
        if choice == 0:
            out = base + noise_std * rng.standard_normal(base.shape[0]).astype(np.float32)
        elif choice == 1:
            s = float(rng.uniform(scale_min, scale_max))
            out = (base * s).astype(np.float32)
        else:
            shift = int(rng.integers(-max_roll, max_roll + 1))
            out = np.roll(base, shift).astype(np.float32)
        aug.append(out)

    X_min2 = np.vstack([X_min, np.stack(aug, axis=0)]) if aug else X_min

    if downsample_majority and X_maj.shape[0] > target:
        sel = rng.choice(X_maj.shape[0], size=target, replace=False)
        X_maj2 = X_maj[sel]
    else:
        X_maj2 = X_maj

    X_min2 = X_min2[:target]
    X_maj2 = X_maj2[:target]

    y_min2 = np.full((X_min2.shape[0],), minc, dtype=np.int64)
    y_maj2 = np.full((X_maj2.shape[0],), maj, dtype=np.int64)

    X_out = np.vstack([X_min2, X_maj2]).astype(np.float32)
    y_out = np.concatenate([y_min2, y_maj2]).astype(np.int64)

    perm = rng.permutation(X_out.shape[0])
    X_out = X_out[perm]
    y_out = y_out[perm]

    after = {"NonBCG": int(np.sum(y_out == 0)), "BCG": int(np.sum(y_out == 1))}
    return X_out, y_out, before, after


# -----------------------------------------------------------------------------
# Calibration metrics
# -----------------------------------------------------------------------------

def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def brier_score_binary(y_true: np.ndarray, p_pos: np.ndarray) -> float:
    y_true = y_true.astype(np.float32)
    p_pos = np.clip(p_pos.astype(np.float32), 0.0, 1.0)
    return float(np.mean((p_pos - y_true) ** 2))


def expected_calibration_error_binary(y_true: np.ndarray, p_pos: np.ndarray, n_bins: int = 10) -> float:
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
        confs.append(conf); accs.append(acc)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.scatter(confs, accs)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Calibrators
# -----------------------------------------------------------------------------

class TemperatureScaler:
    def __init__(self, opt_steps: int = 250, opt_lr: float = 0.05):
        self.opt_steps = int(opt_steps)
        self.opt_lr = float(opt_lr)
        self.temperature_: float = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
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
        return softmax_np(logits / float(self.temperature_))


class PlattScalerBinary:
    def __init__(self):
        self.clf = LogisticRegression(solver="lbfgs")

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "PlattScalerBinary":
        s = (logits[:, 1] - logits[:, 0]).reshape(-1, 1)
        self.clf.fit(s, y_true.astype(np.int64))
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        s = (logits[:, 1] - logits[:, 0]).reshape(-1, 1)
        p1 = self.clf.predict_proba(s)[:, 1]
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)


def fit_calibrator(cfg: Config, logits_cal: np.ndarray, y_cal: np.ndarray):
    if cfg.calibration == "none":
        return None
    if cfg.calibration == "temperature":
        return TemperatureScaler(opt_steps=cfg.calib_opt_steps, opt_lr=cfg.calib_opt_lr).fit(logits_cal, y_cal)
    if cfg.calibration == "platt":
        return PlattScalerBinary().fit(logits_cal, y_cal)
    raise ValueError(f"Unknown calibration method: {cfg.calibration}")


def apply_calibrator(calibrator, logits: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return softmax_np(logits)
    return calibrator.predict_proba(logits)


# -----------------------------------------------------------------------------
# Plot counts
# -----------------------------------------------------------------------------

def plot_counts(counts: Dict[str, int], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.bar(list(counts.keys()), list(counts.values()))
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Number of chunks")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Models (logits)
# -----------------------------------------------------------------------------

def build_cnn(input_len: int, lr: float, dropout: float) -> tf.keras.Model:
    inp = layers.Input(shape=(input_len, 1))
    x = layers.Conv1D(64, 7, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    logits = layers.Dense(2, activation=None, dtype="float32")(x)

    m = models.Model(inp, logits, name="cnn_logits")
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return m


def transformer_block(x: tf.Tensor, num_heads: int, key_dim: int, ff_dim: int, dropout: float) -> tf.Tensor:
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(x.shape[-1])(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x


def build_transformer(input_len: int, lr: float, dropout: float) -> tf.keras.Model:
    inp = layers.Input(shape=(input_len, 1))

    x = layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)

    pos = tf.range(start=0, limit=input_len, delta=1)
    pos_emb = layers.Embedding(input_dim=input_len, output_dim=64)(pos)
    x = x + pos_emb

    x = transformer_block(x, num_heads=4, key_dim=16, ff_dim=128, dropout=dropout)
    x = transformer_block(x, num_heads=4, key_dim=16, ff_dim=128, dropout=dropout)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    logits = layers.Dense(2, activation=None, dtype="float32")(x)

    m = models.Model(inp, logits, name="conv_transformer_logits")
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return m


def build_model(model_name: str, input_len: int, lr: float, dropout: float) -> tf.keras.Model:
    if model_name == "cnn":
        return build_cnn(input_len, lr=lr, dropout=dropout)
    if model_name == "transformer":
        return build_transformer(input_len, lr=lr, dropout=dropout)
    raise ValueError(f"Unknown model: {model_name}")


# -----------------------------------------------------------------------------
# Optional XAI (Integrated Gradients)
# -----------------------------------------------------------------------------

def ig_grad(model: tf.keras.Model, x: tf.Tensor, target_class: int) -> np.ndarray:
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
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x, label="signal")
    ign = ig / (np.max(np.abs(ig)) + 1e-8)
    ax.fill_between(np.arange(len(ign)), 0, ign, alpha=0.35, label="IG (norm)")
    ax.set_title(title, fontsize=7)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Hyperparameter grid + nested selection
# -----------------------------------------------------------------------------

def iter_hyperparam_grid(cfg: Config) -> List[Dict]:
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
    X_aug, y_aug, _, _ = augment_capped_option_b(
        X_tr, y_tr,
        seed=seed,
        target_len=cfg.target_len,
        max_oversample_factor=hp["k"],
        noise_std=hp["noise_std"],
        scale_min=cfg.scale_min, scale_max=cfg.scale_max,
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
    inner = GroupKFold(n_splits=cfg.inner_splits)
    hp_grid = iter_hyperparam_grid(cfg)

    best_hp = hp_grid[0]
    best_score = -1.0

    for t, hp in enumerate(hp_grid, start=1):
        scores = []
        for inner_i, (tr_i, va_i) in enumerate(inner.split(np.zeros_like(y_outer_train), y_outer_train, groups=g_outer_train), start=1):
            tr_pat = set(g_outer_train[tr_i].tolist())
            va_pat = set(g_outer_train[va_i].tolist())
            if not tr_pat.isdisjoint(va_pat):
                raise RuntimeError("LEAKAGE in inner GroupKFold")

            score = train_eval_inner_fold(
                model_name=model_name,
                cfg=cfg,
                hp=hp,
                X_tr=X_outer_train[tr_i], y_tr=y_outer_train[tr_i],
                X_va=X_outer_train[va_i], y_va=y_outer_train[va_i],
                seed=fold_seed + 1000 * inner_i,
            )
            scores.append(score)

        mean_score = float(np.mean(scores)) if scores else -1.0
        logging.info(
            "OuterFoldSeed=%d | HP %02d/%02d | lr=%.1e dr=%.2f k=%.2f noise=%.3f | inner_F1=%.4f",
            fold_seed, t, len(hp_grid), hp["lr"], hp["dropout"], hp["k"], hp["noise_std"], mean_score
        )
        if mean_score > best_score:
            best_score = mean_score
            best_hp = hp

    logging.info("Selected HP: %s | inner_F1=%.4f", best_hp, best_score)
    return best_hp


# -----------------------------------------------------------------------------
# Dataset creation (use ALL annotated intervals)
# -----------------------------------------------------------------------------

def build_dataset(cfg: Config):
    hp, lp = build_filters(cfg)
    min_len = max(min_padlen(*hp), min_padlen(*lp))

    pids = list_patient_ids(cfg.data_dir)
    if not pids:
        raise RuntimeError(f"No patient CSV files found in: {cfg.data_dir}")

    X_list, y_list, g_list = [], [], []
    total_ann = 0
    used = 0
    empty = 0
    short = 0

    for pid in pids:
        data_path = cfg.data_dir / f"{pid}.csv"
        ann_path = cfg.ann_dir / f"patient__{pid}__annotations.txt"
        if not ann_path.exists():
            logging.warning("Missing annotations: %s", ann_path)
            continue

        df = load_data_csv(data_path)
        ann = load_annotations(ann_path)
        total_ann += len(ann)

        raw = df["raw_data_sleepMat"].to_numpy(dtype=np.float32)
        if raw.size < min_len:
            logging.warning("Patient %s too short for filtfilt. Skipping.", pid)
            continue

        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        df["filtered"] = filter_signal(raw, hp, lp)

        for _, r in ann.iterrows():
            cat = str(r["categories"]).strip().lower()
            if cat == "bcg":
                label = 1
            elif cat in {"nonbcg", "non-bcg", "non_bcg"}:
                label = 0
            else:
                continue

            seg = extract_interval(df, "filtered", int(r["start_time"]), int(r["end_time"]))
            if seg is None:
                empty += 1
                continue
            if seg.size < 2:
                short += 1
                continue

            seg = zscore(seg)
            seg = resample_1d(seg, cfg.target_len)

            X_list.append(seg)
            y_list.append(label)
            g_list.append(pid)
            used += 1

    if not X_list:
        raise RuntimeError("No chunks extracted. Check timestamps alignment.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    g = np.array(g_list)

    logging.info("Annotation rows (all): %d", total_ann)
    logging.info("Chunks extracted (BCG/NonBCG used): %d", used)
    logging.info("Empty intervals: %d", empty)
    logging.info("Intervals with <2 samples: %d", short)

    counts = {"NonBCG": int(np.sum(y == 0)), "BCG": int(np.sum(y == 1))}
    return X, y, g, counts


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--project_dir", type=str, default=r"D:\Ibrahim\bcgProject")
    p.add_argument("--out_dir", type=str, default=r"D:\Ibrahim\bcgProject\cv_output_nested")
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

        logging.info("========== OUTER FOLD %02d/%02d | test_patients=%s ==========",
                     fold, cfg.outer_splits, sorted(test_pat))

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
            X_fit, y_fit,
            seed=fold_seed + 7,
            target_len=cfg.target_len,
            max_oversample_factor=best_hp["k"],
            noise_std=best_hp["noise_std"],
            scale_min=cfg.scale_min, scale_max=cfg.scale_max,
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
            fold, acc, prec, rec, f1,
            brier_u, ece_u, nll_u,
            brier_c, ece_c, nll_c,
            best_hp, sorted(test_pat)
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
                    x, ig,
                    f"Fold {fold:02d} | sample {idx} | pred={pred_class} | true={int(y_test[idx])}",
                    xai_dir / f"ig_{j:04d}_pred{pred_class}_true{int(y_test[idx])}.png"
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

        rows.append({
            "fold": fold,
            "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "brier_uncal": brier_u, "ece_uncal": ece_u, "nll_uncal": nll_u,
            "brier_cal": brier_c, "ece_cal": ece_c, "nll_cal": nll_c,
            "best_lr": best_hp["lr"],
            "best_dropout": best_hp["dropout"],
            "best_k": best_hp["k"],
            "best_noise_std": best_hp["noise_std"],
            "test_patients": ",".join(sorted(test_pat)),
        })

    plot_counts(after_sum, "TRAIN_FIT counts AFTER augmentation (summed across folds)", cfg.out_dir / f"counts_after_overall_{args.model}.png")

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


if __name__ == "__main__":
    main()
