# -*- coding: utf-8 -*-
"""Apply a trained BCG model to new recordings (inference).

Loads artifacts written by the training pipeline (see
:mod:`bcg_signal_classifier.persistence`), reproduces the training-time
preprocessing on a new patient CSV, and writes per-segment predictions
(label + calibrated probability) to a CSV file.

Two segmentation modes are supported:

* **Annotation mode** (``--annotations``): classify each interval listed in an
  annotation file. ``categories`` are optional; if present, accuracy is also
  reported. This mirrors the exact unit used during training.
* **Window mode** (``--window_sec`` with ``--stride_sec``): slide a fixed window
  over the whole recording, for deployment scenarios without annotations.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from bcg_signal_classifier.calibration import apply_calibrator, softmax_np
from bcg_signal_classifier.config import Config
from bcg_signal_classifier.dataset import load_data_csv
from bcg_signal_classifier.persistence import load_artifacts
from bcg_signal_classifier.preprocessing import (
    build_filters,
    extract_interval,
    filter_signal,
    min_padlen,
    resample_1d,
    zscore,
)

# Labels accepted in an annotation ``categories`` column (case-insensitive).
_BCG_LABELS = {"bcg"}
_NONBCG_LABELS = {"nonbcg", "non-bcg", "non_bcg"}


def _config_from_inference(infer_cfg: dict) -> Config:
    """Build a Config carrying the saved preprocessing parameters.

    Args:
        infer_cfg: Inference configuration dictionary from the saved artifacts.

    Returns:
        A Config instance usable with the preprocessing helpers.
    """
    return Config(
        project_dir=Path("."),
        data_dir=Path("."),
        ann_dir=Path("."),
        out_dir=Path("."),
        fs=float(infer_cfg["fs"]),
        hp_order=int(infer_cfg["hp_order"]),
        hp_ripple=float(infer_cfg["hp_ripple"]),
        hp_cutoff_hz=float(infer_cfg["hp_cutoff_hz"]),
        lp_order=int(infer_cfg["lp_order"]),
        lp_ripple=float(infer_cfg["lp_ripple"]),
        lp_cutoff_hz=float(infer_cfg["lp_cutoff_hz"]),
        target_len=int(infer_cfg["target_len"]),
    )


def _filter_full_signal(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Apply zero-phase band-pass filtering to the full raw channel.

    Args:
        df: DataFrame with ``epoch`` and ``raw_data_sleepMat`` columns.
        cfg: Config carrying filter parameters.

    Returns:
        The DataFrame with an added ``filtered`` column.

    Raises:
        ValueError: If the recording is too short for zero-phase filtering.
    """
    hp, lp = build_filters(cfg)
    min_len = max(min_padlen(*hp), min_padlen(*lp))
    raw = df["raw_data_sleepMat"].to_numpy(dtype=np.float32)
    if raw.size < min_len:
        raise ValueError(f"Recording too short ({raw.size} samples) for zero-phase filtering (need >= {min_len}).")
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    df = df.copy()
    df["filtered"] = filter_signal(raw, hp, lp)
    return df


def _label_from_category(value: object) -> Optional[int]:
    """Map an annotation category string to a class index, if recognized."""
    cat = str(value).strip().lower()
    if cat in _BCG_LABELS:
        return 1
    if cat in _NONBCG_LABELS:
        return 0
    return None


def segments_from_annotations(
    df: pd.DataFrame, ann: pd.DataFrame, cfg: Config
) -> Tuple[np.ndarray, List[dict]]:
    """Extract preprocessed segments for each annotated interval.

    Args:
        df: Patient DataFrame with a ``filtered`` column.
        ann: Annotation DataFrame with ``start_time`` and ``end_time`` columns
            (optionally ``chunk_id`` and ``categories``).
        cfg: Config carrying ``target_len``.

    Returns:
        Tuple ``(X, meta)`` where ``X`` has shape (n_segments, target_len) and
        ``meta`` is a per-segment list of dictionaries.
    """
    X_list: List[np.ndarray] = []
    meta: List[dict] = []
    has_cat = "categories" in ann.columns

    for i, r in ann.iterrows():
        start_ms, end_ms = int(r["start_time"]), int(r["end_time"])
        seg = extract_interval(df, "filtered", start_ms, end_ms)
        if seg is None or seg.size < 2:
            continue
        seg = resample_1d(zscore(seg), cfg.target_len)
        X_list.append(seg)
        entry = {
            "chunk_id": r["chunk_id"] if "chunk_id" in ann.columns else i,
            "start_time": start_ms,
            "end_time": end_ms,
        }
        if has_cat:
            entry["true_index"] = _label_from_category(r["categories"])
        meta.append(entry)

    X = np.stack(X_list, axis=0).astype(np.float32) if X_list else np.empty((0, cfg.target_len), np.float32)
    return X, meta


def segments_from_windows(
    df: pd.DataFrame, cfg: Config, window_sec: float, stride_sec: float
) -> Tuple[np.ndarray, List[dict]]:
    """Slide a fixed window over the recording and extract preprocessed segments.

    Args:
        df: Patient DataFrame with ``epoch`` and ``filtered`` columns.
        cfg: Config carrying ``fs`` and ``target_len``.
        window_sec: Window length in seconds.
        stride_sec: Hop between consecutive windows in seconds.

    Returns:
        Tuple ``(X, meta)`` as in :func:`segments_from_annotations`.

    Raises:
        ValueError: If window/stride are non-positive.
    """
    if window_sec <= 0 or stride_sec <= 0:
        raise ValueError("window_sec and stride_sec must be positive.")

    win = max(2, int(round(window_sec * cfg.fs)))
    hop = max(1, int(round(stride_sec * cfg.fs)))
    sig = df["filtered"].to_numpy(dtype=np.float32)
    epochs = df["epoch"].to_numpy()
    n = sig.size

    X_list: List[np.ndarray] = []
    meta: List[dict] = []
    idx = 0
    for start in range(0, max(1, n - win + 1), hop):
        end = start + win
        seg = sig[start:end]
        if seg.size < 2:
            continue
        seg = resample_1d(zscore(seg), cfg.target_len)
        X_list.append(seg)
        meta.append(
            {
                "chunk_id": idx,
                "start_time": int(epochs[start]),
                "end_time": int(epochs[min(end - 1, n - 1)]),
            }
        )
        idx += 1

    X = np.stack(X_list, axis=0).astype(np.float32) if X_list else np.empty((0, cfg.target_len), np.float32)
    return X, meta


def predict_dataframe(model, calibrator, infer_cfg: dict, X: np.ndarray, meta: List[dict]) -> pd.DataFrame:
    """Run the model + calibrator on extracted segments and build a results table.

    Args:
        model: Loaded Keras model outputting logits.
        calibrator: Loaded calibrator or None.
        infer_cfg: Inference configuration dictionary (for the label map).
        X: Segments of shape (n_segments, target_len).
        meta: Per-segment metadata from a segmentation function.

    Returns:
        DataFrame with predictions and probabilities, one row per segment.
    """
    label_map = {int(k): v for k, v in infer_cfg.get("label_map", {"0": "NonBCG", "1": "BCG"}).items()}

    if X.shape[0] == 0:
        return pd.DataFrame(columns=["chunk_id", "start_time", "end_time", "pred_index", "pred_label", "prob_BCG"])

    logits = model.predict(X[..., None], verbose=0)
    prob_uncal = softmax_np(logits)
    prob_cal = apply_calibrator(calibrator, logits)
    pred_idx = np.argmax(logits, axis=1)

    df = pd.DataFrame(meta)
    df["pred_index"] = pred_idx.astype(int)
    df["pred_label"] = [label_map.get(int(i), str(int(i))) for i in pred_idx]
    df["prob_BCG"] = prob_cal[:, 1]
    df["prob_BCG_uncal"] = prob_uncal[:, 1]

    if "true_index" in df.columns and df["true_index"].notna().any():
        valid = df["true_index"].notna()
        df.loc[valid, "true_label"] = [label_map.get(int(i), str(int(i))) for i in df.loc[valid, "true_index"]]
        df.loc[valid, "correct"] = df.loc[valid, "pred_index"] == df.loc[valid, "true_index"].astype("Int64")

    return df


def load_inference_annotations(path: Path) -> pd.DataFrame:
    """Load an annotation file for inference (categories optional).

    Args:
        path: Path to the annotation file (delimiter auto-detected).

    Returns:
        DataFrame sorted by start/end time.

    Raises:
        ValueError: If required time columns are missing.
    """
    df = pd.read_csv(path, sep=None, engine="python")
    required = {"start_time", "end_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df.sort_values(["start_time", "end_time"]).reset_index(drop=True)


def run_prediction(args: argparse.Namespace) -> pd.DataFrame:
    """Execute the full inference workflow from parsed CLI arguments.

    Args:
        args: Parsed arguments (see :func:`build_arg_parser`).

    Returns:
        The predictions DataFrame (also written to ``args.output``).

    Raises:
        ValueError: If neither an annotation file nor a positive window is given.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    model, calibrator, infer_cfg = load_artifacts(Path(args.model_dir))
    cfg = _config_from_inference(infer_cfg)
    logging.info("Loaded model '%s' (target_len=%d, fs=%.1f).", infer_cfg.get("model_name"), cfg.target_len, cfg.fs)

    df = load_data_csv(Path(args.input))
    df = _filter_full_signal(df, cfg)

    if args.annotations:
        ann = load_inference_annotations(Path(args.annotations))
        X, meta = segments_from_annotations(df, ann, cfg)
        logging.info("Annotation mode: %d intervals -> %d usable segments.", len(ann), X.shape[0])
    elif args.window_sec > 0:
        X, meta = segments_from_windows(df, cfg, args.window_sec, args.stride_sec)
        logging.info("Window mode: %d segments (window=%.2fs, stride=%.2fs).", X.shape[0], args.window_sec, args.stride_sec)
    else:
        raise ValueError("Provide either --annotations <file> or --window_sec > 0.")

    result = predict_dataframe(model, calibrator, infer_cfg, X, meta)
    out_path = Path(args.output)
    result.to_csv(out_path, index=False)

    if not result.empty:
        n_bcg = int((result["pred_index"] == 1).sum())
        logging.info("Predictions: %d segments | BCG=%d NonBCG=%d", len(result), n_bcg, len(result) - n_bcg)
        if "correct" in result.columns and result["correct"].notna().any():
            acc = float(result["correct"].dropna().mean())
            logging.info("Accuracy on labeled intervals: %.4f", acc)
    logging.info("Wrote predictions to: %s", out_path)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the inference CLI."""
    p = argparse.ArgumentParser(description="Apply a pretrained BCG model to a new recording.")
    p.add_argument("--predict", action="store_true", help="Run inference (dispatch flag from main.py).")
    p.add_argument("--model_dir", type=str, required=True, help="Directory of a saved model (final_model/).")
    p.add_argument("--input", type=str, required=True, help="Patient CSV with epoch + raw_data_sleepMat columns.")
    p.add_argument("--annotations", type=str, default="", help="Annotation file defining intervals to classify.")
    p.add_argument("--window_sec", type=float, default=0.0, help="Sliding-window length in seconds (annotation-free).")
    p.add_argument("--stride_sec", type=float, default=1.0, help="Sliding-window hop in seconds.")
    p.add_argument("--output", type=str, default="predictions.csv", help="Output CSV path.")
    return p


def predict_main() -> None:
    """CLI entry point for inference."""
    args = build_arg_parser().parse_args()
    run_prediction(args)


if __name__ == "__main__":
    predict_main()
