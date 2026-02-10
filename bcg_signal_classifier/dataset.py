# -*- coding: utf-8 -*-
"""Dataset loading and building functions for BCG signal classification."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from bcg_signal_classifier.config import Config
from bcg_signal_classifier.preprocessing import (
    build_filters,
    extract_interval,
    filter_signal,
    min_padlen,
    resample_1d,
    zscore,
)


def list_patient_ids(data_dir: Path) -> List[str]:
    """List patient IDs from CSV files in data directory.

    Args:
        data_dir: Directory containing patient CSV files.

    Returns:
        List of patient IDs (4-digit strings).
    """
    pids: List[str] = []
    for f in sorted(data_dir.glob("*.csv")):
        if f.stem.isdigit() and len(f.stem) == 4:
            pids.append(f.stem)
    return pids


def load_data_csv(path: Path) -> pd.DataFrame:
    """Load patient data CSV file.

    Args:
        path: Path to CSV file.

    Returns:
        DataFrame with epoch and raw_data_sleepMat columns, sorted by epoch.

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(path)
    required = {"epoch", "raw_data_sleepMat"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df.sort_values("epoch").reset_index(drop=True)


def load_annotations(path: Path) -> pd.DataFrame:
    """Load annotation file (auto-detects delimiter).

    Args:
        path: Path to annotation file.

    Returns:
        DataFrame with annotation columns, sorted by start_time and end_time.

    Raises:
        ValueError: If required columns are missing.
    """
    # auto-detect delimiter
    df = pd.read_csv(path, sep=None, engine="python")
    expected = {"chunk_id", "categories", "start_time", "end_time"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df.sort_values(["start_time", "end_time"]).reset_index(drop=True)


def build_dataset(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Build complete dataset from patient data and annotations.

    Loads all patient data, applies filtering and preprocessing, extracts
    annotated intervals, and returns arrays ready for training.

    Args:
        cfg: Configuration object.

    Returns:
        Tuple of (X, y, groups, counts) where:
            - X: Features of shape (n_samples, target_len).
            - y: Labels of shape (n_samples,).
            - groups: Patient IDs of shape (n_samples,).
            - counts: Class counts dictionary.

    Raises:
        RuntimeError: If no patient files found or no chunks extracted.
    """
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
