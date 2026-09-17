# Self-contained BCG/ECG HR pipeline with adaptive per-block motion artifact gating (8 Hz cutoff + relative outlier detection).
# Co-authored with CoCo
"""Self-contained BCG/ECG heart-rate pipeline (inference + analysis).

Merges model-gated 30-second HR inference with downstream cross-patient analysis.
Primary BCG estimate is the per-second diagnostic J-J estimator; autocorrelation
and dense peak counting are fallbacks. ECG reference values are gated to a
physiologic range, with an optional per-subject outlier gate for double-counted
ECG. Requires the trained model and the bcg_signal_classifier package.
"""


from __future__ import annotations

import argparse

import itertools

import json

import logging

import os

import warnings

from pathlib import Path

from typing import Iterable, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

import pandas as pd

from scipy.signal import find_peaks

try:
    import heartpy as hp
    HEARTPY_IMPORT_ERROR = None
except ImportError as exc:
    hp = None
    HEARTPY_IMPORT_ERROR = exc

from bcg_signal_classifier.inference import (
    _config_from_inference,
    _filter_full_signal,
    predict_dataframe,
    segments_from_windows,
)

from bcg_signal_classifier.persistence import load_artifacts

LOGGER = logging.getLogger("exp6_model_gated_hr_inference")

PEAK_SUMMARY_KEYS = ("hr_bpm", "valid_jj_count", "peak_count", "median_jj", "jj_cv")

def empty_peak_summary() -> dict[str, float]:
    return {
        "hr_bpm": np.nan,
        "valid_jj_count": 0,
        "peak_count": 0,
        "median_jj": np.nan,
        "jj_cv": np.nan,
    }

def parse_fraction_threshold(value: float | str, label: str) -> float:
    threshold = float(value)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"{label} must be between 0 and 1, got {threshold}")
    return threshold

def parse_threshold_values(spec: str, default_threshold: float) -> list[float]:
    cleaned = str(spec).strip()
    if not cleaned:
        return []

    values = {parse_fraction_threshold(default_threshold, label="Default threshold")}
    for piece in cleaned.split(","):
        token = piece.strip()
        if not token:
            continue
        values.add(parse_fraction_threshold(token, label="Sweep threshold"))
    return sorted(values)

def parse_patient_threshold_overrides(spec: str) -> dict[str, float]:
    cleaned = str(spec).strip()
    if not cleaned:
        return {}

    overrides: dict[str, float] = {}
    for piece in cleaned.split(","):
        token = piece.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                "Patient-specific threshold entries must use key=value syntax, "
                f"for example 'PA0006=0.90'. Got: {token}"
            )
        key, value = token.split("=", 1)
        lookup_key = key.strip()
        if not lookup_key:
            raise ValueError(f"Patient-specific threshold entry is missing a key: {token}")
        overrides[lookup_key] = parse_fraction_threshold(value.strip(), label=f"Threshold for {lookup_key}")
    return overrides

def resolve_recording_threshold(
    patient_id: str,
    recording_name: str,
    default_threshold: float,
    patient_threshold_overrides: dict[str, float],
) -> tuple[float, str]:
    if recording_name in patient_threshold_overrides:
        return float(patient_threshold_overrides[recording_name]), "recording_name_override"
    if patient_id in patient_threshold_overrides:
        return float(patient_threshold_overrides[patient_id]), "patient_id_override"
    return float(default_threshold), "default"


def load_recording(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep="\t")

    original_columns = tuple(df)
    normalized_columns = tuple(str(col).strip() for col in original_columns)
    df = df.rename(columns=dict(zip(original_columns, normalized_columns)))
    lower_map = {col.lower(): col for col in normalized_columns}

    epoch_col = lower_map.get("epoch")
    if epoch_col is None:
        epoch_col = next((col for col in normalized_columns if "epoch" in col.lower()), None)

    bcg_col = lower_map.get("raw_data_sleepmat")
    if bcg_col is None:
        bcg_col = next(
            (
                col
                for col in normalized_columns
                if any(token in col.lower() for token in ("raw_data_sleepmat", "sleepmat", "raw_data", "bcg"))
            ),
            None,
        )

    ecg_col = lower_map.get("ecg_signal")
    if ecg_col is None:
        ecg_col = next((col for col in normalized_columns if "ecg" in col.lower()), None)

    missing = []
    if epoch_col is None:
        missing.append("epoch")
    if bcg_col is None:
        missing.append("raw_data_sleepMat")
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}; found {list(normalized_columns)}")

    out = pd.DataFrame(
        {
            "epoch": pd.to_numeric(df[epoch_col], errors="coerce"),
            "raw_data_sleepMat": pd.to_numeric(df[bcg_col], errors="coerce"),
        }
    )
    if ecg_col is not None:
        out["ecg_signal"] = pd.to_numeric(df[ecg_col], errors="coerce")

    out = out.dropna(subset=["epoch", "raw_data_sleepMat"]).sort_values("epoch").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{path} has no usable rows after numeric conversion")
    return out

def patient_id_from_path(path: Path) -> str:
    stem = path.stem
    return stem.split("__", 1)[0]

def resolve_sampling_rate(df: pd.DataFrame, fallback_fs: float) -> float:
    epoch = df["epoch"].to_numpy(dtype=np.float64)
    if epoch.size < 2:
        return float(fallback_fs)

    diffs_ms = np.diff(epoch)
    diffs_ms = diffs_ms[np.isfinite(diffs_ms) & (diffs_ms > 0)]
    if diffs_ms.size == 0:
        return float(fallback_fs)

    median_diff_ms = float(np.median(diffs_ms))
    if median_diff_ms <= 0:
        return float(fallback_fs)
    return float(1000.0 / median_diff_ms)

def detect_primary_j_peak(one_second_signal: np.ndarray, fs: float, hr_max: float, prominence_coef: float = 0.05) -> tuple[float, float, float]:
    """Return (peak_index, peak_value, peak_prominence) inside one 1-second window.

    The selected diagnostic J-peak is the highest valid positive peak in the
    1-second BCG slice. If no local peak is found, the maximum sample is used as
    a fallback.
    """
    signal = np.asarray(one_second_signal, dtype=np.float64)
    if signal.size == 0 or not np.isfinite(signal).any():
        return np.nan, np.nan, np.nan

    min_distance = max(1, int(fs * 60.0 / hr_max))
    sig_std = float(np.nanstd(signal))
    height_floor = float(np.nanmean(signal))
    peaks, properties = find_peaks(
        signal,
        distance=min_distance,
        prominence=max(0.0, prominence_coef * sig_std),
        height=height_floor,
    )

    if len(peaks) > 0:
        heights = properties.get("peak_heights", signal[peaks])
        prominences = properties.get("prominences", np.full(len(peaks), np.nan))
        best_idx = int(np.argmax(heights))
        return float(peaks[best_idx]), float(heights[best_idx]), float(prominences[best_idx])

    max_idx = int(np.nanargmax(signal))
    return float(max_idx), float(signal[max_idx]), np.nan


def get_candidates_in_window(
    one_second_signal: np.ndarray, fs: float, hr_max: float
) -> list[tuple[int, float, float]]:
    """Return ALL candidate peaks in a 1-second window as (index, height, prominence)."""
    signal = np.asarray(one_second_signal, dtype=np.float64)
    if signal.size == 0 or not np.isfinite(signal).any():
        return []
    min_distance = max(1, int(fs * 60.0 / hr_max))
    sig_std = float(np.nanstd(signal))
    if sig_std <= 0:
        return []
    # Use a low threshold to get ALL plausible peaks
    peaks, properties = find_peaks(
        signal,
        distance=min_distance,
        prominence=max(0.0, 0.03 * sig_std),
    )
    if len(peaks) == 0:
        # Fallback: just return the max
        max_idx = int(np.nanargmax(signal))
        return [(max_idx, float(signal[max_idx]), 0.0)]
    heights = signal[peaks]
    prominences = properties.get("prominences", np.zeros(len(peaks)))
    return [(int(peaks[i]), float(heights[i]), float(prominences[i])) for i in range(len(peaks))]


def viterbi_select_peaks(
    block_seconds: pd.DataFrame,
    filtered_signal: np.ndarray,
    fs: float,
    hr_min: float = 40.0,
    hr_max: float = 150.0,
) -> list[float]:
    """Select one peak per BCG second using Viterbi-style dynamic programming.

    Finds the globally optimal sequence of peaks that maximizes:
    - Emission: peak prominence and height (quality)
    - Transition: consistency of J-J intervals within physiologic range

    This resolves I/J/K ambiguity by choosing peaks that form the most
    rhythmically coherent sequence across the entire 30s block.
    """
    _peak_col = "is_bcg_peak" if "is_bcg_peak" in block_seconds.columns else "is_bcg"
    bcg_rows = block_seconds[(block_seconds[_peak_col] == 1)].reset_index(drop=True)
    if bcg_rows.empty or len(bcg_rows) < 3:
        return []

    min_jj = fs * 60.0 / hr_max  # ~20 samples at 50Hz/150bpm
    max_jj = fs * 60.0 / hr_min  # ~75 samples at 50Hz/40bpm

    # Step 1: Extract all candidate peaks per window (absolute positions)
    candidates_per_window: list[list[tuple[float, float]]] = []  # [(abs_pos, score), ...]
    for row in bcg_rows.itertuples(index=False):
        start = int(row.window_start_sample)
        end = int(row.window_end_sample)
        seg = filtered_signal[start:end].astype(np.float64)
        cands = get_candidates_in_window(seg, fs=fs, hr_max=hr_max)
        # Convert to absolute positions and compute quality score
        window_cands = []
        for (rel_idx, height, prom) in cands:
            abs_pos = float(start + rel_idx)
            # Score: prominence is the primary indicator of a true cardiac peak
            score = prom + 0.3 * max(0.0, height)
            window_cands.append((abs_pos, score))
        if not window_cands:
            # If no candidates, use the tallest point
            if seg.size > 0:
                max_rel = int(np.argmax(seg))
                window_cands.append((float(start + max_rel), 0.1))
        candidates_per_window.append(window_cands)

    n_windows = len(candidates_per_window)
    if n_windows < 3:
        return [c[0] for c in candidates_per_window if c] if candidates_per_window else []

    # Step 2: Viterbi forward pass
    # State: which candidate peak is selected in each window
    # Cost = emission (peak quality) + transition (interval consistency)
    INF = 1e18

    # For each window, store best_score[window][candidate_idx] and backpointer
    best_scores: list[list[float]] = []
    backpointers: list[list[int]] = []

    # Initialize first window
    first_cands = candidates_per_window[0]
    best_scores.append([c[1] for c in first_cands])  # emission only
    backpointers.append([-1] * len(first_cands))

    # Forward pass
    for w in range(1, n_windows):
        curr_cands = candidates_per_window[w]
        prev_cands = candidates_per_window[w - 1]
        w_scores = []
        w_backptrs = []

        for j, (pos_j, emit_j) in enumerate(curr_cands):
            best_prev_score = -INF
            best_prev_idx = 0

            for i, (pos_i, _) in enumerate(prev_cands):
                jj_interval = pos_j - pos_i
                # Transition score: flat reward for intervals in physiologic range
                if min_jj <= jj_interval <= max_jj:
                    trans_score = 1.0
                elif jj_interval > 0:
                    trans_score = -2.0
                else:
                    trans_score = -INF

                total = best_scores[w - 1][i] + trans_score
                if total > best_prev_score:
                    best_prev_score = total
                    best_prev_idx = i

            w_scores.append(best_prev_score + emit_j)
            w_backptrs.append(best_prev_idx)

        best_scores.append(w_scores)
        backpointers.append(w_backptrs)

    # Step 3: Backtrack to find optimal sequence
    # Find best final state
    last_scores = best_scores[-1]
    best_final = int(np.argmax(last_scores))

    # Trace back
    selected_indices = [best_final]
    for w in range(n_windows - 1, 0, -1):
        selected_indices.append(backpointers[w][selected_indices[-1]])
    selected_indices.reverse()

    # Extract absolute peak positions
    result = []
    for w, idx in enumerate(selected_indices):
        if idx < len(candidates_per_window[w]):
            result.append(candidates_per_window[w][idx][0])

    return result


def detect_candidate_j_peaks(signal_segment: np.ndarray, fs: float, hr_max: float) -> np.ndarray:
    """Detect candidate J-peaks in a continuous BCG segment.

    Unlike the per-second diagnostic peak, this returns physiologically plausible
    local maxima across the block. Detection is intentionally conservative to
    reduce double-counting multiple sub-peaks within one cardiac cycle.
    """
    signal = np.asarray(signal_segment, dtype=np.float64)
    if signal.size < 3:
        return np.asarray([], dtype=np.int64)

    finite_mask = np.isfinite(signal)
    if not finite_mask.any():
        return np.asarray([], dtype=np.int64)
    if not finite_mask.all():
        finite_idx = np.flatnonzero(finite_mask)
        if finite_idx.size < 2:
            return np.asarray([], dtype=np.int64)
        signal = np.interp(np.arange(signal.size), finite_idx, signal[finite_mask])

    centered = signal - float(np.median(signal))
    sig_std = float(np.std(centered))
    if not np.isfinite(sig_std) or sig_std <= 0:
        return np.asarray([], dtype=np.int64)

    min_distance = max(1, int(round(fs * 60.0 / hr_max)))
    peaks, properties = find_peaks(
        centered,
        distance=min_distance,
        prominence=max(1e-6, 0.10 * sig_std),
        height=0.0,
    )
    if len(peaks) == 0:
        peaks, properties = find_peaks(
            centered,
            distance=min_distance,
            prominence=max(1e-6, 0.05 * sig_std),
            height=0.0,
        )
    if len(peaks) == 0:
        return np.asarray([], dtype=np.int64)

    prominences = np.asarray(properties.get("prominences", np.zeros(len(peaks))), dtype=np.float64)
    heights = np.asarray(properties.get("peak_heights", centered[peaks]), dtype=np.float64)
    scores = prominences + 0.5 * np.maximum(heights, 0.0)

    selected: list[int] = []
    selected_scores: list[float] = []
    for idx in np.argsort(scores)[::-1]:
        peak = int(peaks[idx])
        if any(abs(peak - kept_peak) < min_distance for kept_peak in selected):
            continue
        selected.append(peak)
        selected_scores.append(float(scores[idx]))

    if not selected:
        return np.asarray([], dtype=np.int64)

    selected = np.asarray(sorted(selected), dtype=np.int64)
    selected_scores = np.asarray(selected_scores, dtype=np.float64)
    if selected_scores.size >= 3:
        score_floor = 0.5 * float(np.median(selected_scores))
        keep_mask = selected_scores >= score_floor
        if keep_mask.any():
            selected = selected[keep_mask]

    return np.asarray(np.sort(selected), dtype=np.int64)

def collect_block_j_peak_samples(
    block: pd.DataFrame,
    filtered_signal: np.ndarray,
    hr_max: float,
) -> np.ndarray:
    """Detect all candidate J-peaks inside BCG-labeled portions of a block."""
    if block.empty:
        return np.asarray([], dtype=np.float64)

    _peak_col = "is_bcg_peak" if "is_bcg_peak" in block.columns else "is_bcg"
    bcg_rows = block[block[_peak_col] == 1]
    if bcg_rows.empty:
        return np.asarray([], dtype=np.float64)

    block_start_sample = int(block["window_start_sample"].min())
    block_end_sample = int(block["window_end_sample"].max())
    if block_end_sample - block_start_sample < 3:
        return np.asarray([], dtype=np.float64)

    block_signal = filtered_signal[block_start_sample:block_end_sample]
    candidate_peaks_rel = detect_candidate_j_peaks(block_signal, fs=float(block["signal_fs"].iloc[0]), hr_max=hr_max)
    if candidate_peaks_rel.size == 0:
        return np.asarray([], dtype=np.float64)

    valid_mask = np.zeros(block_end_sample - block_start_sample, dtype=bool)
    for row in bcg_rows.itertuples(index=False):
        seg_start = max(0, int(row.window_start_sample) - block_start_sample)
        seg_end = min(block_end_sample - block_start_sample, int(row.window_end_sample) - block_start_sample)
        if seg_end > seg_start:
            valid_mask[seg_start:seg_end] = True

    kept_rel = candidate_peaks_rel[valid_mask[np.clip(candidate_peaks_rel, 0, len(valid_mask) - 1)]]
    if kept_rel.size == 0:
        return np.asarray([], dtype=np.float64)
    return (block_start_sample + kept_rel).astype(np.float64)


def detect_j_peaks_improved(
    signal: np.ndarray,
    fs: float,
    hr_min: float = 40.0,
    hr_max: float = 150.0,
    adaptive_thresh_mult: float = 1.2,
) -> tuple[np.ndarray, dict]:
    """Improved J-peak detection: adaptive threshold + template matching + I/J/K disambiguation.

    Phase 1 - Adaptive Threshold:
      Tracks signal envelope and uses a running adaptive threshold to detect
      candidate peaks. Adapts to amplitude changes across the 30s block.

    Phase 2 - Template Matching:
      Builds a J-peak template from the most prominent/consistent beats, then
      cross-correlates to refine peak locations and find missed beats.

    Phase 3 - I/J/K Disambiguation:
      Validates each candidate by checking morphological context: a true J-peak
      is preceded by a negative I-wave and followed by a negative K-wave.
      Peaks without this signature are rejected.

    Returns (sorted peak indices, diagnostics dict).
    """
    sig = np.asarray(signal, dtype=np.float64)
    diag: dict = {"phase1_count": 0, "phase2_count": 0, "phase3_count": 0, "template_corr_mean": np.nan}

    if sig.size < int(fs * 2):
        return np.asarray([], dtype=np.int64), diag

    # Handle NaN/Inf
    finite_mask = np.isfinite(sig)
    if not finite_mask.any():
        return np.asarray([], dtype=np.int64), diag
    if not finite_mask.all():
        finite_idx = np.flatnonzero(finite_mask)
        sig = np.interp(np.arange(sig.size), finite_idx, sig[finite_mask])

    # Physiologic constraints
    min_spacing = int(round(fs * 60.0 / hr_max))  # min samples between beats
    max_spacing = int(round(fs * 60.0 / hr_min))  # max samples between beats

    # ========== PHASE 1: Adaptive envelope threshold ==========
    # Compute signal envelope using RMS in sliding windows
    env_window = max(3, int(round(fs * 0.1)))  # 100ms RMS window
    padded = np.pad(sig, (env_window // 2, env_window // 2), mode="reflect")
    envelope = np.sqrt(
        np.convolve(padded ** 2, np.ones(env_window) / env_window, mode="valid")[:sig.size]
    )

    # Adaptive threshold: running median of envelope * multiplier
    # Use a 3-second sliding window for the running median
    median_window = max(3, int(round(fs * 3.0)))
    half_mw = median_window // 2
    adaptive_thresh = np.zeros_like(sig)
    for i in range(sig.size):
        lo = max(0, i - half_mw)
        hi = min(sig.size, i + half_mw)
        adaptive_thresh[i] = np.median(envelope[lo:hi])
    adaptive_thresh *= adaptive_thresh_mult  # threshold = mult x local median envelope

    # Find peaks above adaptive threshold with minimum physiologic spacing
    above_thresh = sig > adaptive_thresh
    candidates_p1, props_p1 = find_peaks(
        sig,
        distance=min_spacing,
        height=adaptive_thresh.mean() * 0.5,
        prominence=max(1e-6, 0.08 * float(np.std(sig))),
    )

    # Also keep peaks in above-threshold regions
    if candidates_p1.size > 0:
        # Score by prominence * height
        proms = props_p1.get("prominences", np.ones(len(candidates_p1)))
        heights = props_p1.get("peak_heights", sig[candidates_p1])
        scores_p1 = np.asarray(proms, dtype=np.float64) * np.maximum(heights, 0.0)
    else:
        # Fallback: just use basic find_peaks
        candidates_p1, props_p1 = find_peaks(sig, distance=min_spacing, prominence=0.05 * float(np.std(sig)))
        if candidates_p1.size == 0:
            return np.asarray([], dtype=np.int64), diag
        proms = props_p1.get("prominences", np.ones(len(candidates_p1)))
        heights = sig[candidates_p1]
        scores_p1 = np.asarray(proms, dtype=np.float64) * np.maximum(heights, 0.0)

    diag["phase1_count"] = int(candidates_p1.size)

    if candidates_p1.size < 2:
        return candidates_p1.astype(np.int64), diag

    # ========== PHASE 2: Template matching ==========
    # Select the best peaks for template building (top 30% by score, min 3)
    n_template_beats = max(3, int(round(0.3 * candidates_p1.size)))
    top_indices = np.argsort(scores_p1)[::-1][:n_template_beats]

    # Extract beat windows centered on each top peak (±0.4s around peak)
    half_template = int(round(0.4 * fs))
    template_beats = []
    for idx in top_indices:
        pk = int(candidates_p1[idx])
        lo = pk - half_template
        hi = pk + half_template
        if lo >= 0 and hi < sig.size:
            beat = sig[lo:hi].copy()
            # Normalize each beat to unit energy for averaging
            norm = np.sqrt(np.sum(beat ** 2))
            if norm > 0:
                template_beats.append(beat / norm)

    if len(template_beats) < 2:
        # Can't build template, use phase 1 results
        diag["phase2_count"] = int(candidates_p1.size)
        phase2_peaks = candidates_p1
    else:
        # Build average template
        template = np.mean(template_beats, axis=0)
        template = template / np.sqrt(np.sum(template ** 2))  # normalize

        # Cross-correlate template with signal
        # Normalize signal in sliding windows for correlation
        corr = np.zeros(sig.size, dtype=np.float64)
        for i in range(half_template, sig.size - half_template):
            window = sig[i - half_template:i + half_template]
            w_norm = np.sqrt(np.sum(window ** 2))
            if w_norm > 0:
                corr[i] = float(np.dot(window / w_norm, template))

        # Find correlation peaks with physiologic spacing
        corr_peaks, corr_props = find_peaks(corr, distance=min_spacing, height=0.3)
        if corr_peaks.size == 0:
            corr_peaks, corr_props = find_peaks(corr, distance=min_spacing, height=0.2)

        if corr_peaks.size >= 2:
            # Merge phase1 and correlation peaks, preferring correlation-refined locations
            merged = set()
            for cp in corr_peaks:
                merged.add(int(cp))
            # Add phase1 peaks that aren't near any correlation peak
            for p1 in candidates_p1:
                if not any(abs(int(p1) - m) < min_spacing for m in merged):
                    merged.add(int(p1))
            phase2_peaks = np.sort(np.array(list(merged), dtype=np.int64))

            # Compute mean correlation at phase2 peaks
            valid_corr = corr[phase2_peaks[(phase2_peaks >= 0) & (phase2_peaks < len(corr))]]
            diag["template_corr_mean"] = float(np.mean(valid_corr)) if valid_corr.size > 0 else np.nan
        else:
            phase2_peaks = candidates_p1

        diag["phase2_count"] = int(phase2_peaks.size)

    # ========== PHASE 3: I/J/K Morphological Disambiguation ==========
    # J-peak morphology: preceded by negative I-wave (50-150ms before),
    # followed by negative K-wave (80-250ms after).
    # The J-peak is the dominant positive deflection in the BCG cardiac cycle.
    i_wave_window = (int(round(0.05 * fs)), int(round(0.15 * fs)))  # 50-150ms before
    k_wave_window = (int(round(0.08 * fs)), int(round(0.25 * fs)))  # 80-250ms after

    validated_peaks = []
    for pk in phase2_peaks:
        pk = int(pk)
        if pk < i_wave_window[1] or pk >= sig.size - k_wave_window[1]:
            # Near edges — can't validate, keep with lower confidence
            validated_peaks.append(pk)
            continue

        peak_val = sig[pk]

        # Check for I-wave (negative deflection before J)
        i_region = sig[pk - i_wave_window[1]:pk - i_wave_window[0]]
        has_i_wave = False
        if i_region.size > 0:
            i_min = float(np.min(i_region))
            # I-wave should be below the peak and ideally below signal mean
            has_i_wave = i_min < peak_val * 0.5

        # Check for K-wave (negative deflection after J)
        k_region = sig[pk + k_wave_window[0]:pk + k_wave_window[1]]
        has_k_wave = False
        if k_region.size > 0:
            k_min = float(np.min(k_region))
            has_k_wave = k_min < peak_val * 0.5

        # Accept if either I or K wave signature is present (don't require both —
        # some BCG morphologies have weak I-waves or attenuated K-waves)
        if has_i_wave or has_k_wave:
            validated_peaks.append(pk)
        else:
            # Check if peak is still the local maximum in a wider window —
            # if so, it's likely a valid peak even without clear I/K
            local_window = sig[max(0, pk - min_spacing):min(sig.size, pk + min_spacing)]
            if sig[pk] >= np.max(local_window) * 0.95:
                validated_peaks.append(pk)

    phase3_peaks = np.sort(np.array(validated_peaks, dtype=np.int64))
    diag["phase3_count"] = int(phase3_peaks.size)

    # Final spacing enforcement: remove peaks too close together (keep higher ones)
    if phase3_peaks.size > 1:
        final = [int(phase3_peaks[0])]
        for i in range(1, phase3_peaks.size):
            if phase3_peaks[i] - final[-1] >= min_spacing:
                final.append(int(phase3_peaks[i]))
            else:
                # Keep the one with higher signal value
                if sig[int(phase3_peaks[i])] > sig[final[-1]]:
                    final[-1] = int(phase3_peaks[i])
        phase3_peaks = np.array(final, dtype=np.int64)

    return phase3_peaks, diag


def collect_block_j_peak_samples_improved(
    block: pd.DataFrame,
    filtered_signal: np.ndarray,
    hr_min: float,
    hr_max: float,
    adaptive_thresh_mult: float = 1.2,
) -> tuple[np.ndarray, dict]:
    """Detect J-peaks using the improved 3-phase algorithm."""
    if block.empty:
        return np.asarray([], dtype=np.float64), {}

    _peak_col = "is_bcg_peak" if "is_bcg_peak" in block.columns else "is_bcg"
    bcg_rows = block[block[_peak_col] == 1]
    if bcg_rows.empty:
        return np.asarray([], dtype=np.float64), {}

    block_start_sample = int(block["window_start_sample"].min())
    block_end_sample = int(block["window_end_sample"].max())
    if block_end_sample - block_start_sample < 3:
        return np.asarray([], dtype=np.float64), {}

    block_signal = filtered_signal[block_start_sample:block_end_sample]
    fs = float(block["signal_fs"].iloc[0])

    peaks_rel, diag = detect_j_peaks_improved(block_signal, fs=fs, hr_min=hr_min, hr_max=hr_max, adaptive_thresh_mult=adaptive_thresh_mult)
    if peaks_rel.size == 0:
        return np.asarray([], dtype=np.float64), diag

    # Mask to BCG-labeled regions only
    valid_mask = np.zeros(block_end_sample - block_start_sample, dtype=bool)
    for row in bcg_rows.itertuples(index=False):
        seg_start = max(0, int(row.window_start_sample) - block_start_sample)
        seg_end = min(block_end_sample - block_start_sample, int(row.window_end_sample) - block_start_sample)
        if seg_end > seg_start:
            valid_mask[seg_start:seg_end] = True

    kept_rel = peaks_rel[valid_mask[np.clip(peaks_rel, 0, len(valid_mask) - 1)]]
    if kept_rel.size == 0:
        return np.asarray([], dtype=np.float64), diag
    return (block_start_sample + kept_rel).astype(np.float64), diag

def build_second_level_table(
    pred_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    fs: float,
    hr_max: float,
    window_sec: float,
    stride_sec: float,
    prominence_coef: float = 0.05,
    disable_gate: bool = False,
    block_gate_only: bool = False,
) -> pd.DataFrame:
    bcg_signal = filtered_df["filtered"].to_numpy(dtype=np.float32)
    epochs = filtered_df["epoch"].to_numpy(dtype=np.float64)
    win = max(2, int(round(window_sec * fs)))
    hop = max(1, int(round(stride_sec * fs)))

    rows = []
    pred_df = pred_df.reset_index(drop=True)
    for row_idx, row in enumerate(pred_df.itertuples(index=False)):
        start_sample = row_idx * hop
        end_sample = min(start_sample + win, len(bcg_signal))
        seg = bcg_signal[start_sample:end_sample]
        peak_idx, peak_value, peak_prom = detect_primary_j_peak(seg, fs=fs, hr_max=hr_max, prominence_coef=prominence_coef)

        if np.isfinite(peak_idx):
            peak_sample_abs = int(start_sample + int(peak_idx))
            peak_epoch = float(epochs[min(peak_sample_abs, len(epochs) - 1)])
        else:
            peak_sample_abs = np.nan
            peak_epoch = np.nan

        rows.append(
            {
                "recording_second_idx": int(row_idx),
                "start_time": float(row.start_time),
                "end_time": float(row.end_time),
                "pred_index": int(row.pred_index),
                "pred_label": row.pred_label,
                "prob_BCG": float(row.prob_BCG),
                "prob_BCG_uncal": float(getattr(row, "prob_BCG_uncal", np.nan)),
                "is_bcg": 1 if disable_gate else int(row.pred_index == 1),
                # is_bcg drives bcg_fraction -> BLOCK-level gating.
                # is_bcg_peak drives J-peak masking INSIDE a retained block.
                # --block_gate_only keeps the former and disables the latter.
                "is_bcg_peak": 1 if (disable_gate or block_gate_only) else int(row.pred_index == 1),
                "signal_fs": float(fs),
                "window_start_sample": int(start_sample),
                "window_end_sample": int(end_sample),
                "j_peak_sample_abs": peak_sample_abs,
                "j_peak_epoch": peak_epoch,
                "j_peak_value": peak_value,
                "j_peak_prominence": peak_prom,
            }
        )

    return pd.DataFrame(rows)

def summarize_j_peak_intervals(
    j_peak_samples: Iterable[float],
    fs: float,
    hr_min: float,
    hr_max: float,
    min_valid_jj_intervals: int,
    use_mode_seeking: bool = False,
) -> dict[str, float]:
    peaks = np.asarray([sample for sample in j_peak_samples if np.isfinite(sample)], dtype=np.float64)
    if peaks.size < 2:
        return empty_peak_summary() | {"peak_count": int(peaks.size)}

    peaks = np.unique(np.sort(peaks))
    if peaks.size < 2:
        return empty_peak_summary() | {"peak_count": int(peaks.size)}

    jj = np.diff(peaks)
    min_jj = fs * 60.0 / hr_max
    max_jj = fs * 60.0 / hr_min
    valid_jj = jj[(jj >= min_jj) & (jj <= max_jj)]
    if valid_jj.size == 0:
        return empty_peak_summary() | {"peak_count": int(peaks.size)}

    if valid_jj.size >= 3:
        median_jj = float(np.median(valid_jj))
        mad_jj = float(np.median(np.abs(valid_jj - median_jj)))
        if mad_jj > 0:
            robust_lower = max(min_jj, median_jj - 3.0 * mad_jj)
            robust_upper = min(max_jj, median_jj + 3.0 * mad_jj)
            robust_jj = valid_jj[(valid_jj >= robust_lower) & (valid_jj <= robust_upper)]
            if robust_jj.size >= int(min_valid_jj_intervals):
                valid_jj = robust_jj

    if valid_jj.size < int(min_valid_jj_intervals):
        return empty_peak_summary() | {"valid_jj_count": int(valid_jj.size), "peak_count": int(peaks.size)}

    if use_mode_seeking and valid_jj.size >= 5:
        median_jj = _kde_mode_jj(valid_jj, min_jj, max_jj)
        if not (np.isfinite(median_jj) and median_jj > 0):
            median_jj = float(np.median(valid_jj))
    else:
        median_jj = float(np.median(valid_jj))

    jj_cv = float(np.std(valid_jj) / median_jj) if median_jj > 0 else np.nan
    hr_bpm = 60.0 * fs / median_jj if median_jj > 0 else np.nan
    return {
        "hr_bpm": float(hr_bpm) if np.isfinite(hr_bpm) else np.nan,
        "valid_jj_count": int(valid_jj.size),
        "peak_count": int(peaks.size),
        "median_jj": median_jj,
        "jj_cv": float(jj_cv) if np.isfinite(jj_cv) else np.nan,
    }


def _kde_mode_jj(intervals: np.ndarray, min_jj: float, max_jj: float) -> float:
    """Find the mode of J-J intervals via Gaussian KDE."""
    intervals = np.asarray(intervals, dtype=np.float64)
    if intervals.size < 3:
        return float(np.median(intervals))
    std = float(np.std(intervals))
    if std <= 0:
        return float(np.median(intervals))
    bw = max(1.0, 1.06 * std * (intervals.size ** (-1.0 / 5.0)))
    grid = np.linspace(min_jj, max_jj, 200)
    density = np.zeros_like(grid)
    for iv in intervals:
        density += np.exp(-0.5 * ((grid - iv) / bw) ** 2)
    return float(grid[int(np.argmax(density))])


def refine_peaks_cross_beat(
    peak_samples: list[float],
    filtered_signal: np.ndarray,
    fs: float,
    hr_min: float = 40.0,
    hr_max: float = 150.0,
) -> list[float]:
    """Refine per-second J-peak positions using cross-beat rhythm consistency."""
    peaks = np.asarray([p for p in peak_samples if np.isfinite(p)], dtype=np.float64)
    if peaks.size < 4:
        return peak_samples
    peaks_sorted = np.sort(peaks)
    jj = np.diff(peaks_sorted)
    min_jj = fs * 60.0 / hr_max
    max_jj = fs * 60.0 / hr_min
    valid_jj = jj[(jj >= min_jj) & (jj <= max_jj)]
    if valid_jj.size < 3:
        return peak_samples
    dominant_T = _kde_mode_jj(valid_jj, min_jj, max_jj)
    if not np.isfinite(dominant_T) or dominant_T <= 0:
        return peak_samples
    tolerance = dominant_T * 0.30
    best_seed_idx = 0
    best_score = 0
    for i, pk in enumerate(peaks_sorted):
        score = 0
        for j, other in enumerate(peaks_sorted):
            if i == j:
                continue
            diff = abs(other - pk)
            n_periods = round(diff / dominant_T)
            if n_periods >= 1 and abs(diff - n_periods * dominant_T) <= tolerance:
                score += 1
        if score > best_score:
            best_score = score
            best_seed_idx = i
    seed = peaks_sorted[best_seed_idx]
    grid = [seed]
    pos = seed + dominant_T
    while pos < filtered_signal.size:
        grid.append(pos)
        pos += dominant_T
    pos = seed - dominant_T
    while pos >= 0:
        grid.insert(0, pos)
        pos -= dominant_T
    refined = []
    search_radius = int(round(tolerance))
    for g in grid:
        g_int = int(round(g))
        lo = max(0, g_int - search_radius)
        hi = min(filtered_signal.size, g_int + search_radius)
        if hi <= lo:
            continue
        nearby = peaks_sorted[(peaks_sorted >= lo) & (peaks_sorted < hi)]
        if nearby.size > 0:
            refined.append(float(nearby[int(np.argmin(np.abs(nearby - g)))]))
        else:
            segment = filtered_signal[lo:hi]
            if segment.size > 0:
                candidate = lo + int(np.argmax(segment))
                if filtered_signal[candidate] > np.mean(segment):
                    refined.append(float(candidate))
    return refined if len(refined) >= 3 else peak_samples


def _kde_mode_jj(intervals: np.ndarray, min_jj: float, max_jj: float) -> float:
    """Find the mode of J-J intervals using a Gaussian KDE.

    Returns the interval value at the KDE peak (the most common spacing).
    More robust than median when a few wrong peaks inject outlier intervals.
    """
    intervals = np.asarray(intervals, dtype=np.float64)
    if intervals.size < 3:
        return float(np.median(intervals))

    # Bandwidth: Scott's rule with a floor to avoid overfitting
    std = float(np.std(intervals))
    if std <= 0:
        return float(np.median(intervals))
    bw = max(1.0, 1.06 * std * (intervals.size ** (-1.0 / 5.0)))

    # Evaluate KDE on a fine grid within physiologic range
    grid = np.linspace(min_jj, max_jj, 200)
    density = np.zeros_like(grid)
    for iv in intervals:
        density += np.exp(-0.5 * ((grid - iv) / bw) ** 2)

    mode_idx = int(np.argmax(density))
    return float(grid[mode_idx])


def refine_peaks_cross_beat(
    peak_samples: list[float],
    filtered_signal: np.ndarray,
    fs: float,
    hr_min: float = 40.0,
    hr_max: float = 150.0,
) -> list[float]:
    """Refine per-second J-peak positions using cross-beat rhythm consistency.

    The per-second detector picks the tallest peak independently per window,
    which can pick I or K peaks instead of J. This function enforces rhythm:
    it finds the dominant inter-peak interval and re-selects peaks that form
    the most consistent periodic sequence.

    Algorithm:
    1. Compute all pairwise J-J intervals from the raw per-second peaks.
    2. Find the dominant period T using KDE mode of valid intervals.
    3. Build a rhythm grid starting from the best seed peak (most neighbors at ~T).
    4. For each grid position, pick the closest actual peak within ±T*0.3 tolerance.
    5. For unmatched grid positions, search the signal for the local max near that grid point.
    """
    peaks = np.asarray([p for p in peak_samples if np.isfinite(p)], dtype=np.float64)
    if peaks.size < 4:
        return peak_samples  # too few to refine

    peaks_sorted = np.sort(peaks)
    jj = np.diff(peaks_sorted)
    min_jj = fs * 60.0 / hr_max
    max_jj = fs * 60.0 / hr_min
    valid_jj = jj[(jj >= min_jj) & (jj <= max_jj)]

    if valid_jj.size < 3:
        return peak_samples  # can't determine rhythm

    # Find dominant period via KDE mode
    dominant_T = _kde_mode_jj(valid_jj, min_jj, max_jj)
    if not np.isfinite(dominant_T) or dominant_T <= 0:
        return peak_samples

    tolerance = dominant_T * 0.30  # ±30% of expected period

    # Find seed peak: the one with most neighbors at approximately T spacing
    best_seed_idx = 0
    best_score = 0
    for i, pk in enumerate(peaks_sorted):
        score = 0
        for j, other in enumerate(peaks_sorted):
            if i == j:
                continue
            diff = abs(other - pk)
            # Count how many multiples of T this is close to
            n_periods = round(diff / dominant_T)
            if n_periods >= 1:
                expected = n_periods * dominant_T
                if abs(diff - expected) <= tolerance:
                    score += 1
        if score > best_score:
            best_score = score
            best_seed_idx = i

    seed = peaks_sorted[best_seed_idx]

    # Build rhythm grid extending in both directions from seed
    # IMPORTANT: limit to the block range (min/max of input peaks ± 1 period)
    block_lo = float(peaks_sorted[0]) - dominant_T
    block_hi = float(peaks_sorted[-1]) + dominant_T
    grid = [seed]
    # Forward
    pos = seed + dominant_T
    while pos <= block_hi:
        grid.append(pos)
        pos += dominant_T
    # Backward
    pos = seed - dominant_T
    while pos >= block_lo:
        grid.insert(0, pos)
        pos -= dominant_T

    # For each grid position, find the best matching peak or signal maximum
    refined = []
    search_radius = int(round(tolerance))
    for g in grid:
        g_int = int(round(g))
        lo = max(0, g_int - search_radius)
        hi = min(filtered_signal.size, g_int + search_radius)
        if hi <= lo:
            continue

        # First try: find an existing detected peak near this grid position
        nearby = peaks_sorted[(peaks_sorted >= lo) & (peaks_sorted < hi)]
        if nearby.size > 0:
            # Pick the one closest to the grid position
            best = nearby[int(np.argmin(np.abs(nearby - g)))]
            refined.append(float(best))
        else:
            # No detected peak nearby — find local max in signal
            segment = filtered_signal[lo:hi]
            if segment.size > 0:
                local_max_idx = int(np.argmax(segment))
                candidate = lo + local_max_idx
                # Only accept if it's a genuine local peak (above local mean)
                if filtered_signal[candidate] > np.mean(segment):
                    refined.append(float(candidate))

    if len(refined) < 3:
        return peak_samples  # refinement failed

    return refined

def estimate_spectral_hr(
    block_signal: np.ndarray,
    fs: float,
    hr_min: float,
    hr_max: float,
) -> dict[str, float]:
    """Estimate cardiac fundamental HR from the FFT power spectrum.

    Hann-windowed, zero-padded FFT; the dominant peak within the physiologic
    band [hr_min, hr_max] is refined by parabolic interpolation. Confidence is
    the raw power-spectral magnitude at the peak (order 1e4-1e6 on this cohort),
    used only as a diagnostic column and an opt-in fusion weight.

    NOTE: This is the baseline estimator. It does not feed hr_bcg_bpm (the
    half-rate/double correction that consumed it is disabled); the per-second
    diagnostic J-J estimate is the sole HR output.
    """
    signal = np.asarray(block_signal, dtype=np.float64)
    signal = signal[np.isfinite(signal)]
    min_samples = max(3, int(round(fs * 3.0)))
    if signal.size < min_samples:
        return {"hr_bpm": np.nan, "confidence": np.nan}

    signal = signal - float(np.mean(signal))
    if not np.any(signal):
        return {"hr_bpm": np.nan, "confidence": np.nan}

    f_lo = float(hr_min) / 60.0
    f_hi = float(hr_max) / 60.0
    if f_hi <= f_lo or fs <= 0:
        return {"hr_bpm": np.nan, "confidence": np.nan}

    n = signal.size
    windowed = signal * np.hanning(n)
    nfft = int(2 ** np.ceil(np.log2(max(n * 4, 8))))
    spectrum = np.fft.rfft(windowed, n=nfft)
    psd = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band):
        return {"hr_bpm": np.nan, "confidence": np.nan}

    band_idx = np.where(band)[0]
    local_peak = int(np.argmax(psd[band_idx]))
    peak_bin = int(band_idx[local_peak])
    confidence = float(psd[peak_bin])

    # Parabolic interpolation on the power spectrum for sub-bin frequency.
    if 0 < peak_bin < psd.size - 1:
        alpha = float(psd[peak_bin - 1])
        beta = float(psd[peak_bin])
        gamma = float(psd[peak_bin + 1])
        denom = alpha - 2.0 * beta + gamma
        delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-12 else 0.0
    else:
        delta = 0.0
    df = freqs[1] - freqs[0] if freqs.size > 1 else 0.0
    peak_freq = float(freqs[peak_bin]) + delta * df

    hr_bpm = 60.0 * peak_freq
    if not (np.isfinite(hr_bpm) and hr_min <= hr_bpm <= hr_max):
        return {"hr_bpm": np.nan, "confidence": confidence}
    return {"hr_bpm": float(hr_bpm), "confidence": confidence}

def estimate_hr_from_j_peaks(
    j_peak_samples: Iterable[float],
    fs: float,
    hr_min: float,
    hr_max: float,
    min_valid_jj_intervals: int,
) -> tuple[float, int]:
    summary = summarize_j_peak_intervals(
        j_peak_samples=j_peak_samples,
        fs=fs,
        hr_min=hr_min,
        hr_max=hr_max,
        min_valid_jj_intervals=min_valid_jj_intervals,
    )
    return summary["hr_bpm"], int(summary["valid_jj_count"])

def select_block_hr_estimate(
    diagnostic_peak_samples: Iterable[float],
    dense_peak_samples: Iterable[float],
    fs: float,
    hr_min: float,
    hr_max: float,
    min_valid_jj_intervals: int,
) -> tuple[float, int, str, dict[str, float], dict[str, float]]:
    diagnostic_summary = summarize_j_peak_intervals(
        j_peak_samples=diagnostic_peak_samples,
        fs=fs,
        hr_min=hr_min,
        hr_max=hr_max,
        min_valid_jj_intervals=min_valid_jj_intervals,
    )
    dense_summary = summarize_j_peak_intervals(
        j_peak_samples=dense_peak_samples,
        fs=fs,
        hr_min=hr_min,
        hr_max=hr_max,
        min_valid_jj_intervals=min_valid_jj_intervals,
    )

    diagnostic_hr = float(diagnostic_summary["hr_bpm"]) if np.isfinite(diagnostic_summary["hr_bpm"]) else np.nan
    dense_hr = float(dense_summary["hr_bpm"]) if np.isfinite(dense_summary["hr_bpm"]) else np.nan

    chosen = diagnostic_summary
    strategy = "diagnostic"

    if np.isfinite(diagnostic_hr):
        chosen = diagnostic_summary
        strategy = "diagnostic"
    elif np.isfinite(dense_hr):
        chosen = dense_summary
        strategy = "dense_fallback"
    else:
        strategy = "no_valid_hr"

    return (
        float(chosen["hr_bpm"]) if np.isfinite(chosen["hr_bpm"]) else np.nan,
        int(chosen["valid_jj_count"]),
        strategy,
        diagnostic_summary,
        dense_summary,
    )

def fuse_hr_estimates(
    diagnostic_hr: float,
    dense_hr: float,
    spectral_hr: float,
    diagnostic_jj_cv: float,
    dense_jj_cv: float,
    spectral_confidence: float,
    n_valid_diagnostic: int,
    n_valid_dense: int,
    min_valid_jj_intervals: int,
    spectral_conf_scale: float,
) -> tuple[float, dict[str, float]]:
    """Fuse HR estimates using robust median with agreement validation.

    Strategy:
    1. Collect all valid estimators.
    2. If 3 estimators available: use median (robust to one outlier).
    3. If 2 estimators agree within 20%: use their weighted average.
    4. If 2 estimators disagree by >20%: use the one with lower CV (or spectral
       if it's one of the two).
    5. If only 1 estimator: use it directly.

    This avoids the failure mode of weighted averaging when all estimators
    have similar confidence but wildly different HR values.
    """
    estimates = []
    weights_out = {"diagnostic": 0.0, "dense": 0.0, "spectral": 0.0}

    if np.isfinite(diagnostic_hr) and n_valid_diagnostic >= min_valid_jj_intervals:
        cv = float(diagnostic_jj_cv) if np.isfinite(diagnostic_jj_cv) else 0.5
        estimates.append(("diagnostic", diagnostic_hr, 1.0 / (1.0 + cv) ** 2))
    if np.isfinite(dense_hr) and n_valid_dense >= min_valid_jj_intervals:
        cv = float(dense_jj_cv) if np.isfinite(dense_jj_cv) else 0.5
        estimates.append(("dense", dense_hr, 1.0 / (1.0 + cv) ** 2))
    if np.isfinite(spectral_hr) and np.isfinite(spectral_confidence) and spectral_conf_scale > 0:
        spec_w = min(float(spectral_confidence) / float(spectral_conf_scale), 1.0)
        estimates.append(("spectral", spectral_hr, max(0.0, spec_w)))

    if not estimates:
        return np.nan, weights_out

    if len(estimates) == 1:
        name, hr, w = estimates[0]
        weights_out[name] = w
        return float(hr), weights_out

    # Sort by HR for median selection
    estimates_sorted = sorted(estimates, key=lambda x: x[1])

    if len(estimates) == 3:
        # Use median — robust to one outlier
        median_est = estimates_sorted[1]
        fused = median_est[1]
        # Assign weights: full weight to median, partial to those within 15%
        for name, hr, w in estimates:
            if abs(hr - fused) / max(fused, 1.0) < 0.15:
                weights_out[name] = w
            else:
                weights_out[name] = 0.0
        # If only median has weight, give it full credit
        if sum(weights_out.values()) == 0:
            weights_out[median_est[0]] = median_est[2]
        return float(fused), weights_out

    # 2 estimators: check agreement
    name_a, hr_a, w_a = estimates[0]
    name_b, hr_b, w_b = estimates[1]
    mean_hr = (hr_a + hr_b) / 2.0

    if mean_hr > 0 and abs(hr_a - hr_b) / mean_hr <= 0.20:
        # They agree — weighted average
        total_w = w_a + w_b
        if total_w > 0:
            fused = (w_a * hr_a + w_b * hr_b) / total_w
        else:
            fused = mean_hr
        weights_out[name_a] = w_a
        weights_out[name_b] = w_b
        return float(fused), weights_out
    else:
        # Disagreement — pick the one with higher confidence
        if w_a >= w_b:
            weights_out[name_a] = w_a
            return float(hr_a), weights_out
        else:
            weights_out[name_b] = w_b
            return float(hr_b), weights_out


def estimate_ecg_hr_heartpy(ecg_segment: np.ndarray, fs: float, hr_min: float = 40.0, hr_max: float = 150.0) -> tuple[float, str]:
    if hp is None:
        return np.nan, f"heartpy_import_error: {HEARTPY_IMPORT_ERROR}"

    ecg = np.asarray(ecg_segment, dtype=np.float64)
    ecg = ecg[np.isfinite(ecg)]
    if ecg.size < max(3, int(round(fs * 5))):
        return np.nan, "ecg_too_short"

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=r"Mean of empty slice")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=r"Degrees of freedom <= 0 for slice")
            warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"numpy(\..*)?")
            warnings.filterwarnings("ignore", category=UserWarning, module=r"heartpy(\..*)?")
            warnings.filterwarnings("ignore", category=UserWarning, message=r".*maxit.*")
            _, measures = hp.process(ecg, sample_rate=fs, clean_rr=True)
    except Exception:
        return np.nan, "heartpy_error"

    bpm = float(measures.get("bpm", np.nan))
    if not np.isfinite(bpm):
        return np.nan, "heartpy_invalid_bpm"
    # Reject physiologically impossible ECG estimates (heartpy emits spurious
    # values, e.g. >1000 bpm, on noisy segments; these must not become paired
    # reference targets or they dominate MAE/RMSE).
    if not (hr_min <= bpm <= hr_max):
        return np.nan, "ecg_out_of_range"
    return bpm, "ok"

def detect_motion_artifact_score(
    block_signal: np.ndarray,
    fs: float,
    motion_hf_cutoff_hz: float = 8.0,
) -> tuple[float, dict[str, float]]:
    """Score motion contamination in a 30-second BCG block (0=clean, 1=pure motion).

    Uses three complementary indicators:
    1. High-frequency energy ratio: fraction of spectral power above the motion
       cutoff (default 8 Hz). BCG cardiac content lives in 0.5-5 Hz with harmonics
       up to ~7 Hz; genuine motion injects broadband energy above 8 Hz.
    2. Amplitude spike rate: fraction of samples exceeding 4 sigma from the mean.
       Body movement creates large transient spikes absent in resting BCG.
    3. Variance non-stationarity: coefficient of variation of per-second RMS values.
       Cardiac BCG is quasi-stationary; motion episodes create heterogeneous variance.

    Returns (composite_score, detail_dict). The composite is a weighted average of
    the three indicators, each individually clipped to [0, 1].
    """
    signal = np.asarray(block_signal, dtype=np.float64)
    finite = signal[np.isfinite(signal)]
    if finite.size < max(3, int(round(fs * 2))):
        return 0.0, {"hf_energy_ratio": 0.0, "spike_rate": 0.0, "variance_cv": 0.0}

    signal_clean = finite - float(np.mean(finite))
    sig_std = float(np.std(signal_clean))
    if sig_std <= 0:
        return 0.0, {"hf_energy_ratio": 0.0, "spike_rate": 0.0, "variance_cv": 0.0}

    # 1. High-frequency energy ratio (above motion_hf_cutoff_hz, default 8 Hz)
    n = signal_clean.size
    nfft = int(2 ** np.ceil(np.log2(max(n, 8))))
    spectrum = np.fft.rfft(signal_clean * np.hanning(n), n=nfft)
    psd = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    total_power = float(psd.sum())
    hf_power = float(psd[freqs > motion_hf_cutoff_hz].sum()) if total_power > 0 else 0.0
    hf_energy_ratio = float(hf_power / total_power) if total_power > 0 else 0.0

    # 2. Amplitude spike rate (fraction of samples > 4 sigma)
    spike_threshold = 4.0 * sig_std
    spike_count = int(np.sum(np.abs(signal_clean) > spike_threshold))
    spike_rate = float(spike_count / n)
    spike_score = min(1.0, spike_rate / 0.02)

    # 3. Variance non-stationarity (CV of per-second RMS)
    samples_per_sec = max(1, int(round(fs)))
    n_full_secs = n // samples_per_sec
    variance_cv = 0.0
    if n_full_secs >= 3:
        rms_values = np.array([
            float(np.sqrt(np.mean(signal_clean[i * samples_per_sec:(i + 1) * samples_per_sec] ** 2)))
            for i in range(n_full_secs)
        ])
        rms_mean = float(np.mean(rms_values))
        if rms_mean > 0:
            variance_cv = float(np.std(rms_values) / rms_mean)
    variance_score = min(1.0, variance_cv / 1.0)

    # Weighted composite
    composite = 0.40 * hf_energy_ratio + 0.30 * spike_score + 0.30 * variance_score
    composite = float(min(1.0, max(0.0, composite)))

    detail = {
        "hf_energy_ratio": float(hf_energy_ratio),
        "spike_rate": float(spike_rate),
        "variance_cv": float(variance_cv),
    }
    return composite, detail


def summarize_blocks(
    second_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    source_df: pd.DataFrame,
    patient_id: str,
    fs: float,
    aggregate_sec: int,
    hr_min: float,
    hr_max: float,
    min_valid_jj_intervals: int,
    max_diagnostic_jj_cv: Optional[float],
    disable_ecg_eval: bool,
    use_spectral_check: bool = True,
    spectral_confidence_min: float = 1.5,
    half_double_tolerance: float = 0.18,
    ecg_subject_outlier_mad: float = 0.0,
    motion_artifact_threshold: float = 0.0,
    cross_estimator_max_diff: float = 0.0,
    use_improved_peaks: bool = False,
    refine_diagnostic_peaks: bool = False,
    fuse_estimators: bool = False,
    spectral_conf_scale: float = 500000.0,
    adaptive_thresh_mult: float = 1.2,
) -> pd.DataFrame:
    rows = []
    second_df = second_df.reset_index(drop=True).copy()
    second_df["block_idx"] = second_df.index // aggregate_sec
    source_columns = set(source_df)
    has_ecg = ("ecg_signal" in source_columns) and (not disable_ecg_eval)
    ecg_missing = ("ecg_signal" not in source_columns) and (not disable_ecg_eval)
    filtered_signal = filtered_df["filtered"].to_numpy(dtype=np.float64)

    for block_idx, block in second_df.groupby("block_idx", sort=True):
        n_seconds = int(len(block))
        n_bcg_seconds = int(block["is_bcg"].sum())
        bcg_fraction = float(n_bcg_seconds / n_seconds) if n_seconds else np.nan
        is_complete_block = n_seconds == aggregate_sec

        # Motion artifact pre-screening: compute raw score per block
        block_start_sample = int(block["window_start_sample"].min())
        block_end_sample = int(block["window_end_sample"].max())
        motion_score = 0.0
        is_motion_artifact = False
        if motion_artifact_threshold > 0 and (block_end_sample - block_start_sample) >= 3:
            motion_score, _motion_detail = detect_motion_artifact_score(
                filtered_signal[block_start_sample:block_end_sample],
                fs=fs,
            )

        _peak_col = "is_bcg_peak" if "is_bcg_peak" in block.columns else "is_bcg"
        valid_bcg_seconds = block[(block[_peak_col] == 1) & (block["j_peak_sample_abs"].notna())]
        diagnostic_j_peak_samples = valid_bcg_seconds["j_peak_sample_abs"].tolist()

        # Viterbi peak selection: re-select peaks using global path optimization
        if refine_diagnostic_peaks and len(diagnostic_j_peak_samples) >= 4:
            viterbi_result = viterbi_select_peaks(
                block_seconds=block,
                filtered_signal=filtered_signal,
                fs=fs,
                hr_min=hr_min,
                hr_max=hr_max,
            )
            if len(viterbi_result) >= 4:
                diagnostic_j_peak_samples = viterbi_result

        diagnostic_summary = summarize_j_peak_intervals(
            j_peak_samples=diagnostic_j_peak_samples,
            fs=fs,
            hr_min=hr_min,
            hr_max=hr_max,
            min_valid_jj_intervals=min_valid_jj_intervals,
            use_mode_seeking=False,
        )

        # Dense candidate J-peaks: use improved 3-phase detector if enabled
        if use_improved_peaks:
            block_j_peak_samples, _peak_diag = collect_block_j_peak_samples_improved(
                block=block,
                filtered_signal=filtered_signal,
                hr_min=hr_min,
                hr_max=hr_max,
                adaptive_thresh_mult=adaptive_thresh_mult,
            )
        else:
            block_j_peak_samples = collect_block_j_peak_samples(
                block=block,
                filtered_signal=filtered_signal,
                hr_max=hr_max,
            )
        dense_summary = summarize_j_peak_intervals(
            j_peak_samples=block_j_peak_samples.tolist(),
            fs=fs,
            hr_min=hr_min,
            hr_max=hr_max,
            min_valid_jj_intervals=min_valid_jj_intervals,
        )
        dense_hr = float(dense_summary["hr_bpm"]) if np.isfinite(dense_summary["hr_bpm"]) else np.nan
        diagnostic_hr = float(diagnostic_summary["hr_bpm"]) if np.isfinite(diagnostic_summary["hr_bpm"]) else np.nan

        # PRIMARY estimator: autocorrelation fundamental over the continuous block.
        spectral_hr = np.nan
        spectral_confidence = np.nan
        if use_spectral_check:
            if block_end_sample - block_start_sample >= 3:
                spectral = estimate_spectral_hr(
                    filtered_signal[block_start_sample:block_end_sample],
                    fs=fs,
                    hr_min=hr_min,
                    hr_max=hr_max,
                )
                spectral_hr = float(spectral["hr_bpm"]) if np.isfinite(spectral["hr_bpm"]) else np.nan
                spectral_confidence = float(spectral["confidence"]) if np.isfinite(spectral["confidence"]) else np.nan

        spectral_trusted = (
            use_spectral_check
            and np.isfinite(spectral_hr)
            and np.isfinite(spectral_confidence)
            and spectral_confidence >= spectral_confidence_min
            and hr_min <= spectral_hr <= hr_max
        )

        base_hr_bcg = np.nan
        base_valid_jj_count = 0
        peak_strategy = "no_valid_hr"
        hr_correction_applied = False

        diagnostic_jj_cv = float(diagnostic_summary["jj_cv"]) if np.isfinite(diagnostic_summary["jj_cv"]) else np.nan
        dense_jj_cv = float(dense_summary["jj_cv"]) if np.isfinite(dense_summary["jj_cv"]) else np.nan

        diagnostic_cv_ok = (
            max_diagnostic_jj_cv is None
            or not np.isfinite(diagnostic_jj_cv)
            or diagnostic_jj_cv <= float(max_diagnostic_jj_cv)
        )

        # Confidence-weighted fusion of the three estimators (opt-in, off by
        # default). Retained only as a diagnostic; it does not affect the
        # baseline output, which uses the per-second diagnostic estimate.
        fused_hr = np.nan
        fused_weights = {"diagnostic": np.nan, "dense": np.nan, "spectral": np.nan}
        if fuse_estimators:
            fused_hr, fused_weights = fuse_hr_estimates(
                diagnostic_hr=diagnostic_hr,
                dense_hr=dense_hr,
                spectral_hr=spectral_hr,
                diagnostic_jj_cv=diagnostic_jj_cv,
                dense_jj_cv=dense_jj_cv,
                spectral_confidence=spectral_confidence,
                n_valid_diagnostic=int(diagnostic_summary["valid_jj_count"]),
                n_valid_dense=int(dense_summary["valid_jj_count"]),
                min_valid_jj_intervals=min_valid_jj_intervals,
                spectral_conf_scale=spectral_conf_scale,
            )

        # PRIMARY estimator: per-second diagnostic J-J intervals. On this cohort it
        # tracks the ECG reference closely (median ratio ~0.95). Autocorrelation and
        # dense peak counting are retained only as fallbacks and diagnostic columns.
        # When use_improved_peaks is enabled, the improved dense detector is primary.
        if fuse_estimators and np.isfinite(fused_hr):
            base_hr_bcg = fused_hr
            base_valid_jj_count = int(diagnostic_summary["valid_jj_count"])
            peak_strategy = "fused"
        elif use_improved_peaks and np.isfinite(dense_hr):
            base_hr_bcg = dense_hr
            base_valid_jj_count = int(dense_summary["valid_jj_count"])
            peak_strategy = "improved_peaks"
        elif np.isfinite(diagnostic_hr) and diagnostic_cv_ok:
            base_hr_bcg = diagnostic_hr
            base_valid_jj_count = int(diagnostic_summary["valid_jj_count"])
            peak_strategy = "diagnostic"
        elif use_improved_peaks and np.isfinite(diagnostic_hr):
            # Improved peaks didn't produce valid HR, fall back to diagnostic
            base_hr_bcg = diagnostic_hr
            base_valid_jj_count = int(diagnostic_summary["valid_jj_count"])
            peak_strategy = "diagnostic_fallback_from_improved"
        elif spectral_trusted:
            base_hr_bcg = spectral_hr
            peak_strategy = "autocorr_fallback"
        elif np.isfinite(dense_hr):
            base_hr_bcg = dense_hr
            base_valid_jj_count = int(dense_summary["valid_jj_count"])
            peak_strategy = "dense_fallback"
        elif np.isfinite(diagnostic_hr):
            base_hr_bcg = diagnostic_hr
            base_valid_jj_count = int(diagnostic_summary["valid_jj_count"])
            peak_strategy = "diagnostic_cv_relaxed"

        # Half-rate / double-rate correction: DISABLED. Snapping the base HR to
        # the spectral fundamental replaced the reliable per-second diagnostic
        # (~55-75 bpm, MAE ~14) with the unreliable spectral estimate (often the
        # 2nd harmonic at ~130-150 bpm), which was the dominant error source.
        # The output now always equals the single-peak diagnostic estimate.
        # Re-enable only after the spectral estimator is proven trustworthy.
        if (
            False  # correction intentionally disabled; see note above
            and np.isfinite(base_hr_bcg)
            and spectral_trusted
            and half_double_tolerance > 0
        ):
            ratio = base_hr_bcg / spectral_hr
            if abs(ratio - 0.5) <= half_double_tolerance:
                # Half-rate detected: diagnostic is ~half the true HR
                base_hr_bcg = spectral_hr
                hr_correction_applied = True
                peak_strategy = peak_strategy + "_half_rate_corrected"
            elif abs(ratio - 2.0) <= half_double_tolerance * 2:
                # Double-rate detected: diagnostic is ~double the true HR
                base_hr_bcg = spectral_hr
                hr_correction_applied = True
                peak_strategy = peak_strategy + "_double_rate_corrected"

        block_start_time = float(block["start_time"].iloc[0])
        block_end_time = float(block["end_time"].iloc[-1])

        ecg_hr_bpm = np.nan
        ecg_eval_status = "ecg_disabled"
        if has_ecg and is_complete_block:
            ecg_mask = (source_df["epoch"] >= block_start_time) & (source_df["epoch"] <= block_end_time)
            ecg_segment = source_df.loc[ecg_mask, "ecg_signal"].to_numpy(dtype=np.float64)
            ecg_hr_bpm, ecg_eval_status = estimate_ecg_hr_heartpy(ecg_segment, fs=fs, hr_min=hr_min, hr_max=hr_max)
        elif ecg_missing:
            ecg_eval_status = "ecg_missing"

        rows.append(
            {
                "patient_id": patient_id,
                "block_idx": int(block_idx),
                "block_start_time": block_start_time,
                "block_end_time": block_end_time,
                "n_seconds_in_block": n_seconds,
                "n_bcg_seconds": n_bcg_seconds,
                "bcg_fraction": bcg_fraction,
                "is_complete_block": is_complete_block,
                "n_seconds_with_j_peak": int(valid_bcg_seconds.shape[0]),
                "n_candidate_j_peaks": int(len(block_j_peak_samples)),
                "base_valid_jj_intervals": int(base_valid_jj_count),
                "base_peak_strategy": peak_strategy,
                "max_diagnostic_jj_cv": float(max_diagnostic_jj_cv) if max_diagnostic_jj_cv is not None else np.nan,
                "diagnostic_hr_bpm": float(diagnostic_summary["hr_bpm"]) if np.isfinite(diagnostic_summary["hr_bpm"]) else np.nan,
                "diagnostic_valid_jj_intervals": int(diagnostic_summary["valid_jj_count"]),
                "diagnostic_jj_cv": diagnostic_jj_cv,
                "dense_hr_bpm": float(dense_summary["hr_bpm"]) if np.isfinite(dense_summary["hr_bpm"]) else np.nan,
                "dense_valid_jj_intervals": int(dense_summary["valid_jj_count"]),
                "dense_jj_cv": float(dense_summary["jj_cv"]) if np.isfinite(dense_summary["jj_cv"]) else np.nan,
                "spectral_hr_bpm": float(spectral_hr) if np.isfinite(spectral_hr) else np.nan,
                "spectral_confidence": float(spectral_confidence) if np.isfinite(spectral_confidence) else np.nan,
                "hr_correction_applied": bool(hr_correction_applied),
                "is_motion_artifact": bool(is_motion_artifact),
                "motion_score": float(motion_score),
                "base_hr_bcg_bpm": float(base_hr_bcg) if np.isfinite(base_hr_bcg) else np.nan,
                "hr_ecg_bpm": float(ecg_hr_bpm) if np.isfinite(ecg_hr_bpm) else np.nan,
                "ecg_eval_status": ecg_eval_status,
            }
        )

    result = pd.DataFrame(rows)

    # Adaptive motion artifact gating: flag blocks whose motion score is an
    # outlier relative to the recording's own baseline. This avoids the problem
    # of a fixed threshold rejecting all blocks when the entire recording has
    # uniformly moderate scores (as happens with the 2.5 Hz cutoff issue).
    # A block is flagged only if its score exceeds BOTH:
    #   (a) the user-specified absolute floor (motion_artifact_threshold), AND
    #   (b) the recording's adaptive threshold: median + 2.5 * robust_sigma
    # This ensures only genuinely anomalous blocks get discarded.
    if motion_artifact_threshold > 0 and not result.empty and "motion_score" in result.columns:
        scores = result["motion_score"].to_numpy(dtype=np.float64)
        finite_scores = scores[np.isfinite(scores) & (scores > 0)]
        if finite_scores.size >= 5:
            med_score = float(np.median(finite_scores))
            mad_score = float(np.median(np.abs(finite_scores - med_score)))
            robust_sigma = 1.4826 * mad_score if mad_score > 0 else float(np.std(finite_scores))
            adaptive_threshold = med_score + 2.5 * robust_sigma
            effective_threshold = max(float(motion_artifact_threshold), adaptive_threshold)
        else:
            effective_threshold = float(motion_artifact_threshold)

        is_outlier = result["motion_score"] >= effective_threshold
        result["is_motion_artifact"] = is_outlier
        # Nullify HR for motion-contaminated blocks
        if is_outlier.any():
            result.loc[is_outlier, "base_hr_bcg_bpm"] = np.nan
            result.loc[is_outlier, "base_valid_jj_intervals"] = 0
            result.loc[is_outlier, "base_peak_strategy"] = "motion_artifact_discarded"
            LOGGER.info(
                "%s: flagged %d/%d blocks as motion artifacts (effective_threshold=%.3f, median_score=%.3f)",
                patient_id,
                int(is_outlier.sum()),
                len(result),
                effective_threshold,
                float(np.median(finite_scores)) if finite_scores.size > 0 else 0.0,
            )
    elif "is_motion_artifact" not in result.columns and not result.empty:
        result["is_motion_artifact"] = False

    # Cross-estimator agreement gate: discard blocks where the diagnostic HR
    # disagrees with the spectral fundamental by more than cross_estimator_max_diff bpm.
    # When estimators disagree, the block is ambiguous and the HR estimate unreliable.
    if cross_estimator_max_diff > 0 and not result.empty:
        diag_hr = result["base_hr_bcg_bpm"].to_numpy(dtype=np.float64)
        spec_hr = result["spectral_hr_bpm"].to_numpy(dtype=np.float64)
        both_valid = np.isfinite(diag_hr) & np.isfinite(spec_hr)
        disagreement = np.abs(diag_hr - spec_hr) > cross_estimator_max_diff
        failed_gate = both_valid & disagreement
        # Also flag blocks where spectral is valid but diagnostic is not (low confidence)
        # Don't flag blocks where spectral is NaN — those just lack a second opinion
        if failed_gate.any():
            result.loc[failed_gate, "base_hr_bcg_bpm"] = np.nan
            result.loc[failed_gate, "base_valid_jj_intervals"] = 0
            result.loc[failed_gate, "base_peak_strategy"] = "cross_estimator_disagreement"
            LOGGER.info(
                "%s: rejected %d/%d blocks due to cross-estimator disagreement (max_diff=%.1f bpm)",
                patient_id,
                int(failed_gate.sum()),
                len(result),
                cross_estimator_max_diff,
            )

    # Optional per-subject ECG reference cleaning: heartpy occasionally double-counts
    # beats on a subset of blocks, producing HR that is physiologically valid but far
    # from the subject's own resting rate. Reject those as reference outliers so they
    # do not corrupt the paired metrics.
    if ecg_subject_outlier_mad and ecg_subject_outlier_mad > 0 and not result.empty and "hr_ecg_bpm" in result:
        ecg_vals = result["hr_ecg_bpm"].to_numpy(dtype=np.float64)
        finite = np.isfinite(ecg_vals)
        if int(finite.sum()) >= 10:
            med = float(np.median(ecg_vals[finite]))
            mad = float(np.median(np.abs(ecg_vals[finite] - med)))
            if mad > 0:
                sigma = 1.4826 * mad
                outlier = finite & (np.abs(ecg_vals - med) > float(ecg_subject_outlier_mad) * sigma)
                if outlier.any():
                    result.loc[outlier, "hr_ecg_bpm"] = np.nan
                    result.loc[outlier, "ecg_eval_status"] = "ecg_subject_outlier"
                    LOGGER.info(
                        "%s: rejected %d/%d ECG blocks as subject-level outliers (median=%.1f bpm, k=%.1f)",
                        patient_id,
                        int(outlier.sum()),
                        int(finite.sum()),
                        med,
                        float(ecg_subject_outlier_mad),
                    )

    return result

def apply_block_threshold(block_df: pd.DataFrame, bcg_fraction_threshold: float) -> pd.DataFrame:
    if block_df.empty:
        return block_df.copy()

    out = block_df.copy()
    threshold = float(bcg_fraction_threshold)
    out["bcg_fraction_threshold"] = threshold
    motion_gate = ~out["is_motion_artifact"] if "is_motion_artifact" in out.columns else True
    out["is_valid_bcg_block"] = out["is_complete_block"] & (out["bcg_fraction"] >= threshold) & motion_gate
    out["is_preserved_window"] = out["is_valid_bcg_block"]
    out["peak_strategy"] = np.where(out["is_valid_bcg_block"], out["base_peak_strategy"], "invalid_block")
    out["n_valid_jj_intervals"] = np.where(out["is_valid_bcg_block"], out["base_valid_jj_intervals"], 0).astype(int)
    out["hr_bcg_bpm"] = np.where(out["is_valid_bcg_block"], out["base_hr_bcg_bpm"], np.nan)
    out["paired_hr_available"] = (
        out["is_valid_bcg_block"]
        & out["hr_bcg_bpm"].notna()
        & out["hr_ecg_bpm"].notna()
        & (out["hr_ecg_bpm"].astype(float) != 0.0)
    )
    out["abs_error_bpm"] = np.where(out["paired_hr_available"], (out["hr_bcg_bpm"] - out["hr_ecg_bpm"]).abs(), np.nan)
    out["ape_percent"] = np.where(
        out["paired_hr_available"],
        out["abs_error_bpm"] / out["hr_ecg_bpm"].abs() * 100.0,
        np.nan,
    )

    return out.drop(columns=["base_valid_jj_intervals", "base_peak_strategy", "base_hr_bcg_bpm"])

def compute_error_metrics(hr_df: pd.DataFrame) -> dict:
    paired = hr_df[hr_df["paired_hr_available"] == True].copy()
    if paired.empty:
        return {
            "n_paired_windows": 0,
            "mae_bpm": np.nan,
            "rmse_bpm": np.nan,
            "mape_percent": np.nan,
        }

    diff = paired["hr_bcg_bpm"].to_numpy(dtype=np.float64) - paired["hr_ecg_bpm"].to_numpy(dtype=np.float64)
    abs_pct = paired["ape_percent"].to_numpy(dtype=np.float64)
    return {
        "n_paired_windows": int(len(paired)),
        "mae_bpm": float(np.mean(np.abs(diff))),
        "rmse_bpm": float(np.sqrt(np.mean(np.square(diff)))),
        "mape_percent": float(np.mean(abs_pct)),
    }

def compute_coverage_metrics(hr_df: pd.DataFrame) -> dict:
    if hr_df.empty:
        return {
            "total_windows": 0,
            "preserved_windows": 0,
            "preserved_window_coverage": np.nan,
        }

    total_windows = int(len(hr_df))
    preserved_windows = int(hr_df["is_preserved_window"].sum()) if "is_preserved_window" in hr_df else 0
    preserved_window_coverage = float(preserved_windows / total_windows) if total_windows > 0 else np.nan
    return {
        "total_windows": total_windows,
        "preserved_windows": preserved_windows,
        "preserved_window_coverage": preserved_window_coverage,
    }

def build_patient_summary(all_30s: pd.DataFrame) -> pd.DataFrame:
    if all_30s.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "recording_name",
                "threshold_source",
                "bcg_fraction_threshold",
                "total_windows",
                "preserved_windows",
                "preserved_window_coverage",
                "n_blocks",
                "n_valid_bcg_blocks",
                "valid_block_rate",
                "n_paired_windows",
                "median_hr_bcg_bpm",
                "mean_hr_bcg_bpm",
                "median_hr_ecg_bpm",
                "mean_hr_ecg_bpm",
                "mae_bpm",
                "rmse_bpm",
                "mape_percent",
            ]
        )

    summary_rows = []
    grouped = all_30s.groupby(["patient_id", "recording_name"], dropna=False, sort=True)
    for (patient_id, recording_name), group in grouped:
        metrics = compute_error_metrics(group)
        coverage = compute_coverage_metrics(group)
        n_blocks = int(group["block_idx"].count())
        n_valid_bcg_blocks = int(group["is_valid_bcg_block"].sum())
        threshold_source = "default"
        if "threshold_source" in group.columns and group["threshold_source"].notna().any():
            threshold_source = str(group.loc[group["threshold_source"].notna(), "threshold_source"].iloc[0])
        summary_rows.append(
            {
                "patient_id": patient_id,
                "recording_name": recording_name,
                "threshold_source": threshold_source,
                "bcg_fraction_threshold": float(group["bcg_fraction_threshold"].iloc[0]) if "bcg_fraction_threshold" in group.columns and not group.empty else np.nan,
                "total_windows": coverage["total_windows"],
                "preserved_windows": coverage["preserved_windows"],
                "preserved_window_coverage": coverage["preserved_window_coverage"],
                "n_blocks": n_blocks,
                "n_valid_bcg_blocks": n_valid_bcg_blocks,
                "valid_block_rate": float(n_valid_bcg_blocks / n_blocks) if n_blocks > 0 else np.nan,
                "n_paired_windows": metrics["n_paired_windows"],
                "median_hr_bcg_bpm": float(group["hr_bcg_bpm"].median()) if group["hr_bcg_bpm"].notna().any() else np.nan,
                "mean_hr_bcg_bpm": float(group["hr_bcg_bpm"].mean()) if group["hr_bcg_bpm"].notna().any() else np.nan,
                "median_hr_ecg_bpm": float(group["hr_ecg_bpm"].median()) if group["hr_ecg_bpm"].notna().any() else np.nan,
                "mean_hr_ecg_bpm": float(group["hr_ecg_bpm"].mean()) if group["hr_ecg_bpm"].notna().any() else np.nan,
                "mae_bpm": metrics["mae_bpm"],
                "rmse_bpm": metrics["rmse_bpm"],
                "mape_percent": metrics["mape_percent"],
            }
        )

    return pd.DataFrame(summary_rows)

def build_overall_metrics(all_30s: pd.DataFrame) -> pd.DataFrame:
    metrics = compute_error_metrics(all_30s)
    coverage = compute_coverage_metrics(all_30s)
    if all_30s.empty:
        valid_blocks = 0
    else:
        valid_blocks = int(all_30s["is_valid_bcg_block"].sum())

    return pd.DataFrame(
        [
            {
                "n_recordings": int(all_30s[["patient_id", "recording_name"]].drop_duplicates().shape[0]) if not all_30s.empty else 0,
                "total_windows": coverage["total_windows"],
                "preserved_windows": coverage["preserved_windows"],
                "preserved_window_coverage": coverage["preserved_window_coverage"],
                "n_blocks": coverage["total_windows"],
                "n_valid_bcg_blocks": valid_blocks,
                "n_paired_windows": metrics["n_paired_windows"],
                "mae_bpm": metrics["mae_bpm"],
                "rmse_bpm": metrics["rmse_bpm"],
                "mape_percent": metrics["mape_percent"],
            }
        ]
    )

def build_threshold_sweep_by_recording(sweep_30s_df: pd.DataFrame) -> pd.DataFrame:
    if sweep_30s_df.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "recording_name",
                "bcg_fraction_threshold",
                "total_windows",
                "preserved_windows",
                "preserved_window_coverage",
                "n_valid_bcg_blocks",
                "n_paired_windows",
                "mae_bpm",
                "rmse_bpm",
                "mape_percent",
            ]
        )

    rows = []
    grouped = sweep_30s_df.groupby(["patient_id", "recording_name", "bcg_fraction_threshold"], dropna=False, sort=True)
    for (patient_id, recording_name, threshold), group in grouped:
        coverage = compute_coverage_metrics(group)
        metrics = compute_error_metrics(group)
        rows.append(
            {
                "patient_id": patient_id,
                "recording_name": recording_name,
                "bcg_fraction_threshold": float(threshold),
                "total_windows": coverage["total_windows"],
                "preserved_windows": coverage["preserved_windows"],
                "preserved_window_coverage": coverage["preserved_window_coverage"],
                "n_valid_bcg_blocks": int(group["is_valid_bcg_block"].sum()),
                "n_paired_windows": metrics["n_paired_windows"],
                "mae_bpm": metrics["mae_bpm"],
                "rmse_bpm": metrics["rmse_bpm"],
                "mape_percent": metrics["mape_percent"],
            }
        )
    return pd.DataFrame(rows)

def build_threshold_sweep_overall(sweep_30s_df: pd.DataFrame) -> pd.DataFrame:
    if sweep_30s_df.empty:
        return pd.DataFrame(
            columns=[
                "bcg_fraction_threshold",
                "n_recordings",
                "total_windows",
                "preserved_windows",
                "preserved_window_coverage",
                "n_valid_bcg_blocks",
                "n_paired_windows",
                "mae_bpm",
                "rmse_bpm",
                "mape_percent",
            ]
        )

    rows = []
    grouped = sweep_30s_df.groupby("bcg_fraction_threshold", dropna=False, sort=True)
    for threshold, group in grouped:
        coverage = compute_coverage_metrics(group)
        metrics = compute_error_metrics(group)
        rows.append(
            {
                "bcg_fraction_threshold": float(threshold),
                "n_recordings": int(group[["patient_id", "recording_name"]].drop_duplicates().shape[0]),
                "total_windows": coverage["total_windows"],
                "preserved_windows": coverage["preserved_windows"],
                "preserved_window_coverage": coverage["preserved_window_coverage"],
                "n_valid_bcg_blocks": int(group["is_valid_bcg_block"].sum()),
                "n_paired_windows": metrics["n_paired_windows"],
                "mae_bpm": metrics["mae_bpm"],
                "rmse_bpm": metrics["rmse_bpm"],
                "mape_percent": metrics["mape_percent"],
            }
        )
    return pd.DataFrame(rows)

def process_recording(
    recording_path: Path,
    model,
    calibrator,
    infer_cfg: dict,
    output_dir: Path,
    fs_override: Optional[float],
    window_sec: float,
    stride_sec: float,
    aggregate_sec: int,
    bcg_fraction_threshold: float,
    patient_threshold_overrides: dict[str, float],
    sweep_thresholds: list[float],
    hr_min: float,
    hr_max: float,
    min_valid_jj_intervals: int,
    max_diagnostic_jj_cv: Optional[float],
    disable_ecg_eval: bool,
    use_spectral_check: bool = True,
    spectral_confidence_min: float = 1.5,
    half_double_tolerance: float = 0.18,
    ecg_subject_outlier_mad: float = 0.0,
    disable_filter: bool = False,
    normalize_signal: bool = False,
    motion_artifact_threshold: float = 0.0,
    cross_estimator_max_diff: float = 0.0,
    use_improved_peaks: bool = False,
    refine_diagnostic_peaks: bool = False,
    fuse_estimators: bool = False,
    spectral_conf_scale: float = 500000.0,
    prominence_coef: float = 0.05,
    adaptive_thresh_mult: float = 1.2,
    disable_gate: bool = False,
    block_gate_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patient_id = patient_id_from_path(recording_path)
    LOGGER.info("Processing %s", recording_path.name)

    df = load_recording(recording_path)
    effective_infer_cfg = dict(infer_cfg)
    if fs_override is not None:
        effective_infer_cfg["fs"] = float(fs_override)
    cfg = _config_from_inference(effective_infer_cfg)
    signal_fs = resolve_sampling_rate(df, fallback_fs=float(cfg.fs))

    # Optional z-score normalization before filtering/classification
    if normalize_signal:
        raw = df["raw_data_sleepMat"].to_numpy(dtype=np.float64)
        mu, sigma = raw.mean(), raw.std()
        if sigma > 0:
            df["raw_data_sleepMat"] = (raw - mu) / sigma
        else:
            df["raw_data_sleepMat"] = raw - mu
        LOGGER.debug("Normalized signal: mu=%.4f, sigma=%.4f", mu, sigma)

    # Either apply internal bandpass filter or use pre-filtered signal directly
    if disable_filter:
        filtered_df = df[["epoch", "raw_data_sleepMat"]].copy()
        filtered_df["filtered"] = filtered_df["raw_data_sleepMat"].to_numpy(dtype=np.float32)
    else:
        filtered_df = _filter_full_signal(df[["epoch", "raw_data_sleepMat"]].copy(), cfg)
    X, meta = segments_from_windows(filtered_df, cfg, window_sec=window_sec, stride_sec=stride_sec)
    pred_df = predict_dataframe(model, calibrator, effective_infer_cfg, X, meta)
    if pred_df.empty:
        LOGGER.warning("No inference windows extracted for %s", recording_path.name)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    second_df = build_second_level_table(
        pred_df=pred_df,
        filtered_df=filtered_df,
        fs=signal_fs,
        hr_max=hr_max,
        window_sec=window_sec,
        stride_sec=stride_sec,
        prominence_coef=prominence_coef,
        disable_gate=disable_gate,
        block_gate_only=block_gate_only,
    )
    second_df.insert(0, "patient_id", patient_id)
    second_df.insert(1, "recording_name", recording_path.name)

    block_df = summarize_blocks(
        second_df=second_df,
        filtered_df=filtered_df,
        source_df=df,
        patient_id=patient_id,
        fs=signal_fs,
        aggregate_sec=aggregate_sec,
        hr_min=hr_min,
        hr_max=hr_max,
        min_valid_jj_intervals=min_valid_jj_intervals,
        max_diagnostic_jj_cv=max_diagnostic_jj_cv,
        disable_ecg_eval=disable_ecg_eval,
        use_spectral_check=use_spectral_check,
        spectral_confidence_min=spectral_confidence_min,
        half_double_tolerance=half_double_tolerance,
        ecg_subject_outlier_mad=ecg_subject_outlier_mad,
        motion_artifact_threshold=motion_artifact_threshold,
        cross_estimator_max_diff=cross_estimator_max_diff,
        use_improved_peaks=use_improved_peaks,
        refine_diagnostic_peaks=refine_diagnostic_peaks,
        fuse_estimators=fuse_estimators,
        spectral_conf_scale=spectral_conf_scale,
        adaptive_thresh_mult=adaptive_thresh_mult,
    )
    applied_threshold, threshold_source = resolve_recording_threshold(
        patient_id=patient_id,
        recording_name=recording_path.name,
        default_threshold=bcg_fraction_threshold,
        patient_threshold_overrides=patient_threshold_overrides,
    )
    if threshold_source != "default":
        LOGGER.info(
            "%s: using patient-specific threshold %.2f (%s)",
            patient_id,
            applied_threshold,
            threshold_source,
        )

    hr30_df = apply_block_threshold(block_df, applied_threshold)
    hr30_df.insert(1, "recording_name", recording_path.name)
    hr30_df.insert(2, "threshold_source", threshold_source)

    patient_out_dir = output_dir / patient_id
    patient_out_dir.mkdir(parents=True, exist_ok=True)
    second_df.to_csv(patient_out_dir / f"{patient_id}_per_second_predictions.csv", index=False)
    hr30_df.to_csv(patient_out_dir / f"{patient_id}_per_{aggregate_sec}s_hr.csv", index=False)

    sweep_frames = []
    for threshold in sweep_thresholds:
        sweep_df = apply_block_threshold(block_df, float(threshold))
        sweep_df.insert(1, "recording_name", recording_path.name)
        sweep_frames.append(sweep_df)

    sweep_30s_df = pd.concat(sweep_frames, ignore_index=True) if sweep_frames else pd.DataFrame()
    if not sweep_30s_df.empty:
        recording_sweep_summary = build_threshold_sweep_by_recording(sweep_30s_df)
        recording_sweep_summary.to_csv(patient_out_dir / f"{patient_id}_threshold_sweep_summary.csv", index=False)

    paired_blocks = int(hr30_df["paired_hr_available"].sum()) if not hr30_df.empty else 0
    coverage = compute_coverage_metrics(hr30_df)
    LOGGER.info(
        "%s: %d second-windows, %d %ds blocks, %d preserved windows, coverage=%.4f, %d paired HR windows",
        patient_id,
        len(second_df),
        len(hr30_df),
        aggregate_sec,
        coverage["preserved_windows"],
        coverage["preserved_window_coverage"] if np.isfinite(coverage["preserved_window_coverage"]) else np.nan,
        paired_blocks,
    )
    return second_df, hr30_df, sweep_30s_df


# ------------------------------------------------------------------------------
# Analysis stage (merged from analyze_hr_results.py)
# ------------------------------------------------------------------------------


OPTIONAL_PEAK_STRATEGY_COLUMNS = (
    "bcg_fraction",
    "n_valid_jj_intervals",
    "diagnostic_jj_cv",
    "dense_jj_cv",
)

def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)

def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        LOGGER.info("Optional file not found: %s", path.name)
        return pd.DataFrame()
    return pd.read_csv(path)

def classify_patient_row(row: pd.Series) -> str:
    coverage = float(row.get("preserved_window_coverage", np.nan))
    mae = float(row.get("mae_bpm", np.nan))

    if np.isfinite(coverage) and np.isfinite(mae):
        if coverage >= 0.90 and mae <= 10:
            return "strong"
        if coverage >= 0.90 and mae > 20:
            return "high_coverage_high_error"
        if coverage < 0.80 and mae <= 10:
            return "low_coverage_low_error"
        if coverage < 0.80 and mae > 20:
            return "weak"
    return "mixed"

def safe_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return np.nan
    series = frame[column]
    return float(series.mean()) if series.notna().any() else np.nan

def safe_median(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return np.nan
    series = frame[column]
    return float(series.median()) if series.notna().any() else np.nan

def build_patient_analysis(patient_summary: pd.DataFrame) -> pd.DataFrame:
    if patient_summary.empty:
        return pd.DataFrame()

    df = patient_summary.copy()
    df["coverage_pct"] = df["preserved_window_coverage"] * 100.0
    df["absolute_hr_bias_bpm"] = (df["mean_hr_bcg_bpm"] - df["mean_hr_ecg_bpm"]).abs()
    df["signed_hr_bias_bpm"] = df["mean_hr_bcg_bpm"] - df["mean_hr_ecg_bpm"]
    df["mae_rank"] = df["mae_bpm"].rank(method="min", ascending=False)
    df["rmse_rank"] = df["rmse_bpm"].rank(method="min", ascending=False)
    df["coverage_rank"] = df["preserved_window_coverage"].rank(method="min", ascending=False)
    df["patient_status"] = df.apply(classify_patient_row, axis=1)
    return df.sort_values(["mae_bpm", "rmse_bpm", "preserved_window_coverage"], ascending=[False, False, False])

def build_window_diagnostics(per_30s_hr: pd.DataFrame) -> pd.DataFrame:
    if per_30s_hr.empty:
        return pd.DataFrame()

    rows = []
    grouped = per_30s_hr.groupby(["patient_id", "recording_name"], dropna=False, sort=True)
    for (patient_id, recording_name), group in grouped:
        paired = group[group["paired_hr_available"] == True].copy()
        rows.append(
            {
                "patient_id": patient_id,
                "recording_name": recording_name,
                "n_blocks": int(len(group)),
                "n_preserved_windows": int(group["is_preserved_window"].sum()),
                "n_paired_windows": int(len(paired)),
                "paired_rate": float(len(paired) / len(group)) if len(group) > 0 else np.nan,
                "mean_bcg_fraction": safe_mean(group, "bcg_fraction"),
                "median_bcg_fraction": safe_median(group, "bcg_fraction"),
                "mean_jj_intervals_per_block": safe_mean(group, "n_valid_jj_intervals"),
                "median_jj_intervals_per_block": safe_median(group, "n_valid_jj_intervals"),
                "mean_abs_error_bpm": float(paired["abs_error_bpm"].mean()) if not paired.empty else np.nan,
                "p90_abs_error_bpm": float(paired["abs_error_bpm"].quantile(0.90)) if not paired.empty else np.nan,
                "mean_ape_percent": float(paired["ape_percent"].mean()) if not paired.empty else np.nan,
                "signed_bias_bpm": float((paired["hr_bcg_bpm"] - paired["hr_ecg_bpm"]).mean()) if not paired.empty else np.nan,
                "ecg_ok_blocks": int((group["ecg_eval_status"] == "ok").sum()),
                "ecg_error_blocks": int((group["ecg_eval_status"] == "heartpy_error").sum()),
                "ecg_invalid_bpm_blocks": int((group["ecg_eval_status"] == "heartpy_invalid_bpm").sum()),
                "ecg_too_short_blocks": int((group["ecg_eval_status"] == "ecg_too_short").sum()),
            }
        )

    return pd.DataFrame(rows).sort_values(["mean_abs_error_bpm", "p90_abs_error_bpm"], ascending=[False, False])

def build_bad_patient_peak_strategy(
    per_30s_hr: pd.DataFrame,
    patient_analysis: pd.DataFrame,
    top_bad_patients: int,
) -> pd.DataFrame:
    if per_30s_hr.empty or patient_analysis.empty or "peak_strategy" not in per_30s_hr.columns:
        return pd.DataFrame()

    selected = patient_analysis.sort_values(["mae_bpm", "rmse_bpm"], ascending=[False, False]).head(int(top_bad_patients))
    if selected.empty:
        return pd.DataFrame()

    selected_keys = selected[["patient_id", "recording_name", "mae_bpm", "rmse_bpm", "preserved_window_coverage"]].copy()
    merged = per_30s_hr.merge(selected_keys, on=["patient_id", "recording_name"], how="inner")
    if merged.empty:
        return pd.DataFrame()

    for column in OPTIONAL_PEAK_STRATEGY_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan

    total_blocks = (
        merged.groupby(["patient_id", "recording_name"], dropna=False, sort=False)
        .size()
        .rename("total_blocks")
        .reset_index()
    )
    merged = merged.merge(total_blocks, on=["patient_id", "recording_name"], how="left")

    rows = []
    group_cols = ["patient_id", "recording_name", "mae_bpm", "rmse_bpm", "preserved_window_coverage", "peak_strategy"]
    for keys, group in merged.groupby(group_cols, dropna=False, sort=True):
        patient_id, recording_name, mae_bpm, rmse_bpm, coverage, peak_strategy = keys
        paired = group[group["paired_hr_available"] == True].copy()
        preserved = group[group["is_preserved_window"] == True].copy()
        total_for_recording = int(group["total_blocks"].iloc[0]) if "total_blocks" in group.columns and not group.empty else 0
        rows.append(
            {
                "patient_id": patient_id,
                "recording_name": recording_name,
                "mae_bpm": float(mae_bpm) if np.isfinite(mae_bpm) else np.nan,
                "rmse_bpm": float(rmse_bpm) if np.isfinite(rmse_bpm) else np.nan,
                "preserved_window_coverage": float(coverage) if np.isfinite(coverage) else np.nan,
                "peak_strategy": str(peak_strategy),
                "n_blocks": int(len(group)),
                "block_fraction": float(len(group) / total_for_recording) if total_for_recording > 0 else np.nan,
                "n_preserved_blocks": int(len(preserved)),
                "n_paired_blocks": int(len(paired)),
                "mean_abs_error_bpm": float(paired["abs_error_bpm"].mean()) if not paired.empty else np.nan,
                "median_abs_error_bpm": float(paired["abs_error_bpm"].median()) if not paired.empty else np.nan,
                "mean_bcg_fraction": safe_mean(group, "bcg_fraction"),
                "mean_n_valid_jj_intervals": safe_mean(group, "n_valid_jj_intervals"),
                "mean_diagnostic_jj_cv": safe_mean(group, "diagnostic_jj_cv"),
                "mean_dense_jj_cv": safe_mean(group, "dense_jj_cv"),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["mae_bpm", "n_blocks", "mean_abs_error_bpm"], ascending=[False, False, False])

def _normalize_for_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric[np.isfinite(numeric)]
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)

    lo = float(valid.min())
    hi = float(valid.max())
    if hi == lo:
        out = pd.Series(1.0, index=series.index, dtype=float)
        out[~np.isfinite(numeric)] = np.nan
        return out

    scaled = (numeric - lo) / (hi - lo)
    if not higher_is_better:
        scaled = 1.0 - scaled
    return scaled

def build_threshold_tradeoff(
    threshold_sweep_overall: pd.DataFrame,
    coverage_floor: float,
    paired_floor: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if threshold_sweep_overall.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = threshold_sweep_overall.copy().sort_values("bcg_fraction_threshold").reset_index(drop=True)
    df["coverage_pct"] = df["preserved_window_coverage"] * 100.0
    df["coverage_score"] = _normalize_for_score(df["preserved_window_coverage"], higher_is_better=True)
    df["mae_score"] = _normalize_for_score(df["mae_bpm"], higher_is_better=False)
    df["rmse_score"] = _normalize_for_score(df["rmse_bpm"], higher_is_better=False)
    df["mape_score"] = _normalize_for_score(df["mape_percent"], higher_is_better=False)
    df["composite_score"] = (
        0.40 * df["coverage_score"]
        + 0.25 * df["mae_score"]
        + 0.20 * df["rmse_score"]
        + 0.15 * df["mape_score"]
    )

    df["meets_recommendation_floor"] = (
        (df["preserved_window_coverage"] >= float(coverage_floor))
        & (df["n_paired_windows"] >= int(paired_floor))
    )

    preferred = df[df["meets_recommendation_floor"]].copy()
    if preferred.empty:
        preferred = df.copy()

    recommended = preferred.sort_values(
        ["composite_score", "mae_bpm", "rmse_bpm", "preserved_window_coverage"],
        ascending=[False, True, True, False],
    ).head(1)

    return df.sort_values("composite_score", ascending=False), recommended

def write_executive_summary(
    output_path: Path,
    patient_analysis: pd.DataFrame,
    window_diagnostics: pd.DataFrame,
    threshold_tradeoff: pd.DataFrame,
    recommended_thresholds: pd.DataFrame,
    bad_patient_peak_strategy: pd.DataFrame,
) -> None:
    lines: list[str] = []

    lines.append("BCG/ECG HR results analysis")
    lines.append("=" * 30)
    lines.append("")

    if not patient_analysis.empty:
        lines.append(f"Patients analyzed: {patient_analysis['patient_id'].nunique()}")
        lines.append(f"Mean coverage: {patient_analysis['preserved_window_coverage'].mean():.4f}")
        lines.append(f"Mean MAE: {patient_analysis['mae_bpm'].mean():.4f} bpm")
        lines.append(f"Mean RMSE: {patient_analysis['rmse_bpm'].mean():.4f} bpm")
        lines.append("")

        lines.append("Worst MAE patients:")
        for _, row in patient_analysis.sort_values("mae_bpm", ascending=False).head(5).iterrows():
            lines.append(
                f"  - {row['patient_id']}: MAE={row['mae_bpm']:.4f}, RMSE={row['rmse_bpm']:.4f}, "
                f"coverage={row['preserved_window_coverage']:.4f}, status={row['patient_status']}"
            )
        lines.append("")

    if not window_diagnostics.empty:
        lines.append("Largest signed bias patients:")
        ordered = window_diagnostics.reindex(window_diagnostics["signed_bias_bpm"].abs().sort_values(ascending=False).index)
        for _, row in ordered.head(5).iterrows():
            lines.append(
                f"  - {row['patient_id']}: signed_bias={row['signed_bias_bpm']:.4f} bpm, "
                f"mean_abs_error={row['mean_abs_error_bpm']:.4f}, paired_rate={row['paired_rate']:.4f}"
            )
        lines.append("")

    if not bad_patient_peak_strategy.empty:
        lines.append("Peak strategy mix for worst-MAE patients:")
        ordered = bad_patient_peak_strategy.sort_values(["mae_bpm", "n_blocks"], ascending=[False, False])
        current_key: tuple[str, str] | None = None
        for _, row in ordered.iterrows():
            key = (str(row["patient_id"]), str(row["recording_name"]))
            if key != current_key:
                current_key = key
                lines.append(
                    f"  - {row['patient_id']} ({row['recording_name']}): MAE={row['mae_bpm']:.4f}, "
                    f"coverage={row['preserved_window_coverage']:.4f}"
                )
            lines.append(
                f"      {row['peak_strategy']}: blocks={int(row['n_blocks'])}, paired={int(row['n_paired_blocks'])}, "
                f"mean_abs_error={row['mean_abs_error_bpm']:.4f}"
            )
        lines.append("")

    if not recommended_thresholds.empty:
        row = recommended_thresholds.iloc[0]
        lines.append(
            "Recommended threshold: "
            f"{row['bcg_fraction_threshold']:.2f} "
            f"(coverage={row['preserved_window_coverage']:.4f}, paired={int(row['n_paired_windows'])}, "
            f"MAE={row['mae_bpm']:.4f}, RMSE={row['rmse_bpm']:.4f}, score={row['composite_score']:.4f})"
        )
        lines.append("")

    if not threshold_tradeoff.empty:
        lines.append("Threshold ranking:")
        for _, row in threshold_tradeoff.sort_values("composite_score", ascending=False).iterrows():
            lines.append(
                f"  - thr={row['bcg_fraction_threshold']:.2f}: coverage={row['preserved_window_coverage']:.4f}, "
                f"MAE={row['mae_bpm']:.4f}, RMSE={row['rmse_bpm']:.4f}, paired={int(row['n_paired_windows'])}, "
                f"score={row['composite_score']:.4f}"
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_inference(args: argparse.Namespace) -> None:

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, calibrator, infer_cfg = load_artifacts(args.model_dir)
    infer_cfg["predict_batch_size"] = int(args.predict_batch_size)
    if args.hp_cutoff_hz is not None:
        infer_cfg["hp_cutoff_hz"] = args.hp_cutoff_hz
        LOGGER.info("Overriding hp_cutoff_hz to %.2f Hz", args.hp_cutoff_hz)
    if args.lp_cutoff_hz is not None:
        infer_cfg["lp_cutoff_hz"] = args.lp_cutoff_hz
        LOGGER.info("Overriding lp_cutoff_hz to %.2f Hz", args.lp_cutoff_hz)
    sweep_thresholds = parse_threshold_values(args.sweep_bcg_thresholds, args.bcg_fraction_threshold)
    patient_threshold_overrides = parse_patient_threshold_overrides(args.patient_bcg_fraction_thresholds)

    if patient_threshold_overrides:
        LOGGER.info("Loaded %d patient-specific threshold override(s)", len(patient_threshold_overrides))

    if not args.disable_ecg_eval and hp is None:
        LOGGER.warning("heartpy is not installed; ECG HR evaluation will be unavailable unless you install heartpy or pass --disable_ecg_eval")

    recording_paths = sorted(args.input_dir.glob(args.glob))
    if not recording_paths:
        raise FileNotFoundError(f"No recordings matched {args.glob!r} in {args.input_dir}")

    all_second = []
    all_30s = []
    all_sweep_30s = []
    for recording_path in recording_paths:
        second_df, hr30_df, sweep_30s_df = process_recording(
            recording_path=recording_path,
            model=model,
            calibrator=calibrator,
            infer_cfg=infer_cfg,
            output_dir=args.output_dir,
            fs_override=args.fs,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            aggregate_sec=args.aggregate_sec,
            bcg_fraction_threshold=args.bcg_fraction_threshold,
            patient_threshold_overrides=patient_threshold_overrides,
            sweep_thresholds=sweep_thresholds,
            hr_min=args.hr_min,
            hr_max=args.hr_max,
            min_valid_jj_intervals=args.min_valid_jj_intervals,
            max_diagnostic_jj_cv=args.max_diagnostic_jj_cv,
            disable_ecg_eval=args.disable_ecg_eval,
            use_spectral_check=not args.disable_spectral_check,
            spectral_confidence_min=args.spectral_confidence_min,
            half_double_tolerance=args.half_double_tolerance,
            ecg_subject_outlier_mad=args.ecg_subject_outlier_mad,
            disable_filter=args.disable_filter,
            normalize_signal=args.normalize_signal,
            motion_artifact_threshold=args.motion_artifact_threshold,
            cross_estimator_max_diff=args.cross_estimator_max_diff,
            use_improved_peaks=args.use_improved_peaks,
            refine_diagnostic_peaks=args.refine_diagnostic_peaks,
            fuse_estimators=args.fuse_estimators,
            spectral_conf_scale=args.spectral_conf_scale,
            prominence_coef=args.prominence_coef,
            adaptive_thresh_mult=args.adaptive_thresh_mult,
            disable_gate=args.disable_gate,
            block_gate_only=getattr(args, "block_gate_only", False),
        )
        if not second_df.empty:
            all_second.append(second_df)
        if not hr30_df.empty:
            all_30s.append(hr30_df)
        if not sweep_30s_df.empty:
            all_sweep_30s.append(sweep_30s_df)

    all_second_df = pd.concat(all_second, ignore_index=True) if all_second else pd.DataFrame()
    all_30s_df = pd.concat(all_30s, ignore_index=True) if all_30s else pd.DataFrame()
    patient_summary = build_patient_summary(all_30s_df)
    overall_metrics = build_overall_metrics(all_30s_df)

    all_second_df.to_csv(args.output_dir / "per_second_predictions.csv", index=False)
    all_30s_df.to_csv(args.output_dir / "per_30s_hr.csv", index=False)
    patient_summary.to_csv(args.output_dir / "patient_summary.csv", index=False)
    overall_metrics.to_csv(args.output_dir / "overall_metrics.csv", index=False)

    if all_sweep_30s:
        all_sweep_30s_df = pd.concat(all_sweep_30s, ignore_index=True)
        threshold_sweep_by_recording = build_threshold_sweep_by_recording(all_sweep_30s_df)
        threshold_sweep_overall = build_threshold_sweep_overall(all_sweep_30s_df)
        threshold_sweep_by_recording.to_csv(args.output_dir / "threshold_sweep_by_recording.csv", index=False)
        threshold_sweep_overall.to_csv(args.output_dir / "threshold_sweep_overall.csv", index=False)
        LOGGER.info(
            "Threshold sweep complete for %d thresholds",
            int(threshold_sweep_overall["bcg_fraction_threshold"].nunique()),
        )
    else:
        # Remove stale sweep files from a previous run so the analysis stage does
        # not report threshold metrics computed on outdated data.
        for stale_name in ("threshold_sweep_overall.csv", "threshold_sweep_by_recording.csv"):
            stale_path = args.output_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()
                LOGGER.info("Removed stale %s from a previous run", stale_name)

    LOGGER.info("Finished. Recordings=%d | second_rows=%d | blocks=%d", len(recording_paths), len(all_second_df), len(all_30s_df))
    if not overall_metrics.empty:
        row = overall_metrics.iloc[0]
        LOGGER.info(
            "Overall: preserved=%d / %d | coverage=%.4f | paired=%d | MAE=%.4f | RMSE=%.4f | MAPE=%.4f",
            int(row["preserved_windows"]),
            int(row["total_windows"]),
            float(row["preserved_window_coverage"]) if np.isfinite(row["preserved_window_coverage"]) else np.nan,
            int(row["n_paired_windows"]),
            float(row["mae_bpm"]) if np.isfinite(row["mae_bpm"]) else np.nan,
            float(row["rmse_bpm"]) if np.isfinite(row["rmse_bpm"]) else np.nan,
            float(row["mape_percent"]) if np.isfinite(row["mape_percent"]) else np.nan,
        )


def run_analysis(args: argparse.Namespace) -> None:

    results_dir = args.results_dir
    output_dir = args.output_dir or (results_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_summary = read_csv_required(results_dir / "patient_summary.csv")
    per_30s_hr = read_csv_required(results_dir / "per_30s_hr.csv")
    threshold_sweep_overall = read_csv_optional(results_dir / "threshold_sweep_overall.csv")
    threshold_sweep_by_recording = read_csv_optional(results_dir / "threshold_sweep_by_recording.csv")

    patient_analysis = build_patient_analysis(patient_summary)
    window_diagnostics = build_window_diagnostics(per_30s_hr)
    bad_patient_peak_strategy = build_bad_patient_peak_strategy(
        per_30s_hr=per_30s_hr,
        patient_analysis=patient_analysis,
        top_bad_patients=args.top_bad_patients,
    )
    threshold_tradeoff, recommended_thresholds = build_threshold_tradeoff(
        threshold_sweep_overall=threshold_sweep_overall,
        coverage_floor=args.coverage_floor,
        paired_floor=args.paired_floor,
    )

    patient_analysis.to_csv(output_dir / "patient_analysis.csv", index=False)
    window_diagnostics.to_csv(output_dir / "window_diagnostics.csv", index=False)
    if not bad_patient_peak_strategy.empty:
        bad_patient_peak_strategy.to_csv(output_dir / "bad_patient_peak_strategy.csv", index=False)

    if not threshold_tradeoff.empty:
        threshold_tradeoff.to_csv(output_dir / "threshold_tradeoff.csv", index=False)
    if not recommended_thresholds.empty:
        recommended_thresholds.to_csv(output_dir / "recommended_thresholds.csv", index=False)
    if not threshold_sweep_by_recording.empty:
        threshold_sweep_by_recording.to_csv(output_dir / "threshold_sweep_by_recording_copy.csv", index=False)

    write_executive_summary(
        output_path=output_dir / "executive_summary.txt",
        patient_analysis=patient_analysis,
        window_diagnostics=window_diagnostics,
        threshold_tradeoff=threshold_tradeoff,
        recommended_thresholds=recommended_thresholds,
        bad_patient_peak_strategy=bad_patient_peak_strategy,
    )

    LOGGER.info("Analysis complete. Outputs written to %s", output_dir)
    if not bad_patient_peak_strategy.empty:
        LOGGER.info(
            "Peak-strategy diagnostics written for %d high-MAE patient recordings",
            int(bad_patient_peak_strategy[["patient_id", "recording_name"]].drop_duplicates().shape[0]),
        )
    if not recommended_thresholds.empty:
        row = recommended_thresholds.iloc[0]
        LOGGER.info(
            "Recommended threshold %.2f | coverage=%.4f | paired=%d | MAE=%.4f | RMSE=%.4f",
            float(row["bcg_fraction_threshold"]),
            float(row["preserved_window_coverage"]),
            int(row["n_paired_windows"]),
            float(row["mae_bpm"]),
            float(row["rmse_bpm"]),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BCG/ECG HR inference and analysis end-to-end in one file")
    parser.add_argument("--model_dir", type=Path, required=True, help="Path to trained final_model directory")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing patient .txt/.csv recordings")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to write inference outputs (analysis goes in <output_dir>/analysis by default)")
    parser.add_argument("--glob", type=str, default="*_aligned_data_ecg.txt", help="Input file glob pattern")
    parser.add_argument("--fs", type=float, default=None, help="Override sampling rate if needed")
    parser.add_argument("--hp_cutoff_hz", type=float, default=None, help="Override high-pass cutoff frequency in Hz (model default: 2.5)")
    parser.add_argument("--lp_cutoff_hz", type=float, default=None, help="Override low-pass cutoff frequency in Hz (model default: 5.0)")
    parser.add_argument("--window_sec", type=float, default=1.0, help="Inference window length in seconds")
    parser.add_argument("--stride_sec", type=float, default=1.0, help="Inference stride in seconds")
    parser.add_argument("--aggregate_sec", type=int, default=30, help="Aggregation block size in seconds")
    parser.add_argument("--bcg_fraction_threshold", type=float, default=0.90, help="Minimum BCG fraction required in a 30-second block")
    parser.add_argument("--block_gate_only", action="store_true", help="Ablation: keep BLOCK-level quality gating (bcg_fraction vs beta) but do NOT restrict J-peak detection to gate-accepted seconds within a retained block. Isolates block-level gating from interval-level restriction.")
    parser.add_argument("--disable_gate", action="store_true", help="Ablation: bypass the quality gate by marking every 1-second interval usable (is_bcg=1). Disables both block-level BCG-fraction gating and interval-level restriction of J-peak detection. All other parameters unchanged.")
    parser.add_argument("--sweep_bcg_thresholds", type=str, default="", help="Optional comma-separated 30-second BCG fraction thresholds to evaluate")
    parser.add_argument("--patient_bcg_fraction_thresholds", type=str, default="", help="Optional comma-separated patient/recording specific thresholds")
    parser.add_argument("--hr_min", type=float, default=40.0, help="Minimum physiologic HR (also gates the ECG reference)")
    parser.add_argument("--hr_max", type=float, default=150.0, help="Maximum physiologic HR (also gates the ECG reference)")
    parser.add_argument("--min_valid_jj_intervals", type=int, default=1, help="Minimum valid J-J intervals to report BCG HR")
    parser.add_argument("--max_diagnostic_jj_cv", type=float, default=None, help="Optional maximum allowed diagnostic J-J CV before falling back")
    parser.add_argument("--disable_spectral_check", action="store_true", help="Disable the autocorrelation/FFT fallback estimator")
    parser.add_argument("--spectral_confidence_min", type=float, default=1.5, help="Minimum FFT peak-to-median PSD ratio to trust the autocorrelation fallback")
    parser.add_argument("--half_double_tolerance", type=float, default=0.18, help="Relative tolerance for half-rate/double-rate correction vs spectral HR (0 disables). If diagnostic/spectral ratio is within 0.5±tol, snaps to spectral.")
    parser.add_argument("--ecg_subject_outlier_mad", type=float, default=0.0, help="If > 0, reject ECG blocks deviating from the subject robust median by more than this many robust sigmas (1.4826*MAD). 0 disables (default). Try 3.0-3.5.")
    parser.add_argument("--disable_filter", action="store_true", help="Skip the internal bandpass filter (use raw_data_sleepMat directly as 'filtered'). Use when input is already pre-filtered.")
    parser.add_argument("--normalize_signal", action="store_true", help="Z-score normalize raw_data_sleepMat before filtering/classification (zero mean, unit variance)")
    parser.add_argument("--motion_artifact_threshold", type=float, default=0.0, help="Motion artifact score threshold (0-1). Blocks scoring above this are discarded before HR estimation. 0 disables (default). Try 0.35-0.50.")
    parser.add_argument("--cross_estimator_max_diff", type=float, default=0.0, help="Max allowed difference (bpm) between diagnostic J-J HR and spectral fundamental. Blocks exceeding this are discarded. 0 disables (default). Try 10.")
    parser.add_argument("--use_improved_peaks", action="store_true", help="Use the improved 3-phase J-peak detector (adaptive threshold + template matching + I/J/K disambiguation) as the primary HR estimator instead of the per-second diagnostic.")
    parser.add_argument("--refine_diagnostic_peaks", action="store_true", help="Refine per-second diagnostic peaks using cross-beat rhythm consistency and mode-seeking interval aggregation. Works within the diagnostic estimator to reduce error without losing coverage.")
    parser.add_argument("--fuse_estimators", action="store_true", help="Fuse the diagnostic, dense, and spectral HR estimates via confidence weighting (inverse J-J CV for peak estimators, scaled PSD ratio for spectral) instead of picking a single estimator. Off by default.")
    parser.add_argument("--spectral_conf_scale", type=float, default=500000.0, help="Scale used to map spectral_confidence to a 0-1 fusion weight (weight=min(confidence/scale, 1)). Only used with --fuse_estimators.")
    parser.add_argument("--disable_ecg_eval", action="store_true", help="Skip ECG HR estimation and downstream paired metrics")
    parser.add_argument("--analysis_output_dir", type=Path, default=None, help="Optional analysis output directory. Defaults to <output_dir>/analysis")
    parser.add_argument("--coverage_floor", type=float, default=0.80, help="Minimum preserved-window coverage when recommending thresholds")
    parser.add_argument("--paired_floor", type=int, default=100, help="Minimum paired windows when recommending thresholds")
    parser.add_argument("--top_bad_patients", type=int, default=5, help="Number of highest-MAE patients in peak-strategy diagnostics")
    parser.add_argument("--skip_analysis", action="store_true", help="Run only inference and skip analysis")
    parser.add_argument("--prominence_coef", type=float, default=0.05, help="Prominence coefficient (x signal std) for the per-second diagnostic J-peak detector. Default 0.05.")
    parser.add_argument("--adaptive_thresh_mult", type=float, default=1.2, help="Adaptive-threshold multiplier for the improved 3-phase detector (x local median envelope). Default 1.2. Only affects --use_improved_peaks.")
    # ---- GPU / performance ----
    parser.add_argument("--disable_gpu", action="store_true", help="Force CPU execution even if a GPU is available.")
    parser.add_argument("--mixed_precision", action="store_true", help="Enable TensorFlow mixed_float16 precision for faster GPU inference (modern NVIDIA GPUs).")
    parser.add_argument("--predict_batch_size", type=int, default=256, help="Batch size for Keras model.predict. Larger values improve GPU throughput (default 256; Keras default is 32).")
    # ---- Grid search (ECG-referenced hyperparameter optimization) ----
    parser.add_argument("--grid_search", action="store_true", help="Run an ECG-referenced grid search over detection/gating params instead of a single inference pass. Minimizes the objective metric subject to a coverage floor, with leave-one-subject-out scoring when >1 subject is present.")
    parser.add_argument("--grid_objective", type=str, default="mae", choices=["mae", "rmse"], help="Objective to minimize during grid search (subject to coverage floor). Default mae.")
    parser.add_argument("--grid_coverage_floor", type=float, default=0.80, help="Minimum mean preserved-window coverage a grid config must meet to be eligible. Default 0.80.")
    parser.add_argument("--grid_prominence_coef", type=str, default="0.03,0.05,0.08,0.12", help="Comma-separated prominence_coef values to search.")
    parser.add_argument("--grid_adaptive_thresh_mult", type=str, default="1.0,1.2,1.5", help="Comma-separated adaptive_thresh_mult values to search (only meaningful with --use_improved_peaks).")
    parser.add_argument("--grid_bcg_fraction_threshold", type=str, default="0.30,0.50,0.70,0.90", help="Comma-separated bcg_fraction_threshold values to search.")
    parser.add_argument("--grid_min_valid_jj_intervals", type=str, default="1,3", help="Comma-separated min_valid_jj_intervals values to search.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


def _parse_grid_floats(spec: str) -> list[float]:
    return [float(tok) for tok in str(spec).split(",") if tok.strip()]


def _parse_grid_ints(spec: str) -> list[int]:
    return [int(float(tok)) for tok in str(spec).split(",") if tok.strip()]


def prepare_recording_for_grid(
    recording_path: Path,
    model,
    calibrator,
    infer_cfg: dict,
    args: argparse.Namespace,
) -> Optional[dict]:
    """Run the expensive, param-independent stage once: load, filter, classify.

    Returns a cache dict (df, filtered_df, pred_df, signal_fs, patient_id) reused
    across all grid combinations, so model inference is not repeated per combo.
    """
    patient_id = patient_id_from_path(recording_path)
    df = load_recording(recording_path)
    effective_infer_cfg = dict(infer_cfg)
    if args.fs is not None:
        effective_infer_cfg["fs"] = float(args.fs)
    cfg = _config_from_inference(effective_infer_cfg)
    signal_fs = resolve_sampling_rate(df, fallback_fs=float(cfg.fs))

    if args.normalize_signal:
        raw = df["raw_data_sleepMat"].to_numpy(dtype=np.float64)
        mu, sigma = raw.mean(), raw.std()
        df["raw_data_sleepMat"] = (raw - mu) / sigma if sigma > 0 else raw - mu

    if args.disable_filter:
        filtered_df = df[["epoch", "raw_data_sleepMat"]].copy()
        filtered_df["filtered"] = filtered_df["raw_data_sleepMat"].to_numpy(dtype=np.float32)
    else:
        filtered_df = _filter_full_signal(df[["epoch", "raw_data_sleepMat"]].copy(), cfg)

    X, meta = segments_from_windows(filtered_df, cfg, window_sec=args.window_sec, stride_sec=args.stride_sec)
    pred_df = predict_dataframe(model, calibrator, effective_infer_cfg, X, meta)
    if pred_df.empty:
        LOGGER.warning("No inference windows extracted for %s (skipping in grid search)", recording_path.name)
        return None
    return {
        "patient_id": patient_id,
        "recording_name": recording_path.name,
        "df": df,
        "filtered_df": filtered_df,
        "pred_df": pred_df,
        "signal_fs": signal_fs,
    }


def evaluate_grid_combo_on_recording(
    prepared: dict,
    args: argparse.Namespace,
    prominence_coef: float,
    adaptive_thresh_mult: float,
    bcg_fraction_threshold: float,
    min_valid_jj_intervals: int,
) -> dict:
    """Run the param-dependent detection + gating for one combo on one recording.

    Returns per-recording MAE/RMSE/coverage/n_paired for this parameter set.
    """
    signal_fs = prepared["signal_fs"]
    second_df = build_second_level_table(
        pred_df=prepared["pred_df"],
        filtered_df=prepared["filtered_df"],
        fs=signal_fs,
        hr_max=args.hr_max,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        prominence_coef=prominence_coef,
        disable_gate=bool(getattr(args, "disable_gate", False)),
        block_gate_only=bool(getattr(args, "block_gate_only", False)),
    )
    second_df.insert(0, "patient_id", prepared["patient_id"])
    second_df.insert(1, "recording_name", prepared["recording_name"])

    block_df = summarize_blocks(
        second_df=second_df,
        filtered_df=prepared["filtered_df"],
        source_df=prepared["df"],
        patient_id=prepared["patient_id"],
        fs=signal_fs,
        aggregate_sec=args.aggregate_sec,
        hr_min=args.hr_min,
        hr_max=args.hr_max,
        min_valid_jj_intervals=min_valid_jj_intervals,
        max_diagnostic_jj_cv=args.max_diagnostic_jj_cv,
        disable_ecg_eval=args.disable_ecg_eval,
        use_spectral_check=not args.disable_spectral_check,
        spectral_confidence_min=args.spectral_confidence_min,
        half_double_tolerance=args.half_double_tolerance,
        ecg_subject_outlier_mad=args.ecg_subject_outlier_mad,
        motion_artifact_threshold=args.motion_artifact_threshold,
        cross_estimator_max_diff=args.cross_estimator_max_diff,
        use_improved_peaks=args.use_improved_peaks,
        refine_diagnostic_peaks=args.refine_diagnostic_peaks,
        fuse_estimators=args.fuse_estimators,
        spectral_conf_scale=args.spectral_conf_scale,
        adaptive_thresh_mult=adaptive_thresh_mult,
    )
    hr30_df = apply_block_threshold(block_df, float(bcg_fraction_threshold))
    metrics = compute_error_metrics(hr30_df)
    coverage = compute_coverage_metrics(hr30_df)
    return {
        "patient_id": prepared["patient_id"],
        "recording_name": prepared["recording_name"],
        "mae_bpm": metrics["mae_bpm"],
        "rmse_bpm": metrics["rmse_bpm"],
        "n_paired_windows": metrics["n_paired_windows"],
        "preserved_window_coverage": coverage["preserved_window_coverage"],
    }


def run_grid_search(args: argparse.Namespace) -> None:
    """ECG-referenced grid search over detection/gating params.

    Objective: minimize `args.grid_objective` (MAE or RMSE) subject to mean
    preserved-window coverage >= args.grid_coverage_floor. When more than one
    subject is present, selection uses leave-one-subject-out (LOSO) CV so the
    reported error is on held-out subjects (guards against overfitting the
    tuning to the evaluation labels). With a single subject, scoring is
    in-sample and a warning is logged.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, calibrator, infer_cfg = load_artifacts(args.model_dir)
    infer_cfg["predict_batch_size"] = int(args.predict_batch_size)
    if args.hp_cutoff_hz is not None:
        infer_cfg["hp_cutoff_hz"] = args.hp_cutoff_hz
    if args.lp_cutoff_hz is not None:
        infer_cfg["lp_cutoff_hz"] = args.lp_cutoff_hz

    recording_paths = sorted(args.input_dir.glob(args.glob))
    if not recording_paths:
        raise FileNotFoundError(f"No recordings matched {args.glob!r} in {args.input_dir}")

    prom_grid = _parse_grid_floats(args.grid_prominence_coef)
    at_grid = _parse_grid_floats(args.grid_adaptive_thresh_mult) if args.use_improved_peaks else [args.adaptive_thresh_mult]
    bt_grid = _parse_grid_floats(args.grid_bcg_fraction_threshold)
    mj_grid = _parse_grid_ints(args.grid_min_valid_jj_intervals)
    combos = list(itertools.product(prom_grid, at_grid, bt_grid, mj_grid))
    LOGGER.info(
        "Grid search: %d recordings x %d combos (prominence=%s, adaptive_thresh=%s, bcg_fraction=%s, min_jj=%s)",
        len(recording_paths), len(combos), prom_grid, at_grid, bt_grid, mj_grid,
    )
    if not args.use_improved_peaks:
        LOGGER.info("adaptive_thresh_mult is inert without --use_improved_peaks; grid pinned to %.3f", args.adaptive_thresh_mult)

    # Prepare each recording once (expensive inference reused across combos).
    prepared_list = []
    for rp in recording_paths:
        prep = prepare_recording_for_grid(rp, model, calibrator, infer_cfg, args)
        if prep is not None:
            prepared_list.append(prep)
    if not prepared_list:
        raise RuntimeError("No usable recordings after preparation; aborting grid search.")

    obj_key = "mae_bpm" if args.grid_objective == "mae" else "rmse_bpm"
    # metrics_by_combo[combo_idx][patient_id] = per-recording metric dict
    metrics_by_combo: dict[int, dict[str, dict]] = {}
    result_rows = []
    for ci, (pc, at, bt, mj) in enumerate(combos):
        per_subject = {}
        for prep in prepared_list:
            res = evaluate_grid_combo_on_recording(prep, args, pc, at, bt, mj)
            per_subject[prep["patient_id"]] = res
        metrics_by_combo[ci] = per_subject

        maes = [r["mae_bpm"] for r in per_subject.values() if np.isfinite(r["mae_bpm"])]
        rmses = [r["rmse_bpm"] for r in per_subject.values() if np.isfinite(r["rmse_bpm"])]
        covs = [r["preserved_window_coverage"] for r in per_subject.values() if np.isfinite(r["preserved_window_coverage"])]
        paired = [r["n_paired_windows"] for r in per_subject.values()]
        result_rows.append({
            "prominence_coef": pc,
            "adaptive_thresh_mult": at,
            "bcg_fraction_threshold": bt,
            "min_valid_jj_intervals": mj,
            "mean_mae_bpm": float(np.mean(maes)) if maes else np.nan,
            "mean_rmse_bpm": float(np.mean(rmses)) if rmses else np.nan,
            "mean_coverage": float(np.mean(covs)) if covs else np.nan,
            "total_paired_windows": int(np.sum(paired)),
            "n_subjects_scored": len(maes),
        })
        LOGGER.info(
            "combo %d/%d: prom=%.3f at=%.2f bt=%.2f mj=%d -> mean_MAE=%.3f mean_cov=%.3f",
            ci + 1, len(combos), pc, at, bt, mj,
            result_rows[-1]["mean_mae_bpm"], result_rows[-1]["mean_coverage"],
        )

    results_df = pd.DataFrame(result_rows)
    results_path = args.output_dir / "grid_search_results.csv"
    results_df.to_csv(results_path, index=False)
    LOGGER.info("Wrote %s (%d combos)", results_path, len(results_df))

    subjects = [prep["patient_id"] for prep in prepared_list]
    floor = float(args.grid_coverage_floor)

    def mean_metric(combo_idx: int, subject_subset: list[str], key: str) -> float:
        vals = [metrics_by_combo[combo_idx][s][key] for s in subject_subset if s in metrics_by_combo[combo_idx]]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else np.nan

    def best_combo_on(subject_subset: list[str]) -> Optional[int]:
        eligible = []
        for ci in range(len(combos)):
            cov = mean_metric(ci, subject_subset, "preserved_window_coverage")
            obj = mean_metric(ci, subject_subset, obj_key)
            if np.isfinite(obj) and np.isfinite(cov) and cov >= floor:
                eligible.append((obj, ci))
        if not eligible:  # coverage floor unreachable; fall back to best objective outright
            fallback = [(mean_metric(ci, subject_subset, obj_key), ci) for ci in range(len(combos))]
            fallback = [(o, ci) for o, ci in fallback if np.isfinite(o)]
            if not fallback:
                return None
            return min(fallback, key=lambda t: t[0])[1]
        return min(eligible, key=lambda t: t[0])[1]

    loso_used = len(subjects) > 1
    loso_records = []
    if loso_used:
        held_out_objs = []
        for held in subjects:
            train = [s for s in subjects if s != held]
            best_ci = best_combo_on(train)
            if best_ci is None:
                continue
            held_obj = metrics_by_combo[best_ci][held][obj_key]
            held_cov = metrics_by_combo[best_ci][held]["preserved_window_coverage"]
            pc, at, bt, mj = combos[best_ci]
            loso_records.append({
                "held_out_subject": held,
                "chosen_prominence_coef": pc,
                "chosen_adaptive_thresh_mult": at,
                "chosen_bcg_fraction_threshold": bt,
                "chosen_min_valid_jj_intervals": mj,
                f"held_out_{obj_key}": held_obj,
                "held_out_coverage": held_cov,
            })
            if np.isfinite(held_obj):
                held_out_objs.append(held_obj)
        loso_df = pd.DataFrame(loso_records)
        loso_path = args.output_dir / "grid_search_loso.csv"
        loso_df.to_csv(loso_path, index=False)
        loso_mean = float(np.mean(held_out_objs)) if held_out_objs else np.nan
        LOGGER.info("LOSO mean held-out %s = %.3f (across %d folds); wrote %s", obj_key, loso_mean, len(held_out_objs), loso_path)
    else:
        loso_mean = np.nan
        LOGGER.warning("Only 1 subject present: LOSO not possible. Selection is IN-SAMPLE and overfits the tuning to the evaluation labels.")

    # Recommended config = best on ALL subjects (subject to coverage floor).
    best_ci_all = best_combo_on(subjects)
    if best_ci_all is None:
        LOGGER.error("No grid combo produced a finite objective; nothing to recommend.")
        return
    pc, at, bt, mj = combos[best_ci_all]
    best = {
        "objective_metric": args.grid_objective,
        "coverage_floor": floor,
        "loso_cross_validated": loso_used,
        "loso_mean_held_out_objective": loso_mean if np.isfinite(loso_mean) else None,
        "recommended_config": {
            "prominence_coef": pc,
            "adaptive_thresh_mult": at if args.use_improved_peaks else None,
            "bcg_fraction_threshold": bt,
            "min_valid_jj_intervals": mj,
        },
        "recommended_config_in_sample": {
            f"mean_{obj_key}": mean_metric(best_ci_all, subjects, obj_key),
            "mean_coverage": mean_metric(best_ci_all, subjects, "preserved_window_coverage"),
        },
        "n_subjects": len(subjects),
        "n_combos": len(combos),
    }
    best_path = args.output_dir / "grid_search_best.json"
    best_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    LOGGER.info(
        "Recommended: prominence_coef=%.3f%s bcg_fraction_threshold=%.2f min_valid_jj_intervals=%d (in-sample mean %s=%.3f, coverage=%.3f)",
        pc,
        f" adaptive_thresh_mult={at:.2f}" if args.use_improved_peaks else "",
        bt, mj, obj_key,
        best["recommended_config_in_sample"][f"mean_{obj_key}"],
        best["recommended_config_in_sample"]["mean_coverage"],
    )
    LOGGER.info("Grid search complete. Results: %s | Best: %s", results_path, best_path)


def configure_gpu(disable_gpu: bool = False, mixed_precision: bool = False) -> None:
    """Configure TensorFlow to use the GPU when available.

    - Enables per-GPU memory growth (prevents TF from grabbing all VRAM up front,
      which otherwise causes OOM/crashes and lets other processes share the card).
    - Optionally enables mixed_float16 precision for faster inference on modern GPUs.
    - Logs whether a GPU is actually in use so slow CPU runs are obvious.

    Must be called before the model is loaded / any TF op runs (memory growth and
    visible-device settings are only honored before GPU initialization).
    """
    try:
        import tensorflow as tf
    except Exception as exc:  # tensorflow not importable
        LOGGER.warning("TensorFlow not importable; GPU configuration skipped (%s)", exc)
        return

    gpus = tf.config.list_physical_devices("GPU")

    if disable_gpu:
        try:
            tf.config.set_visible_devices([], "GPU")
            LOGGER.info("GPU disabled by --disable_gpu; running on CPU.")
        except Exception as exc:
            LOGGER.warning("Could not disable GPU: %s", exc)
        return

    if not gpus:
        LOGGER.warning(
            "No GPU detected by TensorFlow; running on CPU. For GPU acceleration install a "
            "CUDA-enabled tensorflow build with matching CUDA/cuDNN drivers (nvidia-smi should list the card)."
        )
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            LOGGER.warning("set_memory_growth failed for %s: %s", gpu, exc)

    if mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            LOGGER.info("Enabled mixed_float16 precision for faster GPU inference.")
        except Exception as exc:
            LOGGER.warning("Could not enable mixed precision: %s", exc)

    LOGGER.info("GPU acceleration ACTIVE: %d device(s) -> %s", len(gpus), [g.name for g in gpus])


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    configure_gpu(disable_gpu=args.disable_gpu, mixed_precision=args.mixed_precision)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "grid_search", False):
        LOGGER.info("Running ECG-referenced grid search into %s", args.output_dir)
        run_grid_search(args)
        LOGGER.info("Grid search finished (no single-pass inference/analysis performed).")
        return
    LOGGER.info("Stage 1/2: running model-gated HR inference into %s", args.output_dir)
    run_inference(args)
    if args.skip_analysis:
        LOGGER.info("Inference complete. Analysis skipped (--skip_analysis).")
        return
    analysis_args = argparse.Namespace(
        results_dir=args.output_dir,
        output_dir=args.analysis_output_dir,
        coverage_floor=args.coverage_floor,
        paired_floor=args.paired_floor,
        top_bad_patients=args.top_bad_patients,
        log_level=args.log_level,
    )
    LOGGER.info("Stage 2/2: running HR results analysis")
    run_analysis(analysis_args)
    analysis_dir = args.analysis_output_dir or (args.output_dir / "analysis")
    LOGGER.info("Pipeline complete. Analysis outputs in %s", analysis_dir)


if __name__ == "__main__":
    main()
