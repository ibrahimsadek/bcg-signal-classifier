#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
patch5_example_windows.py
==========================
Addresses Reviewer 1 comment #2 (and 2a): "A validation may be performed of how
well the classifier identifies the acceptable BCG windows in the external
dataset using manual inspection, example windows identified as acceptable or
not acceptable may be included in the paper. The authors may include 30 seconds
windows for BCG and non-BCG windows."

Selects exemplar 30-SECOND segments from the external recordings, stratified by
the gate's block-level BCG fraction, and plots them with per-second accept/reject
shading so a reader (or reviewer) can visually audit what the gate is doing.

SELECTION IS BLINDED-BY-CONSTRUCTION AND REPRODUCIBLE
------------------------------------------------------
Segments are chosen by rank within BCG-fraction strata using a fixed seed, not
hand-picked. That matters: cherry-picked exemplars are worthless as validation
evidence, and a referee will assume cherry-picking unless the selection rule is
stated. The rule used here is printed into the summary file so it can go
straight into the figure caption.

NOTE ON WHAT THIS DOES AND DOES NOT ESTABLISH
----------------------------------------------
This is a QUALITATIVE audit aid. It does not substitute for external ground-truth
annotation -- which the manuscript correctly lists as an outstanding limitation.
If you want it to count as validation rather than illustration, have a second
reader score the exported segments blind and report agreement (Cohen's kappa);
--export_blind writes an unlabelled set plus a hidden key for exactly that.

USAGE (from the repo root):

    python patch5_example_windows.py
    python patch5_example_windows.py --n_per_stratum 3 --export_blind

OUTPUTS (into --out_dir):
    patch5_example_windows.png/.pdf    figure: accepted vs rejected 30-s segments
    patch5_selected_segments.csv       exactly which segments were plotted
    patch5_summary.txt                 selection rule, for the caption
    patch5_blind/                      (with --export_blind) unlabelled segments
                                        + key.csv for a blinded second-reader study
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# BCG-fraction strata to sample exemplars from
STRATA = [
    ("clean",      0.95, 1.01, "Accepted - high BCG fraction"),
    ("borderline", 0.45, 0.60, "Borderline - near the beta=0.50 boundary"),
    ("degraded",   0.00, 0.20, "Rejected - low BCG fraction"),
]


def load_recording(path: Path) -> pd.DataFrame:
    """Load a raw aligned recording. Mirrors run_hr_pipeline's loader loosely."""
    for sep in ["\t", ",", r"\s+"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    raise SystemExit(f"Could not parse {path}")


def pick_signal_column(df: pd.DataFrame) -> str:
    for cand in ["raw_data_sleepMat", "filtered", "bcg", "BCG", "signal"]:
        if cand in df.columns:
            return cand
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    num = [c for c in num if "epoch" not in c.lower() and "time" not in c.lower()]
    if not num:
        raise SystemExit(f"No numeric signal column found. Columns: {list(df.columns)}")
    return num[0]


def detect_time_scale(epoch: np.ndarray, fs_hint: float) -> tuple[float, str]:
    """Infer whether an epoch/timestamp column is in milliseconds or seconds.

    Uses the median inter-sample interval, which is unambiguous: at 50 Hz a
    millisecond column steps by ~20 and a second column by ~0.02.

    Returns (scale, unit_name) where scale = raw units per second, i.e.
    seconds = raw / scale.
    """
    if epoch.size < 3:
        return (1.0, "s")
    dt = float(np.median(np.diff(epoch[: min(len(epoch), 10000)])))
    if dt <= 0 or not np.isfinite(dt):
        return (1.0, "s")
    expected_s = 1.0 / fs_hint if fs_hint > 0 else 0.02
    # ratio of observed step to the step we'd expect if the column were seconds
    if dt > 50 * expected_s:          # far too big for seconds -> milliseconds
        return (1000.0, "ms")
    return (1.0, "s")


def align_block_start(t0_raw: float, epoch: np.ndarray, scale: float) -> float | None:
    """Put a block start-time into the same units as the recording's epoch column.

    per_30s_hr.csv and the raw recording normally share units, but this guards
    against the case where one is in seconds and the other in milliseconds.
    Returns the start time in RAW epoch units, or None if it cannot be matched.
    """
    lo, hi = float(epoch.min()), float(epoch.max())
    for candidate in (t0_raw, t0_raw * 1000.0, t0_raw / 1000.0):
        if lo <= candidate <= hi:
            return candidate
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Export example accepted/rejected 30-s windows")
    ap.add_argument("--per_30s_hr", type=Path,
                    default=Path("grid_search") / "per_30s_hr.csv")
    ap.add_argument("--per_second", type=Path,
                    default=Path("grid_search") / "per_second_predictions.csv")
    ap.add_argument("--input_dir", type=Path, default=Path("NewData_processed"),
                    help="Directory holding the raw Sub*_aligned_data_ecg.txt recordings")
    ap.add_argument("--out_dir", type=Path, default=Path("figures"))
    ap.add_argument("--n_per_stratum", type=int, default=2)
    ap.add_argument("--fs", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--export_blind", action="store_true",
                    help="Also export unlabelled segments + key.csv for a blinded reader study")
    args = ap.parse_args()

    for p in (args.per_30s_hr, args.per_second):
        if not p.exists():
            raise SystemExit(f"{p} not found -- pass the path explicitly.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    blocks = pd.read_csv(args.per_30s_hr)
    frac_col = next((c for c in ["bcg_fraction", "mean_bcg_fraction", "block_bcg_fraction"]
                     if c in blocks.columns), None)
    if frac_col is None:
        raise SystemExit(
            f"No BCG-fraction column in {args.per_30s_hr}. "
            f"Columns present: {list(blocks.columns)}"
        )
    start_col = next((c for c in ["block_start_time", "start_time", "block_start"]
                      if c in blocks.columns), None)
    if start_col is None:
        raise SystemExit(
            f"No block start-time column in {args.per_30s_hr}. "
            f"Columns present: {list(blocks.columns)}"
        )

    rng = np.random.default_rng(args.seed)
    chosen = []
    for name, lo, hi, label in STRATA:
        pool = blocks[(blocks[frac_col] >= lo) & (blocks[frac_col] < hi)]
        if pool.empty:
            print(f"  [warn] stratum '{name}' [{lo},{hi}) is empty -- skipping")
            continue
        take = min(args.n_per_stratum, len(pool))
        idx = rng.choice(pool.index.to_numpy(), size=take, replace=False)
        for i in idx:
            r = pool.loc[i]
            chosen.append(dict(
                stratum=name, stratum_label=label,
                patient_id=r["patient_id"],
                block_start=float(r[start_col]),
                bcg_fraction=float(r[frac_col]),
                hr_bcg=float(r["hr_bcg_bpm"]) if "hr_bcg_bpm" in r else np.nan,
                hr_ecg=float(r["hr_ecg_bpm"]) if "hr_ecg_bpm" in r else np.nan,
            ))

    if not chosen:
        raise SystemExit("No segments selected -- check the BCG-fraction strata.")
    sel = pd.DataFrame(chosen)

    # per-second mask for shading
    ps = pd.read_csv(args.per_second,
                     usecols=lambda c: c in ("patient_id", "start_time", "end_time",
                                             "pred_index", "prob_BCG", "is_bcg"))

    n = len(sel)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.9 * nrows), squeeze=False)

    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    plotted = []
    for k, (_, r) in enumerate(sel.iterrows()):
        ax = axes[k // ncols][k % ncols]
        pid = str(r["patient_id"])
        sub = re.search(r"(Sub\d+)", pid)
        stem = sub.group(1) if sub else pid
        candidates = list(args.input_dir.glob(f"{stem}*aligned_data_ecg.txt"))
        if not candidates:
            ax.text(0.5, 0.5, f"raw file for {stem} not found\nin {args.input_dir}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        if stem not in cache:
            raw = load_recording(candidates[0])
            col = pick_signal_column(raw)
            sig = raw[col].to_numpy(dtype=float)
            if "epoch" in raw.columns:
                t = raw["epoch"].to_numpy(dtype=float)
            else:
                # no timestamp column: synthesise one in seconds
                t = np.arange(len(sig)) / args.fs
            scale, unit = detect_time_scale(t, args.fs)
            cache[stem] = (t, sig, scale, unit)
            print(f"  {stem}: {len(sig):,} samples, epoch in {unit} "
                  f"(range {t.min():.0f}..{t.max():.0f}, "
                  f"duration {(t.max()-t.min())/scale/3600:.2f} h)")
        t, sig, scale, unit = cache[stem]

        t0 = align_block_start(float(r["block_start"]), t, scale)
        if t0 is None:
            ax.text(0.5, 0.5,
                    f"block start {r['block_start']:.0f} is outside\n"
                    f"{stem} epoch range [{t.min():.0f}, {t.max():.0f}]",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_axis_off()
            continue

        block_len = 30.0 * scale          # 30 seconds expressed in raw epoch units
        m = (t >= t0) & (t < t0 + block_len)
        if m.sum() < 10:
            ax.text(0.5, 0.5, f"no samples in 30-s block at {t0:.0f} {unit}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_axis_off()
            continue

        # x-axis: seconds elapsed within the block
        ax.plot((t[m] - t0) / scale, sig[m], lw=0.6, color="black")

        # shade rejected seconds (per-second table shares the recording's units)
        pm = ps[(ps["patient_id"].astype(str) == pid)
                & (ps["start_time"] >= t0) & (ps["start_time"] < t0 + block_len)]
        for _, w in pm.iterrows():
            if int(w.get("pred_index", 1)) == 0:
                ws = (float(w["start_time"]) - t0) / scale
                we = (float(w.get("end_time", w["start_time"] + scale)) - t0) / scale
                ax.axvspan(ws, we, color="crimson", alpha=0.16, lw=0)

        hr_txt = ""
        if np.isfinite(r["hr_bcg"]) and np.isfinite(r["hr_ecg"]):
            hr_txt = f" | BCG {r['hr_bcg']:.0f} vs ECG {r['hr_ecg']:.0f} bpm"
        # elapsed time from the start of the recording, not the raw epoch value
        elapsed_s = (t0 - float(t.min())) / scale
        hh, rem = divmod(int(elapsed_s), 3600)
        mm, ss = divmod(rem, 60)
        ax.set_title(f"{stem}  t+{hh:02d}:{mm:02d}:{ss:02d}  |  "
                     f"BCG fraction {r['bcg_fraction']:.2f}"
                     f"  ({r['stratum']}){hr_txt}", fontsize=9)
        ax.set_xlabel("Time within 30-s block (s)", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
        plotted.append(dict(r, raw_file=candidates[0].name))

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_axis_off()

    fig.suptitle("Example 30-second BCG blocks by quality-gate output "
                 "(red shading = seconds classified Non-BCG)", y=1.005, fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out_dir / "patch5_example_windows.png", dpi=200, bbox_inches="tight")
    fig.savefig(args.out_dir / "patch5_example_windows.pd", bbox_inches="tight")

    sel.to_csv(args.out_dir / "patch5_selected_segments.csv", index=False)

    # ---- optional blinded export for a second-reader agreement study ----
    if args.export_blind:
        bdir = args.out_dir / "patch5_blind"
        bdir.mkdir(parents=True, exist_ok=True)
        key = []
        order = rng.permutation(len(sel))
        for j, i in enumerate(order, start=1):
            r = sel.iloc[i]
            stem = re.search(r"(Sub\d+)", str(r["patient_id"]))
            stem = stem.group(1) if stem else str(r["patient_id"])
            if stem not in cache:
                continue
            t, sig, scale, unit = cache[stem]
            t0 = align_block_start(float(r["block_start"]), t, scale)
            if t0 is None:
                continue
            m = (t >= t0) & (t < t0 + 30.0 * scale)
            if m.sum() < 10:
                continue
            f, a = plt.subplots(figsize=(10, 2.6))
            a.plot((t[m] - t0) / scale, sig[m], lw=0.6, color="black")
            a.set_title(f"Segment {j:03d}", fontsize=10)   # no quality info shown
            a.set_xlabel("Time (s)")
            a.set_ylabel("Amplitude")
            a.grid(alpha=0.25)
            f.tight_layout()
            f.savefig(bdir / f"segment_{j:03d}.png", dpi=150, bbox_inches="tight")
            plt.close(f)
            key.append(dict(segment=f"segment_{j:03d}", patient_id=r["patient_id"],
                            block_start=t0, bcg_fraction=r["bcg_fraction"],
                            stratum=r["stratum"]))
        pd.DataFrame(key).to_csv(bdir / "key.csv", index=False)
        print(f"  Blinded set: {len(key)} segments -> {bdir} (key.csv withheld from the reader)")

    # ---- summary / caption text ----
    lines = []
    A = lines.append
    A("=" * 78)
    A("EXAMPLE ACCEPTED / REJECTED 30-SECOND WINDOWS (Reviewer 1, comment 2)")
    A("=" * 78)
    A("")
    A("SELECTION RULE (state this in the figure caption -- it is what makes the")
    A("figure evidence rather than illustration):")
    A("")
    A("  Blocks were stratified by the gate's block-level BCG fraction into")
    for name, lo, hi, label in STRATA:
        A(f"    - {name:<11} [{lo:.2f}, {hi:.2f})   {label}")
    A(f"  and {args.n_per_stratum} block(s) per stratum were drawn UNIFORMLY AT RANDOM")
    A(f"  with a fixed seed ({args.seed}). No block was inspected before selection.")
    A("")
    A("SELECTED SEGMENTS")
    A("-" * 78)
    A(f"{'Stratum':<12}{'Subject':<28}{'block_start (raw)':>19}{'BCGfrac':>10}{'HR_BCG':>9}{'HR_ECG':>9}")
    for _, r in sel.iterrows():
        hb = f"{r['hr_bcg']:.0f}" if np.isfinite(r["hr_bcg"]) else "--"
        he = f"{r['hr_ecg']:.0f}" if np.isfinite(r["hr_ecg"]) else "--"
        A(f"{r['stratum']:<12}{str(r['patient_id'])[:27]:<28}{r['block_start']:>19.0f}"
          f"{r['bcg_fraction']:>10.2f}{hb:>9}{he:>9}")
    A("")
    A("SCOPE -- BE PRECISE ABOUT THIS IN THE RESPONSE LETTER")
    A("-" * 78)
    A("  This figure is a qualitative audit of gate behaviour on the external cohort.")
    A("  It does NOT constitute external validation against ground truth, because no")
    A("  independent quality annotation exists for this dataset -- which the")
    A("  manuscript already lists as an outstanding limitation.")
    A("")
    A("  To answer the reviewer with evidence rather than illustration, run")
    A("  --export_blind, have a second reader score the unlabelled segments, and")
    A("  report Cohen's kappa against the gate. That converts this from 'here are")
    A("  some pictures' into a measured agreement statistic.")
    A("=" * 78)

    summary = "\n".join(lines)
    (args.out_dir / "patch5_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print()
    for f in ["patch5_example_windows.png", "patch5_selected_segments.csv",
              "patch5_summary.txt"]:
        print(f"Wrote: {args.out_dir / f}")


if __name__ == "__main__":
    main()
