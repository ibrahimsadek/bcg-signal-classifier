#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
patch4_calibration_external.py
===============================
Addresses Reviewer 1 comment #3: "The calibrated and uncalibrated model may be
used for assessing the external dataset, report if any difference in performance
is observed... The calibrated model is expected to perform more reliably."

THE REVIEWER'S EXPECTATION IS MATHEMATICALLY UNFOUNDED -- AND THIS PROVES IT
----------------------------------------------------------------------------
Temperature scaling divides the logits by a single positive constant T before
the softmax. That transform is strictly monotonic, so for BINARY classification
at a fixed 0.5 threshold it cannot change which class wins argmax: the decision
boundary sits at logit difference = 0, and dividing by T > 0 leaves the sign of
that difference untouched. Calibrated and uncalibrated gates therefore emit the
IDENTICAL accept/reject mask, hence identical coverage, identical retained
blocks, and identical downstream HR error.

Calibration changes the CONFIDENCE VALUES (Brier, ECE, NLL), not the DECISIONS.
The manuscript already states this for the internal ablation ("because
temperature scaling is monotonic it leaves accuracy, balanced accuracy,
macro-F1 and AUC unchanged") but never restates it for the external section,
which is what left the reviewer room to ask.

This script produces the empirical demonstration to put in the rebuttal:
it verifies decision-identity directly on the external per-second predictions,
and quantifies how much the probabilities themselves shift.

Requires no rerun -- per_second_predictions.csv already carries BOTH
prob_BCG (calibrated) and prob_BCG_uncal (uncalibrated).

USAGE (from the repo root):

    python patch4_calibration_external.py
    python patch4_calibration_external.py --per_second grid_search\\per_second_predictions.csv

OUTPUTS (into --out_dir):
    patch4_calibration_external.csv       per-subject decision agreement
    patch4_calibration_external.png/.pdf  probability shift + agreement figure
    patch4_calibration_summary.txt        rebuttal-ready readout
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Calibrated vs uncalibrated gate decisions on the external cohort")
    ap.add_argument("--per_second", type=Path,
                    default=Path("grid_search") / "per_second_predictions.csv")
    ap.add_argument("--out_dir", type=Path, default=Path("figures"))
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Decision threshold applied to prob_BCG (default 0.5)")
    args = ap.parse_args()

    if not args.per_second.exists():
        raise SystemExit(f"{args.per_second} not found -- pass --per_second explicitly.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    head = pd.read_csv(args.per_second, nrows=1)
    if "prob_BCG_uncal" not in head.columns:
        raise SystemExit(
            "prob_BCG_uncal column is missing from per_second_predictions.csv.\n"
            "Re-run inference with a build of run_hr_pipeline.py that records it "
            "(the current version does)."
        )
    keep = [c for c in ["patient_id", "recording_name", "prob_BCG", "prob_BCG_uncal",
                        "pred_index", "is_bcg"] if c in head.columns]
    df = pd.read_csv(args.per_second, usecols=keep)
    df = df.dropna(subset=["prob_BCG", "prob_BCG_uncal"])
    if df.empty:
        raise SystemExit("No rows with both calibrated and uncalibrated probabilities.")

    thr = args.threshold
    df["decision_cal"] = (df["prob_BCG"] >= thr).astype(int)
    df["decision_uncal"] = (df["prob_BCG_uncal"] >= thr).astype(int)
    df["agree"] = (df["decision_cal"] == df["decision_uncal"]).astype(int)
    df["prob_shift"] = df["prob_BCG"] - df["prob_BCG_uncal"]

    n = len(df)
    n_disagree = int((df["agree"] == 0).sum())
    pct_agree = 100.0 * df["agree"].mean()

    # per-subject
    grp = df.groupby("patient_id", dropna=False)
    per_sub = grp.agg(
        n_windows=("agree", "size"),
        pct_agreement=("agree", lambda s: 100.0 * s.mean()),
        n_disagree=("agree", lambda s: int((s == 0).sum())),
        accept_rate_cal=("decision_cal", "mean"),
        accept_rate_uncal=("decision_uncal", "mean"),
        mean_abs_prob_shift=("prob_shift", lambda s: float(np.abs(s).mean())),
        max_abs_prob_shift=("prob_shift", lambda s: float(np.abs(s).max())),
    ).reset_index()
    per_sub["accept_rate_delta"] = per_sub["accept_rate_cal"] - per_sub["accept_rate_uncal"]
    per_sub.to_csv(args.out_dir / "patch4_calibration_external.csv", index=False)

    # ---------------- figure ----------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.4))

    sample = df.sample(min(len(df), 40000), random_state=0)
    ax1.scatter(sample["prob_BCG_uncal"], sample["prob_BCG"], s=2, alpha=0.08,
                color="steelblue", edgecolors="none")
    ax1.plot([0, 1], [0, 1], "k--", lw=1.0, label="identity")
    ax1.axvline(thr, color="crimson", ls=":", lw=1.2)
    ax1.axhline(thr, color="crimson", ls=":", lw=1.2, label=f"threshold {thr:g}")
    ax1.set_xlabel("P(BCG) uncalibrated")
    ax1.set_ylabel("P(BCG) calibrated")
    ax1.set_title("Monotonic remap:\npoints never cross into the off-diagonal quadrants")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.hist(df["prob_shift"], bins=120, color="darkorange", alpha=0.85)
    ax2.axvline(0, color="black", lw=1.0)
    ax2.set_xlabel("P(BCG) calibrated $-$ uncalibrated")
    ax2.set_ylabel("1-second windows")
    ax2.set_title(f"Probability shift\nmean |shift| = {np.abs(df['prob_shift']).mean():.4f}")
    ax2.grid(alpha=0.3)

    ax3.hist(per_sub["pct_agreement"], bins=20, color="seagreen", alpha=0.85)
    ax3.set_xlabel("Per-subject decision agreement (%)")
    ax3.set_ylabel("Subjects")
    ax3.set_title(f"Decision agreement\noverall {pct_agree:.4f}% ({n_disagree}/{n} differ)")
    ax3.grid(alpha=0.3)

    fig.suptitle("Calibrated vs uncalibrated quality gate on the external cohort", y=1.03)
    fig.tight_layout()
    fig.savefig(args.out_dir / "patch4_calibration_external.png", dpi=200, bbox_inches="tight")
    fig.savefig(args.out_dir / "patch4_calibration_external.pdf", bbox_inches="tight")

    # ---------------- summary ----------------
    lines = []
    A = lines.append
    A("=" * 78)
    A("CALIBRATED vs UNCALIBRATED QUALITY GATE -- EXTERNAL COHORT")
    A("(Reviewer 1, comment 3)")
    A("=" * 78)
    A("")
    A(f"  1-second windows analysed        : {n:,}")
    A(f"  Subjects                         : {per_sub['patient_id'].nunique()}")
    A(f"  Decision threshold               : {thr:g}")
    A("")
    A(f"  Windows with IDENTICAL decision  : {n - n_disagree:,} ({pct_agree:.6f}%)")
    A(f"  Windows with DIFFERENT decision  : {n_disagree:,}")
    A("")
    A(f"  Accept rate, calibrated          : {df['decision_cal'].mean():.6f}")
    A(f"  Accept rate, uncalibrated        : {df['decision_uncal'].mean():.6f}")
    A(f"  Difference                       : "
      f"{df['decision_cal'].mean() - df['decision_uncal'].mean():+.6f}")
    A("")
    A(f"  Mean |probability shift|         : {np.abs(df['prob_shift']).mean():.6f}")
    A(f"  Max  |probability shift|         : {np.abs(df['prob_shift']).max():.6f}")
    A("")
    A("INTERPRETATION")
    A("-" * 78)
    if n_disagree == 0:
        A("  Decision agreement is EXACTLY 100%.")
        A("")
        A("  This is the expected result, not a coincidence. Temperature scaling")
        A("  divides logits by a single T > 0 before the softmax. For binary")
        A("  classification at a fixed threshold this is strictly monotonic, so it")
        A("  cannot move any window across the decision boundary. The calibrated and")
        A("  uncalibrated gates emit the same accept/reject mask, therefore identical")
        A("  block coverage and identical downstream HR error -- there is no MAE,")
        A("  coverage or F1 difference to report, by construction.")
        A("")
        A("  Calibration changes CONFIDENCE (Brier / ECE / NLL), not DECISIONS. The")
        A("  manuscript reports those internal calibration effects already, and they")
        A("  were negligible on average (dECE = +0.0026).")
        A("")
        A("  SUGGESTED REBUTTAL WORDING:")
        A("    'We verified this directly on the external cohort: calibrated and")
        A("     uncalibrated gates agreed on 100% of N 1-second windows, because")
        A("     temperature scaling is a monotonic transform of the logits and so")
        A("     cannot alter the argmax at a fixed threshold. Downstream coverage and")
        A("     HR error are consequently identical. We have added a sentence to the")
        A("     external-validation section making this explicit.'")
    else:
        A(f"  {n_disagree} windows ({100*n_disagree/n:.4f}%) received different decisions.")
        A("")
        A("  For pure temperature scaling this should be ZERO. A non-zero count means")
        A("  one of the following, and is worth checking before you write the rebuttal:")
        A("    * the calibrator is not a pure temperature (check calibrator.json 'type')")
        A("    * probabilities were rounded/serialised at limited precision, so windows")
        A("      sitting exactly at the threshold flip on floating-point noise")
        A("    * the decision used elsewhere in the pipeline is not a plain 0.5 threshold")
        A("  Inspect the affected rows: they should all sit within ~1e-6 of the threshold.")
    A("=" * 78)

    summary = "\n".join(lines)
    (args.out_dir / "patch4_calibration_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print()
    for f in ["patch4_calibration_external.csv", "patch4_calibration_external.png",
              "patch4_calibration_summary.txt"]:
        print(f"Wrote: {args.out_dir / f}")


if __name__ == "__main__":
    main()
