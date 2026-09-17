#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
patch2_half_rate_diagnostic.py
===============================
Addresses editorial point #8 and the manuscript's own flagged gap.

The Discussion HYPOTHESISES that the systematic negative bias (-8.7 bpm sinus,
-15.9 bpm AF) reflects the J-peak detector selecting every other cardiac cycle
during tachycardia, and explicitly defers "a histogram of interval ratios" to
future work. You already have every paired BCG/ECG block needed to test it now.

WHAT IT TESTS
-------------
For each paired 30-second block, compute ratio = HR_BCG / HR_ECG.
  * ratio ~ 1.0  -> correct detection
  * ratio ~ 0.5  -> half-rate (every other beat missed)  <- the hypothesis
  * ratio ~ 2.0  -> double-rate (spurious extra peaks)
If half-rate detection drives the bias, the ratio distribution should be
BIMODAL with a clear secondary mass near 0.5, and that mass should grow with
reference HR (tachycardia) and be larger in AF.

Also reports what fraction of the total negative bias is attributable to
half-rate blocks -- i.e. whether the hypothesis actually explains the effect
or only a small slice of it.

USAGE (from the repo root):

    python patch2_half_rate_diagnostic.py
    python patch2_half_rate_diagnostic.py --per_30s_hr grid_search\\per_30s_hr.csv

OUTPUTS (into --out_dir):
    patch2_half_rate_histogram.png/.pdf   ratio distribution, all/sinus/AF
    patch2_half_rate_vs_hr.png/.pdf       ratio vs reference HR (the key panel)
    patch2_half_rate_stats.csv            per-band counts and bias decomposition
    patch2_half_rate_summary.txt          verdict: supported / not supported
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

# ratio bands
HALF_LO, HALF_HI = 0.40, 0.60
NORM_LO, NORM_HI = 0.85, 1.15
DBL_LO, DBL_HI = 1.80, 2.20


def load_af_lookup(info_path: Path) -> set[int]:
    """Subject indices with documented atrial fibrillation."""
    if not info_path.exists():
        return set()
    info = pd.read_excel(info_path)
    af = set()
    for _, row in info.iterrows():
        if "atrial fibrillation" in str(row.get("Conclusion", "")).lower():
            try:
                af.add(int(row["Idx"]))
            except (ValueError, TypeError, KeyError):
                pass
    return af


def band(r: float) -> str:
    if HALF_LO <= r <= HALF_HI:
        return "half"
    if NORM_LO <= r <= NORM_HI:
        return "normal"
    if DBL_LO <= r <= DBL_HI:
        return "double"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description="Half-rate detection diagnostic")
    ap.add_argument("--per_30s_hr", type=Path,
                    default=Path("grid_search") / "per_30s_hr.csv")
    ap.add_argument("--info_xlsx", type=Path, default=Path("Overall_info.xlsx"))
    ap.add_argument("--out_dir", type=Path, default=Path("figures"))
    args = ap.parse_args()

    if not args.per_30s_hr.exists():
        raise SystemExit(f"{args.per_30s_hr} not found -- pass --per_30s_hr explicitly.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.per_30s_hr)
    if "paired_hr_available" in df.columns:
        df = df[df["paired_hr_available"] == True].copy()  # noqa: E712
    df = df[(df["hr_bcg_bpm"] > 0) & (df["hr_ecg_bpm"] > 0)].copy()
    if df.empty:
        raise SystemExit("No paired blocks with positive HR found.")

    df["ratio"] = df["hr_bcg_bpm"] / df["hr_ecg_bpm"]
    df["signed_error"] = df["hr_bcg_bpm"] - df["hr_ecg_bpm"]
    df["band"] = df["ratio"].apply(band)

    def _sub_num(x: str) -> int:
        m = re.search(r"Sub(\d+)", str(x))
        return int(m.group(1)) if m else -1

    df["sub_num"] = df["patient_id"].apply(_sub_num)
    af_set = load_af_lookup(args.info_xlsx)
    df["group"] = df["sub_num"].apply(lambda x: "AF" if x in af_set else "Sinus")
    if not af_set:
        print("  [warn] Overall_info.xlsx not found or no AF labels; "
              "group stratification will show everything as 'Sinus'.")

    # ---------------- stats ----------------
    rows = []
    for grp in ["All", "Sinus", "AF"]:
        g = df if grp == "All" else df[df["group"] == grp]
        if g.empty:
            continue
        n = len(g)
        counts = g["band"].value_counts()
        half_mask = g["band"] == "half"
        total_bias = g["signed_error"].sum()
        half_bias = g.loc[half_mask, "signed_error"].sum()
        rows.append(dict(
            group=grp, n_blocks=n,
            pct_half=100 * counts.get("half", 0) / n,
            pct_normal=100 * counts.get("normal", 0) / n,
            pct_double=100 * counts.get("double", 0) / n,
            pct_other=100 * counts.get("other", 0) / n,
            mean_ratio=g["ratio"].mean(),
            median_ratio=g["ratio"].median(),
            mean_signed_error=g["signed_error"].mean(),
            mean_signed_error_excl_half=g.loc[~half_mask, "signed_error"].mean(),
            pct_of_total_bias_from_half=(100 * half_bias / total_bias)
            if total_bias != 0 else np.nan,
        ))
    stats = pd.DataFrame(rows)
    stats.to_csv(args.out_dir / "patch2_half_rate_stats.csv", index=False)

    # half-rate prevalence vs reference HR (the tachycardia prediction)
    bins = [0, 60, 80, 100, 120, 250]
    labels = ["<60", "60-80", "80-100", "100-120", ">120"]
    df["hr_band"] = pd.cut(df["hr_ecg_bpm"], bins=bins, labels=labels, right=False)
    by_hr = df.groupby("hr_band", observed=True).agg(
        n_blocks=("ratio", "size"),
        pct_half=("band", lambda s: 100 * (s == "half").mean()),
        mean_signed_error=("signed_error", "mean"),
    ).reset_index()

    # ---------------- figures ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, grp, col in zip(axes, ["All", "Sinus", "AF"],
                            ["gray", "steelblue", "crimson"]):
        g = df if grp == "All" else df[df["group"] == grp]
        if g.empty:
            ax.set_visible(False)
            continue
        ax.hist(g["ratio"].clip(0, 2.5), bins=100, color=col, alpha=0.75)
        ax.axvline(1.0, color="black", ls="-", lw=1.2, label="1.0 (correct)")
        ax.axvline(0.5, color="darkorange", ls="--", lw=1.6, label="0.5 (half-rate)")
        ax.axvline(2.0, color="purple", ls=":", lw=1.4, label="2.0 (double-rate)")
        pct = 100 * (g["band"] == "half").mean()
        ax.set_title(f"{grp} (n={len(g)} blocks)\nhalf-rate band: {pct:.1f}%")
        ax.set_xlabel("HR$_{BCG}$ / HR$_{ECG}$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Blocks")
    axes[0].legend(fontsize=8)
    fig.suptitle("Half-rate detection test: distribution of BCG/ECG heart-rate ratio", y=1.03)
    fig.tight_layout()
    fig.savefig(args.out_dir / "patch2_half_rate_histogram.png", dpi=200, bbox_inches="tight")
    fig.savefig(args.out_dir / "patch2_half_rate_histogram.pdf", bbox_inches="tight")

    fig2, (axa, axb) = plt.subplots(1, 2, figsize=(13, 4.6))
    for grp, col in [("Sinus", "steelblue"), ("AF", "crimson")]:
        g = df[df["group"] == grp]
        if g.empty:
            continue
        axa.scatter(g["hr_ecg_bpm"], g["ratio"].clip(0, 2.5), s=4, alpha=0.12,
                    color=col, edgecolors="none", label=grp)
    axa.axhline(1.0, color="black", lw=1.2)
    axa.axhline(0.5, color="darkorange", ls="--", lw=1.6)
    axa.set_xlabel("Reference HR from ECG (bpm)")
    axa.set_ylabel("HR$_{BCG}$ / HR$_{ECG}$")
    axa.set_title("Ratio vs reference HR\n(half-rate hypothesis predicts drift toward 0.5 at high HR)")
    axa.grid(alpha=0.3)
    leg = axa.legend(fontsize=9)
    # matplotlib renamed this attribute in 3.7; support both
    for h in getattr(leg, "legend_handles", getattr(leg, "legendHandles", [])):
        h.set_alpha(1.0)

    if not by_hr.empty:
        axb.bar(by_hr["hr_band"].astype(str), by_hr["pct_half"], color="darkorange", alpha=0.85)
        for i, r in by_hr.iterrows():
            axb.text(i, r["pct_half"], f"n={int(r['n_blocks'])}",
                     ha="center", va="bottom", fontsize=8)
        axb.set_xlabel("Reference HR band (bpm)")
        axb.set_ylabel("Blocks in half-rate band (%)")
        axb.set_title("Half-rate prevalence by reference HR")
        axb.grid(alpha=0.3, axis="y")
    fig2.tight_layout()
    fig2.savefig(args.out_dir / "patch2_half_rate_vs_hr.png", dpi=200, bbox_inches="tight")
    fig2.savefig(args.out_dir / "patch2_half_rate_vs_hr.pdf", bbox_inches="tight")

    # ---------------- verdict ----------------
    lines = []
    A = lines.append
    A("=" * 78)
    A("HALF-RATE DETECTION DIAGNOSTIC")
    A("Tests the Discussion's hypothesis that the negative HR bias arises from")
    A("the J-peak detector selecting every other cardiac cycle during tachycardia.")
    A("=" * 78)
    A("")
    A(f"{'Group':<8}{'Blocks':>8}{'%half':>8}{'%normal':>9}{'%double':>9}{'%other':>8}"
      f"{'MeanRatio':>11}{'Bias':>9}{'Bias(excl half)':>17}")
    A("-" * 87)
    for _, r in stats.iterrows():
        A(f"{r['group']:<8}{int(r['n_blocks']):>8}{r['pct_half']:>8.2f}"
          f"{r['pct_normal']:>9.2f}{r['pct_double']:>9.2f}{r['pct_other']:>8.2f}"
          f"{r['mean_ratio']:>11.3f}{r['mean_signed_error']:>9.2f}"
          f"{r['mean_signed_error_excl_half']:>17.2f}")
    A("")
    A("HALF-RATE PREVALENCE BY REFERENCE HR (the tachycardia prediction)")
    A("-" * 78)
    A(f"{'HR band':<12}{'Blocks':>9}{'% half-rate':>14}{'Mean signed err':>18}")
    for _, r in by_hr.iterrows():
        A(f"{str(r['hr_band']):<12}{int(r['n_blocks']):>9}{r['pct_half']:>14.2f}"
          f"{r['mean_signed_error']:>18.2f}")
    A("")
    A("VERDICT")
    A("-" * 78)
    allrow = stats[stats["group"] == "All"]
    if not allrow.empty:
        pct_half = float(allrow["pct_half"].iloc[0])
        share = float(allrow["pct_of_total_bias_from_half"].iloc[0])
        bias_all = float(allrow["mean_signed_error"].iloc[0])
        bias_excl = float(allrow["mean_signed_error_excl_half"].iloc[0])
        A(f"  Blocks in the half-rate band (0.40-0.60): {pct_half:.2f}%")
        A(f"  Share of total signed bias contributed by those blocks: {share:.1f}%")
        A(f"  Mean signed bias: {bias_all:.2f} bpm  ->  excluding half-rate blocks: {bias_excl:.2f} bpm")
        A("")
        if pct_half >= 10 and abs(bias_excl) < 0.5 * abs(bias_all):
            A("  => SUPPORTED. Half-rate detection accounts for a large share of the")
            A("     negative bias; removing those blocks substantially reduces it.")
            A("     Report the histogram and consider --half_double_tolerance as a fix.")
        elif pct_half >= 5:
            A("  => PARTIALLY SUPPORTED. Half-rate blocks are present and contribute,")
            A("     but a substantial bias remains after excluding them. The Discussion")
            A("     should say half-rate is ONE contributor, not THE mechanism.")
        else:
            A("  => NOT SUPPORTED. Too few blocks fall in the half-rate band to explain")
            A("     the bias. The current Discussion wording attributes the bias to a")
            A("     mechanism the data does not show -- revise it. Look instead at")
            A("     systematic J-peak mistiming or the 20 ms ECG resampling jitter.")
    A("")
    A("  NOTE: this is a block-level HR-ratio test, which is the practical proxy for")
    A("  the interval-ratio histogram the manuscript proposes. It shares the")
    A("  hypothesis's logic but aggregates within 30-s blocks, so a block mixing")
    A("  correct and halved beats lands between the modes rather than at 0.5.")
    A("=" * 78)

    summary = "\n".join(lines)
    (args.out_dir / "patch2_half_rate_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print()
    for f in ["patch2_half_rate_histogram.png", "patch2_half_rate_vs_hr.png",
              "patch2_half_rate_stats.csv", "patch2_half_rate_summary.txt"]:
        print(f"Wrote: {args.out_dir / f}")


if __name__ == "__main__":
    main()
