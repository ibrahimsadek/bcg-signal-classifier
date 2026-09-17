#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
patch1_compare_gate_ablation.py
================================
Addresses editorial point #7 and the core "does the gate actually help?"
question: aggregates the gated-vs-ungated runs produced by
patch1_run_gate_ablation.bat into a single comparison table + figure.

WHY A SWEEP AND NOT A SINGLE COMPARISON
----------------------------------------
At the manuscript's selected operating point (beta = 0.50) the gate retains
99.2% of blocks. Gating vs not gating can therefore differ on at most ~0.8%
of blocks, and a single ungated-vs-beta=0.50 comparison will show essentially
nothing -- that is arithmetic, not a finding about the model. Reporting it
alone hands a referee the conclusion that the gate is unnecessary.

The informative experiment is the SWEEP: ungated baseline vs beta in
{0.30, 0.50, 0.70, 0.90}, which characterises the MAE/coverage frontier and
shows whether error falls as gating tightens. Be prepared for the honest
possibility that the incremental benefit is small -- the manuscript's own
grid search already hints at it (beta 0.30 -> 0.70 buys 0.47 bpm for 34
points of coverage). If that is the answer, say so; it is still publishable
as "we formalised and measured the gate's actual effect size".

USAGE (after running patch1_run_gate_ablation.bat):

    python patch1_compare_gate_ablation.py
    python patch1_compare_gate_ablation.py --ablation_root gate_ablation --out_dir figures

OUTPUTS (into --out_dir):
    patch1_gate_ablation.csv          one row per condition
    patch1_gate_ablation.tex          LaTeX table, drop-in
    patch1_gate_ablation.png/.pdf     MAE-coverage frontier, ungated marked
    patch1_gate_ablation_summary.txt  human-readable readout incl. paired test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from scipy import stats as st
except ImportError:
    st = None


CONDITION_ORDER = ["ungated", "beta0.30", "beta0.50", "beta0.70", "beta0.90"]


def load_condition(run_dir: Path) -> pd.DataFrame | None:
    """Load the per-subject summary for one condition."""
    ps = run_dir / "patient_summary.csv"
    if not ps.exists():
        # analysis subfolder is the other place it can land
        alt = run_dir / "analysis" / "patient_analysis.csv"
        if alt.exists():
            return pd.read_csv(alt)
        return None
    return pd.read_csv(ps)


def subject_key(df: pd.DataFrame) -> pd.Series:
    return df["patient_id"].astype(str).str.extract(r"(Sub\d+)", expand=False).fillna(
        df["patient_id"].astype(str)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate gated-vs-ungated ablation runs")
    ap.add_argument("--ablation_root", type=Path, default=Path("gate_ablation"),
                    help="Directory containing the per-condition run subfolders")
    ap.add_argument("--out_dir", type=Path, default=Path("figures"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.ablation_root.exists():
        raise SystemExit(
            f"{args.ablation_root} not found.\n"
            f"Run patch1_run_gate_ablation.bat first (it creates the per-condition runs)."
        )

    per_subject: dict[str, pd.DataFrame] = {}
    rows = []
    for cond in CONDITION_ORDER:
        run_dir = args.ablation_root / cond
        df = load_condition(run_dir)
        if df is None:
            print(f"  [skip] {cond}: no patient_summary.csv under {run_dir}")
            continue
        df = df.copy()
        df["subject"] = subject_key(df)
        per_subject[cond] = df

        rows.append(dict(
            condition=cond,
            gate="OFF" if cond == "ungated" else "ON",
            beta=np.nan if cond == "ungated" else float(cond.replace("beta", "")),
            n_subjects=df["subject"].nunique(),
            mean_mae_bpm=df["mae_bpm"].mean(),
            median_mae_bpm=df["mae_bpm"].median(),
            sd_mae_bpm=df["mae_bpm"].std(ddof=1),
            mean_rmse_bpm=df["rmse_bpm"].mean() if "rmse_bpm" in df else np.nan,
            mean_coverage=df["preserved_window_coverage"].mean()
            if "preserved_window_coverage" in df else np.nan,
            total_paired_windows=int(df["n_paired_windows"].sum())
            if "n_paired_windows" in df else -1,
        ))

    if not rows:
        raise SystemExit("No conditions loaded -- check that the .bat run completed.")

    res = pd.DataFrame(rows)
    res.to_csv(args.out_dir / "patch1_gate_ablation.csv", index=False)

    # ---- paired per-subject tests vs the ungated baseline ----
    lines = []
    A = lines.append
    A("=" * 78)
    A("GATED vs UNGATED ABLATION -- does the quality gate improve downstream HR?")
    A("=" * 78)
    A("")
    A(f"{'Condition':<12}{'Gate':<6}{'beta':>6}{'MeanMAE':>10}{'MedMAE':>9}"
      f"{'SD':>8}{'Coverage':>10}{'PairedWin':>11}")
    A("-" * 78)
    for _, r in res.iterrows():
        beta = "--" if np.isnan(r["beta"]) else f"{r['beta']:.2f}"
        A(f"{r['condition']:<12}{r['gate']:<6}{beta:>6}{r['mean_mae_bpm']:>10.3f}"
          f"{r['median_mae_bpm']:>9.3f}{r['sd_mae_bpm']:>8.3f}"
          f"{r['mean_coverage']:>10.3f}{r['total_paired_windows']:>11d}")
    A("")

    if "ungated" in per_subject and st is not None:
        base = per_subject["ungated"][["subject", "mae_bpm"]].rename(
            columns={"mae_bpm": "mae_ungated"})
        A("PAIRED PER-SUBJECT COMPARISON vs UNGATED BASELINE")
        A("-" * 78)
        A(f"{'Condition':<12}{'n':>4}{'dMAE':>9}{'Wilcoxon p':>13}{'Cohen dz':>11}"
          f"{'Improved':>11}")
        A("-" * 78)
        for cond, df in per_subject.items():
            if cond == "ungated":
                continue
            m = base.merge(df[["subject", "mae_bpm"]], on="subject", how="inner")
            if len(m) < 3:
                continue
            d = m["mae_bpm"].to_numpy() - m["mae_ungated"].to_numpy()  # negative = gate better
            try:
                _, p = st.wilcoxon(m["mae_bpm"], m["mae_ungated"])
            except ValueError:
                p = np.nan
            dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
            A(f"{cond:<12}{len(m):>4}{d.mean():>9.3f}{p:>13.4f}{dz:>11.3f}"
              f"{int((d < 0).sum()):>7}/{len(m)}")
        A("")
        A("  dMAE < 0 means the gated run has LOWER error than ungated (gate helps).")
        A("")

    A("INTERPRETATION GUIDE")
    A("-" * 78)
    A("  * At beta=0.50 the gate retains ~99% of blocks, so a near-zero dMAE there")
    A("    is expected by construction and is NOT evidence the gate is useless.")
    A("  * The meaningful signal is the TREND across beta: if MAE falls as beta")
    A("    rises, the gate's ranking is informative even when the operating point")
    A("    is permissive.")
    A("  * If MAE is flat across the whole sweep, the honest conclusion is that the")
    A("    gate does not measurably improve this downstream estimator on this")
    A("    cohort -- report it plainly rather than leaving the claim implicit.")
    A("=" * 78)

    summary = "\n".join(lines)
    (args.out_dir / "patch1_gate_ablation_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

    # ---- figure: MAE-coverage frontier ----
    gated = res[res["gate"] == "ON"].sort_values("beta")
    ungated = res[res["gate"] == "OFF"]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    if not gated.empty:
        ax.plot(gated["mean_coverage"] * 100, gated["mean_mae_bpm"],
                "o-", color="steelblue", lw=1.8, ms=7, label="Gated (varying $\\beta$)", zorder=3)
        for _, r in gated.iterrows():
            ax.annotate(f"$\\beta$={r['beta']:.2f}",
                        (r["mean_coverage"] * 100, r["mean_mae_bpm"]),
                        textcoords="offset points", xytext=(7, 5), fontsize=9)
    if not ungated.empty:
        u = ungated.iloc[0]
        ax.axhline(u["mean_mae_bpm"], color="crimson", ls="--", lw=1.4,
                   label=f"Ungated baseline ({u['mean_mae_bpm']:.2f} bpm)", zorder=2)
        ax.plot(u["mean_coverage"] * 100, u["mean_mae_bpm"], "s",
                color="crimson", ms=9, zorder=4)
    ax.set_xlabel("Mean block coverage retained (%)")
    ax.set_ylabel("Mean per-subject MAE (bpm)")
    ax.set_title("Quality gate: MAE-coverage trade-off vs ungated baseline")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "patch1_gate_ablation.png", dpi=200, bbox_inches="tight")
    fig.savefig(args.out_dir / "patch1_gate_ablation.pdf", bbox_inches="tight")

    # ---- LaTeX ----
    tex = [
        r"% Auto-generated by patch1_compare_gate_ablation.py",
        r"\begin{table}[!ht]",
        r"\centering", r"\small",
        r"\caption{\textbf{Gated versus ungated downstream heart-rate estimation} on the "
        r"independent cohort. The ungated condition bypasses the quality gate entirely "
        r"(every 1-second interval marked usable); all other pipeline parameters are "
        r"identical. Coverage is the mean per-recording block-retention fraction.}",
        r"\label{tab:gate_ablation}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{llcccc}", r"\toprule",
        r"Condition & Gate & $\beta$ & Mean MAE (bpm) & Median MAE (bpm) & Coverage (\%) \\",
        r"\midrule",
    ]
    for _, r in res.iterrows():
        beta = "--" if np.isnan(r["beta"]) else f"{r['beta']:.2f}"
        tex.append(f"{r['condition']} & {r['gate']} & {beta} & "
                   f"{r['mean_mae_bpm']:.2f} & {r['median_mae_bpm']:.2f} & "
                   f"{r['mean_coverage']*100:.1f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (args.out_dir / "patch1_gate_ablation.tex").write_text("\n".join(tex), encoding="utf-8")

    print()
    for f in ["patch1_gate_ablation.csv", "patch1_gate_ablation.tex",
              "patch1_gate_ablation.png", "patch1_gate_ablation.pdf",
              "patch1_gate_ablation_summary.txt"]:
        print(f"Wrote: {args.out_dir / f}")


if __name__ == "__main__":
    main()
