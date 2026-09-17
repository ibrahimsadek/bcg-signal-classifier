#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
patch3_minority_class_metrics.py
=================================
Addresses editorial points #2, #3 and #5:

  #2  Per-fold Non-BCG (minority) F1 is never reported -- the headline
      "F1 = 0.845" is positive-class only under 87.4% prevalence.
  #3  Per-fold minority-class COUNTS vary by two orders of magnitude
      (10 windows in one fold vs >1200 in another), yet all folds are
      averaged with equal weight.
  #5  Balanced accuracy -- which the manuscript calls "the decisive
      comparison for a quality gate" -- has no Nadeau-Bengio corrected
      confidence interval, while accuracy/precision/recall/F1 all do.

Reads the existing per-fold reports. Retrains nothing. Runs in seconds.

USAGE (from the repo root, e.g. D:\\Ibrahim\\bcgProject\\bcg-test\\bcg_signal_classifier):

    python patch3_minority_class_metrics.py
    python patch3_minority_class_metrics.py --cv_dir cv_output_nested_v2 --out_dir figures

OUTPUTS (into --out_dir):
    patch3_minority_class_metrics.csv   per-fold table, all metrics
    patch3_minority_class_metrics.tex   LaTeX table, drop-in for the manuscript
    patch3_corrected_ci.csv             NB-corrected CIs incl. balanced accuracy
    patch3_summary.txt                  human-readable summary
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as st
except ImportError:  # scipy is in requirements.txt, but fail loudly rather than silently
    raise SystemExit("scipy is required (pip install scipy)")


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------
def parse_fold_report(path: Path) -> dict:
    """Parse one fold_report.txt into a flat dict.

    The confusion matrix is stored as sklearn's layout:
        [[TN, FP],
         [FN, TP]]
    with class 0 = NonBCG (minority) and class 1 = BCG (majority).
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    rec: dict = {"fold_report_path": str(path)}

    # simple "key: value" scalars
    for key in ("fold", "model", "acc", "prec", "rec", "f1", "temperature_T",
                "brier_uncal", "ece_uncal", "nll_uncal",
                "brier_cal", "ece_cal", "nll_cal"):
        m = re.search(rf"^{key}\s*:\s*(.+)$", text, re.M)
        if m:
            raw = m.group(1).strip()
            try:
                rec[key] = float(raw)
            except ValueError:
                rec[key] = raw

    # list/dict-valued fields
    for key in ("test_patients", "train_fit_patients", "calib_patients", "best_hp",
                "train_fit_counts_before", "train_fit_counts_after"):
        m = re.search(rf"^{key}\s*:\s*(.+)$", text, re.M)
        if m:
            try:
                rec[key] = ast.literal_eval(m.group(1).strip())
            except (ValueError, SyntaxError):
                rec[key] = m.group(1).strip()

    # confusion matrix: spans two lines after the label
    m = re.search(r"^confusion_matrix\s*:\s*\n\s*\[\[(.*?)\]\]", text, re.M | re.S)
    if not m:
        raise ValueError(f"No confusion_matrix found in {path}")
    nums = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
    if len(nums) != 4:
        raise ValueError(f"Expected 4 confusion-matrix entries in {path}, got {len(nums)}")
    tn, fp, fn, tp = nums
    rec.update(TN=tn, FP=fp, FN=fn, TP=tp)
    return rec


def find_fold_reports(cv_dir: Path) -> list[Path]:
    paths = sorted(cv_dir.glob("fold_*/fold_report.txt"))
    if not paths:
        raise SystemExit(
            f"No fold_*/fold_report.txt found under {cv_dir}\n"
            f"Point --cv_dir at the nested-CV output directory "
            f"(e.g. cv_output_nested_v2)."
        )
    return paths


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def derive_metrics(rec: dict) -> dict:
    """Derive per-class metrics from the confusion counts.

    Class 0 = NonBCG (minority, the class the gate exists to detect).
    Class 1 = BCG    (majority).
    """
    tn, fp, fn, tp = rec["TN"], rec["FP"], rec["FN"], rec["TP"]
    n = tn + fp + fn + tp

    n_minority = tn + fp          # true NonBCG windows present in this fold
    n_majority = fn + tp          # true BCG windows

    def safe_div(a, b):
        return float(a) / float(b) if b else np.nan

    # majority / positive class (BCG)
    prec_bcg = safe_div(tp, tp + fp)
    rec_bcg = safe_div(tp, tp + fn)
    f1_bcg = safe_div(2 * prec_bcg * rec_bcg, prec_bcg + rec_bcg) \
        if np.isfinite(prec_bcg) and np.isfinite(rec_bcg) and (prec_bcg + rec_bcg) > 0 else np.nan

    # minority / negative class (NonBCG)
    prec_non = safe_div(tn, tn + fn)
    rec_non = safe_div(tn, tn + fp)      # = specificity
    f1_non = safe_div(2 * prec_non * rec_non, prec_non + rec_non) \
        if np.isfinite(prec_non) and np.isfinite(rec_non) and (prec_non + rec_non) > 0 else np.nan

    bal_acc = np.nanmean([rec_bcg, rec_non])
    macro_f1 = np.nanmean([f1_bcg, f1_non])
    acc = safe_div(tp + tn, n)

    # Wilson 95% interval on the minority recall (specificity) -- shows how
    # unreliable a fold with very few minority windows really is.
    lo, hi = wilson_interval(tn, n_minority)

    return dict(
        N=n,
        n_minority=n_minority,
        n_majority=n_majority,
        minority_prevalence=safe_div(n_minority, n),
        acc=acc,
        bal_acc=bal_acc,
        macro_f1=macro_f1,
        prec_BCG=prec_bcg, rec_BCG=rec_bcg, f1_BCG=f1_bcg,
        prec_NonBCG=prec_non, rec_NonBCG=rec_non, f1_NonBCG=f1_non,
        rec_NonBCG_ci_lo=lo, rec_NonBCG_ci_hi=hi,
    )


def wilson_interval(successes: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval -- behaves sensibly at tiny n, unlike the normal approx."""
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def nadeau_bengio_ci(values: np.ndarray, n_test: np.ndarray, n_train: np.ndarray,
                     alpha: float = 0.05) -> dict:
    """Nadeau-Bengio corrected interval for a K-fold CV mean.

    Variance is inflated by (1/K + n_test/n_train) instead of 1/K, because the
    K training sets overlap (~90% pairwise here). Student-t critical value at
    K-1 df, matching the manuscript's existing Table 5 methodology.
    """
    values = np.asarray(values, dtype=float)
    k = len(values)
    mean = float(np.mean(values))
    var = float(np.var(values, ddof=1))
    rho = float(np.mean(n_test / n_train))
    corrected_se = np.sqrt(var * (1.0 / k + rho))
    naive_se = np.sqrt(var / k)
    tcrit = st.t.ppf(1 - alpha / 2, df=k - 1)
    return dict(
        mean=mean,
        sd=np.sqrt(var),
        rho=rho,
        naive_lo=mean - tcrit * naive_se,
        naive_hi=mean + tcrit * naive_se,
        corrected_lo=mean - tcrit * corrected_se,
        corrected_hi=mean + tcrit * corrected_se,
        width_ratio=(corrected_se / naive_se) if naive_se > 0 else np.nan,
        k=k,
    )


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def write_latex(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"% Auto-generated by patch3_minority_class_metrics.py",
        r"% Per-fold minority-class (Non-BCG) performance and fold-level minority counts.",
        r"\begin{table}[!ht]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Per-fold minority-class performance and minority-class sample size.} "
        r"$n_{\text{Non-BCG}}$ is the number of true noise-dominated windows available in each "
        r"held-out patient. Non-BCG recall (specificity) is reported with a Wilson 95\% interval "
        r"to expose how weakly it is determined in folds with few minority windows. Macro-F1 and "
        r"balanced accuracy weight both classes equally and are the appropriate headline metrics "
        r"under 87.4\% positive-class prevalence; the positive-class F1 is shown for reference only.}",
        r"\label{tab:minority_perfold}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{ccrrcccccc}",
        r"\toprule",
        r"Fold & PID & $N$ & $n_{\text{Non-BCG}}$ & Prev. & Bal.\ Acc & Macro-F1 "
        r"& F1$_{\text{Non-BCG}}$ & Rec$_{\text{Non-BCG}}$ (95\% CI) & F1$_{\text{BCG}}$ \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        pid = str(r.get("test_pid", "--"))
        lines.append(
            f"{int(r['fold'])} & {pid} & {int(r['N'])} & {int(r['n_minority'])} & "
            f"{r['minority_prevalence']:.3f} & {r['bal_acc']:.3f} & {r['macro_f1']:.3f} & "
            f"{r['f1_NonBCG']:.3f} & {r['rec_NonBCG']:.3f} "
            f"[{r['rec_NonBCG_ci_lo']:.2f}, {r['rec_NonBCG_ci_hi']:.2f}] & "
            f"{r['f1_BCG']:.3f} \\\\"
        )
    lines += [
        r"\midrule",
        f"Mean & -- & -- & -- & {df['minority_prevalence'].mean():.3f} & "
        f"{df['bal_acc'].mean():.3f} & {df['macro_f1'].mean():.3f} & "
        f"{df['f1_NonBCG'].mean():.3f} & {df['rec_NonBCG'].mean():.3f} & "
        f"{df['f1_BCG'].mean():.3f} \\\\",
        f"$\\pm$SD & & & & {df['minority_prevalence'].std(ddof=1):.3f} & "
        f"{df['bal_acc'].std(ddof=1):.3f} & {df['macro_f1'].std(ddof=1):.3f} & "
        f"{df['f1_NonBCG'].std(ddof=1):.3f} & {df['rec_NonBCG'].std(ddof=1):.3f} & "
        f"{df['f1_BCG'].std(ddof=1):.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-fold minority-class metrics + NB-corrected CI incl. balanced accuracy"
    )
    ap.add_argument("--cv_dir", type=Path, default=Path("cv_output_nested_v2"),
                    help="Nested-CV output directory containing fold_*/fold_report.txt")
    ap.add_argument("--out_dir", type=Path, default=Path("figures"),
                    help="Where to write outputs")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = find_fold_reports(args.cv_dir)
    print(f"Found {len(paths)} fold reports under {args.cv_dir}")

    rows = []
    for p in paths:
        rec = parse_fold_report(p)
        rec.update(derive_metrics(rec))
        tp_list = rec.get("test_patients")
        rec["test_pid"] = tp_list[0] if isinstance(tp_list, (list, tuple)) and tp_list else "--"
        rows.append(rec)

    df = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)

    keep = ["fold", "test_pid", "N", "n_minority", "n_majority", "minority_prevalence",
            "acc", "bal_acc", "macro_f1",
            "prec_BCG", "rec_BCG", "f1_BCG",
            "prec_NonBCG", "rec_NonBCG", "f1_NonBCG",
            "rec_NonBCG_ci_lo", "rec_NonBCG_ci_hi",
            "TN", "FP", "FN", "TP"]
    out = df[keep].copy()

    csv_path = args.out_dir / "patch3_minority_class_metrics.csv"
    out.to_csv(csv_path, index=False)
    write_latex(df, args.out_dir / "patch3_minority_class_metrics.tex")

    # ---- Nadeau-Bengio corrected CIs, now including balanced accuracy ----
    n_test = df["N"].to_numpy(dtype=float)
    total = float(n_test.sum())
    n_train = total - n_test        # leave-one-patient-out: train = everything else

    ci_rows = []
    for metric, label in [("acc", "Accuracy"), ("prec_BCG", "Precision (BCG)"),
                          ("rec_BCG", "Recall (BCG)"), ("f1_BCG", "F1 (BCG)"),
                          ("bal_acc", "Balanced accuracy"), ("macro_f1", "Macro-F1"),
                          ("f1_NonBCG", "F1 (Non-BCG)")]:
        ci = nadeau_bengio_ci(df[metric].to_numpy(), n_test, n_train)
        ci_rows.append(dict(
            metric=label, mean=ci["mean"], sd=ci["sd"],
            naive_lo=ci["naive_lo"], naive_hi=ci["naive_hi"],
            corrected_lo=ci["corrected_lo"], corrected_hi=ci["corrected_hi"],
            width_vs_naive=ci["width_ratio"],
        ))
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(args.out_dir / "patch3_corrected_ci.csv", index=False)

    # ---- summary ----
    lines = []
    A = lines.append
    A("=" * 74)
    A("MINORITY-CLASS PERFORMANCE AND FOLD-LEVEL SAMPLE SIZE")
    A("=" * 74)
    A("")
    A("Class 0 = Non-BCG (noise-dominated, minority) -- the class the gate exists to detect.")
    A("Class 1 = BCG (usable, majority).")
    A("")
    A(f"{'Fold':<5}{'PID':<7}{'N':>7}{'n_nonBCG':>10}{'Prev':>8}"
      f"{'BalAcc':>9}{'MacroF1':>9}{'F1_non':>8}{'Rec_non':>9}{'  Wilson 95% CI':<20}")
    A("-" * 92)
    for _, r in out.iterrows():
        A(f"{int(r['fold']):<5}{str(r['test_pid']):<7}{int(r['N']):>7}{int(r['n_minority']):>10}"
          f"{r['minority_prevalence']:>8.3f}{r['bal_acc']:>9.3f}{r['macro_f1']:>9.3f}"
          f"{r['f1_NonBCG']:>8.3f}{r['rec_NonBCG']:>9.3f}"
          f"  [{r['rec_NonBCG_ci_lo']:.2f}, {r['rec_NonBCG_ci_hi']:.2f}]")
    A("-" * 92)
    A(f"{'Mean':<12}{out['N'].mean():>7.0f}{out['n_minority'].mean():>10.0f}"
      f"{out['minority_prevalence'].mean():>8.3f}{out['bal_acc'].mean():>9.3f}"
      f"{out['macro_f1'].mean():>9.3f}{out['f1_NonBCG'].mean():>8.3f}"
      f"{out['rec_NonBCG'].mean():>9.3f}")
    A("")
    A("HETEROGENEITY OF MINORITY-CLASS SAMPLE SIZE")
    A("-" * 74)
    lo_i = out["n_minority"].idxmin()
    hi_i = out["n_minority"].idxmax()
    A(f"  Smallest: fold {int(out.loc[lo_i, 'fold'])} (PID {out.loc[lo_i, 'test_pid']}) "
      f"-- {int(out.loc[lo_i, 'n_minority'])} minority windows, "
      f"Non-BCG recall {out.loc[lo_i, 'rec_NonBCG']:.3f} "
      f"[{out.loc[lo_i, 'rec_NonBCG_ci_lo']:.2f}, {out.loc[lo_i, 'rec_NonBCG_ci_hi']:.2f}]")
    A(f"  Largest:  fold {int(out.loc[hi_i, 'fold'])} (PID {out.loc[hi_i, 'test_pid']}) "
      f"-- {int(out.loc[hi_i, 'n_minority'])} minority windows, "
      f"Non-BCG recall {out.loc[hi_i, 'rec_NonBCG']:.3f} "
      f"[{out.loc[hi_i, 'rec_NonBCG_ci_lo']:.2f}, {out.loc[hi_i, 'rec_NonBCG_ci_hi']:.2f}]")
    ratio = out["n_minority"].max() / max(1, out["n_minority"].min())
    A(f"  Ratio largest:smallest = {ratio:.0f}:1")
    A("")
    A(f"  Folds where the gate is near-blind to the minority class (F1_NonBCG < 0.25): "
      f"{int((out['f1_NonBCG'] < 0.25).sum())}/{len(out)} "
      f"-> folds {sorted(out.loc[out['f1_NonBCG'] < 0.25, 'fold'].astype(int).tolist())}")
    A("")
    A("NADEAU-BENGIO CORRECTED 95% CONFIDENCE INTERVALS")
    A("-" * 74)
    A(f"{'Metric':<22}{'Mean':>8}{'Corrected 95% CI':>26}{'Width vs naive':>16}")
    for _, r in ci_df.iterrows():
        A(f"{r['metric']:<22}{r['mean']:>8.3f}"
          f"     [{r['corrected_lo']:.3f}, {r['corrected_hi']:.3f}]"
          f"{r['width_vs_naive']:>15.2f}x")
    A("")
    A("NOTE: balanced accuracy, macro-F1 and Non-BCG F1 are the rows the manuscript")
    A("      currently omits from its corrected-CI table (Table 5). They are the")
    A("      metrics the Discussion calls decisive, so they belong there.")
    A("=" * 74)

    summary = "\n".join(lines)
    (args.out_dir / "patch3_summary.txt").write_text(summary, encoding="utf-8")
    print()
    print(summary)
    print()
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {args.out_dir / 'patch3_minority_class_metrics.tex'}")
    print(f"Wrote: {args.out_dir / 'patch3_corrected_ci.csv'}")
    print(f"Wrote: {args.out_dir / 'patch3_summary.txt'}")


if __name__ == "__main__":
    main()
