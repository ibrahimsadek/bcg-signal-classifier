#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
patch9_coverage_curves.py
==========================
Does the Conv-Transformer gate sit ON the model-free J-J regularity curve,
or BELOW it?

patch8 established, at ONE coverage level (66%):

    keep everything            13.907
    random, coverage-matched   13.910   (+0.003 -- flat, so no shrinkage artefact)
    JJ-count heuristic         14.636   (+0.728 -- actively misleading)
    GATE (beta=0.70)           13.668   (-0.239, beats random p < 0.0005)
    JJ-CV heuristic            13.598   (-0.310, ties the gate, p = 0.69)

A tie at a single point is weak evidence either way: the two methods might
coincide there and diverge elsewhere. This script sweeps every method across
the full coverage range (5%..100%) and plots MAE-vs-coverage curves, so the
comparison becomes a characterisation rather than a single contested number.

INTERPRETATION
--------------
  * Gate ON the JJ-CV curve  -> the learned model recovered J-J regularity and
                                nothing beyond it. Clean, defensible, and worth
                                stating plainly: a one-line statistic replaces
                                a Conv-Transformer for this task.
  * Gate BELOW the curve     -> the learned representation captures something
                                the regularity statistic misses. Quantify the
                                gap and where in coverage it appears.
  * Gate ABOVE the curve     -> the heuristic dominates outright.

The gate contributes only two points (its beta=0.50 and beta=0.70 operating
points) because its ranking is a fixed classifier output, not a continuous
score that can be swept. The heuristics sweep continuously. That asymmetry is
inherent, not a flaw in the comparison -- the gate's points are plotted as
markers against the heuristic curves.

Everything is scored on the UNGATED run, so all methods rank the same blocks
with the same underlying HR estimates. Only the keep/drop decision differs.

USAGE (from the repo root):

    python patch9_coverage_curves.py
    python patch9_coverage_curves.py --n_perm 500

OUTPUTS (into --out_dir):
    patch9_coverage_curves.csv          MAE at every coverage level, per method
    patch9_coverage_curves.png/.pdf     the figure (publication-ready)
    patch9_coverage_curves_summary.txt  verdict + the gate-vs-curve comparison
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

COVERAGES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.66,
             0.70, 0.80, 0.90, 0.95, 0.99, 1.00]


def load_blocks(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "per_30s_hr.csv"
    if not p.exists():
        raise SystemExit(f"{p} not found.")
    return pd.read_csv(p)


def mark_scored(df: pd.DataFrame) -> pd.Series:
    if "paired_hr_available" in df.columns:
        return df["paired_hr_available"].astype(bool)
    bcg = "hr_bcg_bpm" if "hr_bcg_bpm" in df.columns else "base_hr_bcg_bpm"
    return df[bcg].notna() & df["hr_ecg_bpm"].notna()


def pick_col(df: pd.DataFrame, *names):
    for n in names:
        if n in df.columns:
            return n
    return None


def sweep_metric(d: pd.DataFrame, col: str, ascending: bool,
                 coverages: list[float]) -> dict[float, float]:
    """Per-subject: keep the top fraction by `col`, return cohort mean MAE."""
    out = {}
    for cov in coverages:
        per_sub = []
        for _, s in d.groupby("patient_id", sort=False):
            err = s["abs_err"].to_numpy(dtype=float)
            v = s[col].to_numpy(dtype=float)
            n_keep = max(1, int(round(cov * len(s))))
            finite = np.isfinite(v)
            if finite.sum() < n_keep:
                # not enough ranked values: fall back to keeping everything finite
                idx = np.where(np.isfinite(err))[0][:n_keep]
            else:
                order = np.argsort(np.where(finite, v, np.inf if ascending else -np.inf),
                                   kind="stable")
                if not ascending:
                    order = order[::-1]
                idx = order[:n_keep]
            vals = err[idx]
            if np.any(np.isfinite(vals)):
                per_sub.append(float(np.nanmean(vals)))
        out[cov] = float(np.mean(per_sub)) if per_sub else np.nan
    return out


def sweep_random(d: pd.DataFrame, coverages: list[float], n_perm: int,
                 seed: int) -> tuple[dict[float, float], dict[float, float]]:
    rng = np.random.default_rng(seed)
    mean_out, lo_out = {}, {}
    groups = [s["abs_err"].to_numpy(dtype=float) for _, s in d.groupby("patient_id", sort=False)]
    for cov in coverages:
        perms = np.empty(n_perm)
        for p in range(n_perm):
            per_sub = []
            for err in groups:
                n_keep = max(1, int(round(cov * len(err))))
                idx = rng.choice(len(err), size=n_keep, replace=False)
                vals = err[idx]
                if np.any(np.isfinite(vals)):
                    per_sub.append(float(np.nanmean(vals)))
            perms[p] = np.mean(per_sub) if per_sub else np.nan
        mean_out[cov] = float(np.nanmean(perms))
        lo_out[cov] = float(np.nanpercentile(perms, 5))
    return mean_out, lo_out


def gate_point(d: pd.DataFrame, gated_dir: Path) -> tuple[float, float] | None:
    """Return (coverage, cohort mean MAE) for a gate condition, scored on ungated."""
    g = load_blocks(gated_dir)
    g["scored"] = mark_scored(g)
    kept = g.loc[g["scored"], ["patient_id", "block_idx"]].copy()
    kept["gate_kept"] = True
    m = d.merge(kept, on=["patient_id", "block_idx"], how="left")
    m["gate_kept"] = m["gate_kept"].fillna(False).infer_objects(copy=False).astype(bool)
    per_sub, covs = [], []
    for _, s in m.groupby("patient_id", sort=False):
        k = s["gate_kept"].to_numpy()
        if k.sum() == 0:
            continue
        per_sub.append(float(np.nanmean(s["abs_err"].to_numpy(dtype=float)[k])))
        covs.append(k.mean())
    if not per_sub:
        return None
    return float(np.mean(covs)), float(np.mean(per_sub))


def main() -> None:
    ap = argparse.ArgumentParser(description="MAE-vs-coverage curves: gate vs model-free heuristics")
    ap.add_argument("--ablation_root", type=Path, default=Path("gate_ablation"))
    ap.add_argument("--ungated", type=str, default="ungated")
    ap.add_argument("--gates", type=str, nargs="*",
                    default=["blockonly0.50", "blockonly0.70"])
    ap.add_argument("--out_dir", type=Path, default=Path("figures"))
    ap.add_argument("--n_perm", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    u = load_blocks(args.ablation_root / args.ungated)
    u["scored"] = mark_scored(u)
    u = u[u["scored"]].copy()
    bcg = pick_col(u, "hr_bcg_bpm", "base_hr_bcg_bpm")
    u["abs_err"] = (u[bcg] - u["hr_ecg_bpm"]).abs()

    jj_col = pick_col(u, "diagnostic_valid_jj_intervals", "dense_valid_jj_intervals",
                      "n_valid_jj_intervals")
    cv_col = pick_col(u, "mean_diagnostic_jj_cv", "diagnostic_jj_cv", "jj_cv",
                      "mean_dense_jj_cv")

    print(f"  blocks: {len(u):,}  subjects: {u['patient_id'].nunique()}")
    print(f"  J-J count column : {jj_col}")
    print(f"  J-J CV column    : {cv_col}")
    print(f"  sweeping {len(COVERAGES)} coverage levels, {args.n_perm} permutations each...")

    curves: dict[str, dict[float, float]] = {}
    if cv_col:
        curves["JJ-CV (lowest first)"] = sweep_metric(u, cv_col, True, COVERAGES)
    if jj_col:
        curves["JJ-count (most first)"] = sweep_metric(u, jj_col, False, COVERAGES)
    rand_mean, rand_lo = sweep_random(u, COVERAGES, args.n_perm, args.seed)
    curves["Random (coverage-matched)"] = rand_mean

    gates = {}
    for gname in args.gates:
        gp = gate_point(u, args.ablation_root / gname)
        if gp:
            gates[gname] = gp

    # ---------------- table ----------------
    rows = []
    for cov in COVERAGES:
        r = {"coverage": cov}
        for name, c in curves.items():
            r[name] = c.get(cov, np.nan)
        r["random_p5"] = rand_lo.get(cov, np.nan)
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(args.out_dir / "patch9_coverage_curves.csv", index=False)

    # ---------------- gate vs interpolated heuristic ----------------
    L = []
    A = L.append
    A("=" * 80)
    A("MAE-vs-COVERAGE CURVES: does the learned gate beat model-free J-J statistics?")
    A("=" * 80)
    A("")
    A("  All methods are scored on the ungated run, so every method ranks the same")
    A("  blocks with the same HR estimates. Only the keep/drop decision differs.")
    A("")
    hdr = f"{'Coverage':>9}" + "".join(f"{n[:22]:>24}" for n in curves)
    A(hdr)
    A("-" * len(hdr))
    for cov in COVERAGES:
        line = f"{cov:>9.2f}"
        for name, c in curves.items():
            v = c.get(cov, np.nan)
            line += f"{v:>24.3f}" if np.isfinite(v) else f"{'--':>24}"
        A(line)
    A("")
    A("GATE OPERATING POINTS vs THE JJ-CV CURVE AT MATCHED COVERAGE")
    A("-" * 80)
    verdicts = []
    if cv_col and "JJ-CV (lowest first)" in curves:
        cvc = curves["JJ-CV (lowest first)"]
        xs = np.array(sorted(cvc))
        ys = np.array([cvc[x] for x in xs])
        for gname, (gcov, gmae) in gates.items():
            h = float(np.interp(gcov, xs, ys))
            diff = gmae - h
            verdicts.append((gname, gcov, gmae, h, diff))
            tag = ("BELOW curve (gate better)" if diff < -0.02 else
                   "ABOVE curve (heuristic better)" if diff > 0.02 else
                   "ON the curve (tied)")
            A(f"  {gname:<16} coverage {gcov:.3f}   gate {gmae:7.3f}   "
              f"JJ-CV {h:7.3f}   diff {diff:+6.3f}   -> {tag}")
    A("")
    A("VERDICT")
    A("-" * 80)
    meaningful = [v for v in verdicts if v[1] < 0.98]   # ignore near-100% coverage points
    if not meaningful:
        A("  No gate operating point below 98% coverage -- nothing to compare.")
    else:
        diffs = [v[4] for v in meaningful]
        if all(abs(x) <= 0.02 for x in diffs):
            A("  => THE GATE SITS ON THE MODEL-FREE CURVE.")
            A("")
            A("     At matched coverage the Conv-Transformer performs the same as a")
            A("     one-line J-J regularity statistic. The most defensible reading is")
            A("     that the learned model recovered beat-spacing regularity and")
            A("     little beyond it for this downstream task.")
            A("")
            A("     State this plainly. It is a legitimate finding, it pre-empts the")
            A("     comparison a referee would make anyway, and it strengthens rather")
            A("     than weakens the paper's methodological argument: downstream value")
            A("     must be demonstrated, not inferred from classification metrics.")
        elif all(x < -0.02 for x in diffs):
            A("  => THE GATE SITS BELOW THE CURVE (learned model adds value).")
            A("     At matched coverage it beats the model-free statistic. Report the")
            A("     gap and where in the coverage range it appears.")
        elif all(x > 0.02 for x in diffs):
            A("  => THE HEURISTIC DOMINATES THE GATE at matched coverage.")
            A("     A model-free J-J statistic is the better selector on this cohort.")
        else:
            A("  => MIXED: the gate's advantage depends on the operating point.")
            A("     Report the curve, not a single comparison.")
    A("")
    A("NOTE ON JJ-COUNT")
    A("-" * 80)
    if jj_col and "JJ-count (most first)" in curves:
        c = curves["JJ-count (most first)"]
        at66 = c.get(0.66, np.nan)
        base = curves["Random (coverage-matched)"].get(0.66, np.nan)
        if np.isfinite(at66) and np.isfinite(base):
            if at66 > base + 0.02:
                A("  Keeping blocks with the MOST detected J-J intervals is WORSE than")
                A(f"  random at 66% coverage ({at66:.3f} vs {base:.3f}). More detectable")
                A("  peaks indicates artefact, not signal -- the same inversion already")
                A("  documented for RMS energy in the internal feature baselines.")
                A("  Regularity of beat spacing predicts usability; quantity of peaks")
                A("  actively misleads. That is a coherent physiological thread running")
                A("  through both halves of the paper and is worth stating explicitly.")
            elif at66 < base - 0.02:
                A("  Keeping blocks with the MOST detected J-J intervals BEATS random")
                A(f"  at 66% coverage ({at66:.3f} vs {base:.3f}), so peak count carries")
                A("  usable information on this cohort. Compare it against JJ-CV above")
                A("  to see which statistic is the stronger selector.")
            else:
                A("  J-J count is indistinguishable from random at 66% coverage")
                A(f"  ({at66:.3f} vs {base:.3f}): peak quantity carries no selection")
                A("  information here, in contrast to beat-spacing regularity.")
    A("=" * 80)

    summary = "\n".join(L)
    (args.out_dir / "patch9_coverage_curves_summary.txt").write_text(summary, encoding="utf-8")
    print()
    print(summary)

    # ---------------- figure ----------------
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    xs = np.array(COVERAGES) * 100

    style = {
        "JJ-CV (lowest first)": dict(color="darkorange", marker="o", lw=2.0),
        "JJ-count (most first)": dict(color="seagreen", marker="^", lw=1.6),
        "Random (coverage-matched)": dict(color="grey", marker="", lw=1.4, ls="--"),
    }
    for name, c in curves.items():
        ys = [c.get(x, np.nan) for x in COVERAGES]
        ax.plot(xs, ys, label=name, ms=5, **style.get(name, {}))

    if "Random (coverage-matched)" in curves:
        lo = [rand_lo.get(x, np.nan) for x in COVERAGES]
        hi = [2 * rand_mean.get(x, np.nan) - rand_lo.get(x, np.nan) for x in COVERAGES]
        ax.fill_between(xs, lo, hi, color="grey", alpha=0.12,
                        label="Random 5-95th pct")

    colors = ["crimson", "purple", "teal"]
    for i, (gname, (gcov, gmae)) in enumerate(gates.items()):
        ax.plot(gcov * 100, gmae, marker="*", ms=19, ls="none",
                color=colors[i % len(colors)],
                markeredgecolor="white", markeredgewidth=0.8,
                label=f"Gate {gname.replace('blockonly', 'β=')} "
                      f"({gcov*100:.0f}%, {gmae:.3f})", zorder=5)

    ax.set_xlabel("Block coverage retained (%)")
    ax.set_ylabel("Cohort mean per-subject MAE (bpm)")
    ax.set_title("Downstream HR error vs coverage:\nlearned quality gate against model-free J-J statistics")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(args.out_dir / "patch9_coverage_curves.png", dpi=200, bbox_inches="tight")
    fig.savefig(args.out_dir / "patch9_coverage_curves.pd", bbox_inches="tight")

    print()
    for f in ["patch9_coverage_curves.csv", "patch9_coverage_curves_summary.txt",
              "patch9_coverage_curves.png"]:
        print(f"Wrote: {args.out_dir / f}")


if __name__ == "__main__":
    main()
