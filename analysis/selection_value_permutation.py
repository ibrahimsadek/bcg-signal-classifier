#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
patch8_selection_value.py
==========================
Is the gate's block selection actually GOOD, or just SMALL?

patch6 showed that block-only gating at beta=0.70 lowers MAE by 0.239 bpm
(p = 0.035) -- but it achieves that by discarding 34% of blocks, and the entire
effect is SELECTION (interval-level effect was exactly 0.000). That shape is
identical to the beta=0.90 mirage: any procedure that drops a third of blocks
will lower the MAE of what remains, partly by luck, because some discarded
blocks were hard ones.

So "lower MAE on a smaller subset" is NOT evidence the gate discriminates. The
question is whether the gate's CHOICE of which blocks to drop beats alternatives
that drop the same NUMBER of blocks.

THREE COMPARATORS, ALL COVERAGE-MATCHED PER SUBJECT
----------------------------------------------------
  1. RANDOM       -- drop the same count at random (permutation null).
                     Repeated n_perm times to get a null distribution and an
                     empirical p-value. If the gate does not beat this, its
                     selection carries no information.

  2. JJ-COUNT     -- keep blocks with the most valid J-J intervals.
                     A one-line heuristic needing no model at all.

  3. JJ-CV        -- keep blocks whose J-J intervals are most regular
                     (lowest coefficient of variation). Also model-free, and
                     the physiologically motivated choice: a clean cardiac
                     block should have consistent beat spacing.

Comparators 2 and 3 double as the "benchmark against simpler methods" that
Reviewer 2 asked for. If a trivial heuristic matches a Conv-Transformer, that
is something you need to know BEFORE publishing, not after.

All comparisons are computed on the UNGATED run, so every method is ranking the
same blocks with the same underlying HR estimates. Only the keep/drop decision
differs. That is the cleanest possible isolation of selection quality.

USAGE (from the repo root):

    python patch8_selection_value.py
    python patch8_selection_value.py --gated blockonly0.50 --n_perm 2000

OUTPUTS (into --out_dir):
    patch8_selection_value.csv        per-subject results for every method
    patch8_selection_value_summary.txt  verdict + permutation p-values
    patch8_selection_value.png/.pdf   null distribution vs the gate
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Does the gate's block selection beat coverage-matched alternatives?")
    ap.add_argument("--ablation_root", type=Path, default=Path("gate_ablation"))
    ap.add_argument("--ungated", type=str, default="ungated")
    ap.add_argument("--gated", type=str, default="blockonly0.70")
    ap.add_argument("--out_dir", type=Path, default=Path("figures"))
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    u = load_blocks(args.ablation_root / args.ungated)
    g = load_blocks(args.ablation_root / args.gated)
    u["scored"] = mark_scored(u)
    g["scored"] = mark_scored(g)

    ubcg = pick_col(u, "hr_bcg_bpm", "base_hr_bcg_bpm")
    u["abs_err"] = (u[ubcg] - u["hr_ecg_bpm"]).abs()

    # which blocks the gate kept
    kept = g.loc[g["scored"], ["patient_id", "block_idx"]].copy()
    kept["gate_kept"] = True
    d = u[u["scored"]].merge(kept, on=["patient_id", "block_idx"], how="left")
    d["gate_kept"] = d["gate_kept"].fillna(False).astype(bool)

    jj_col = pick_col(d, "diagnostic_valid_jj_intervals", "dense_valid_jj_intervals",
                      "n_valid_jj_intervals")
    cv_col = pick_col(d, "mean_diagnostic_jj_cv", "diagnostic_jj_cv", "jj_cv",
                      "mean_dense_jj_cv")
    if jj_col is None:
        print("  [warn] no J-J count column found; JJ-COUNT comparator skipped")
    if cv_col is None:
        print("  [warn] no J-J CV column found; JJ-CV comparator skipped")

    rng = np.random.default_rng(args.seed)
    rows = []

    subjects = sorted(d["patient_id"].unique())
    perm_matrix = np.full((args.n_perm, len(subjects)), np.nan)

    for si, sub in enumerate(subjects):
        s = d[d["patient_id"] == sub]
        n_all = len(s)
        n_keep = int(s["gate_kept"].sum())
        if n_all == 0 or n_keep == 0:
            continue

        err = s["abs_err"].to_numpy(dtype=float)
        mae_all = float(np.nanmean(err))
        mae_gate = float(np.nanmean(err[s["gate_kept"].to_numpy()]))

        # random coverage-matched selections
        perms = np.empty(args.n_perm)
        for p in range(args.n_perm):
            idx = rng.choice(n_all, size=n_keep, replace=False)
            perms[p] = np.nanmean(err[idx])
        perm_matrix[:, si] = perms
        mae_rand = float(np.nanmean(perms))

        # heuristic selections: keep the top-n_keep blocks by each criterion
        def topk_mae(col: str | None, ascending: bool) -> float:
            if col is None or col not in s.columns:
                return np.nan
            v = s[col].to_numpy(dtype=float)
            if np.all(~np.isfinite(v)):
                return np.nan
            order = np.argsort(v, kind="stable")
            if not ascending:
                order = order[::-1]
            return float(np.nanmean(err[order[:n_keep]]))

        mae_jjcount = topk_mae(jj_col, ascending=False)   # MORE intervals = better
        mae_jjcv = topk_mae(cv_col, ascending=True)       # LOWER CV = more regular

        rows.append(dict(
            patient_id=sub, n_blocks=n_all, n_kept=n_keep,
            coverage=n_keep / n_all,
            mae_all=mae_all,
            mae_gate=mae_gate,
            mae_random_mean=mae_rand,
            mae_jjcount=mae_jjcount,
            mae_jjcv=mae_jjcv,
            gate_vs_all=mae_gate - mae_all,
            gate_vs_random=mae_gate - mae_rand,
            pct_random_better=float(np.mean(perms <= mae_gate)),
        ))

    res = pd.DataFrame(rows)
    res.to_csv(args.out_dir / "patch8_selection_value.csv", index=False)

    # cohort-level permutation test: per-permutation mean across subjects
    valid_cols = ~np.all(np.isnan(perm_matrix), axis=0)
    perm_cohort = np.nanmean(perm_matrix[:, valid_cols], axis=1)
    obs_gate = float(res["mae_gate"].mean())
    obs_all = float(res["mae_all"].mean())
    p_perm = float(np.mean(perm_cohort <= obs_gate))

    def wilcox(a, b):
        if st is None:
            return np.nan
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            return np.nan
        try:
            return float(st.wilcoxon(a[m], b[m]).pvalue)
        except ValueError:
            return np.nan

    p_gate_vs_rand = wilcox(res["mae_gate"].to_numpy(), res["mae_random_mean"].to_numpy())
    p_gate_vs_jjc = wilcox(res["mae_gate"].to_numpy(), res["mae_jjcount"].to_numpy())
    p_gate_vs_cv = wilcox(res["mae_gate"].to_numpy(), res["mae_jjcv"].to_numpy())

    L = []
    A = L.append
    A("=" * 80)
    A("SELECTION VALUE: does the gate's CHOICE of blocks beat matched alternatives?")
    A(f"Gate condition: {args.gated}   |   scored on the {args.ungated} run")
    A("=" * 80)
    A("")
    A("  All methods keep the SAME NUMBER of blocks per subject as the gate, and")
    A("  all score those blocks with the SAME ungated HR estimates. Only the")
    A("  keep/drop decision differs, so any MAE difference is selection quality.")
    A("")
    A(f"  Subjects                        : {len(res)}")
    A(f"  Mean coverage kept              : {res['coverage'].mean():.3f}")
    A(f"  Permutations per subject        : {args.n_perm:,}")
    A("")
    A("COHORT MEANS (per-subject MAE, bpm)")
    A("-" * 80)
    A(f"  Keep everything (no selection)  : {obs_all:8.3f}")
    A(f"  GATE selection                  : {obs_gate:8.3f}   ({obs_gate - obs_all:+.3f} vs all)")
    A(f"  RANDOM, coverage-matched        : {res['mae_random_mean'].mean():8.3f}"
      f"   ({res['mae_random_mean'].mean() - obs_all:+.3f} vs all)")
    if jj_col:
        A(f"  JJ-COUNT heuristic (no model)   : {res['mae_jjcount'].mean():8.3f}"
          f"   ({res['mae_jjcount'].mean() - obs_all:+.3f} vs all)")
    if cv_col:
        A(f"  JJ-CV heuristic (no model)      : {res['mae_jjcv'].mean():8.3f}"
          f"   ({res['mae_jjcv'].mean() - obs_all:+.3f} vs all)")
    A("")
    A("HEAD-TO-HEAD vs THE GATE (negative = comparator is BETTER than the gate)")
    A("-" * 80)
    A(f"  Gate - Random   : {obs_gate - res['mae_random_mean'].mean():+8.3f} bpm"
      f"   Wilcoxon p = {p_gate_vs_rand:.4g}")
    if jj_col:
        A(f"  Gate - JJ-count : {obs_gate - res['mae_jjcount'].mean():+8.3f} bpm"
          f"   Wilcoxon p = {p_gate_vs_jjc:.4g}")
    if cv_col:
        A(f"  Gate - JJ-CV    : {obs_gate - res['mae_jjcv'].mean():+8.3f} bpm"
          f"   Wilcoxon p = {p_gate_vs_cv:.4g}")
    A("")
    A("PERMUTATION TEST (cohort level)")
    A("-" * 80)
    A(f"  Observed gate MAE               : {obs_gate:.4f}")
    A(f"  Random null mean                : {perm_cohort.mean():.4f}")
    A(f"  Random null 5th percentile      : {np.percentile(perm_cohort, 5):.4f}")
    A(f"  Empirical p (random <= gate)    : {p_perm:.4f}")
    A("")
    A(f"  Subjects where the gate beats its own random null (p<0.05): "
      f"{int((res['pct_random_better'] < 0.05).sum())}/{len(res)}")
    A("")
    A("VERDICT")
    A("-" * 80)
    beats_random = (p_perm < 0.05) and (obs_gate < res["mae_random_mean"].mean())
    heur = [v for v in [res["mae_jjcount"].mean() if jj_col else np.nan,
                        res["mae_jjcv"].mean() if cv_col else np.nan]
            if np.isfinite(v)]
    beats_heuristics = all(obs_gate <= h + 1e-9 for h in heur) if heur else None

    if not beats_random:
        A("  => THE GATE'S SELECTION CARRIES NO DEMONSTRABLE INFORMATION.")
        A("")
        A("     Dropping the same number of blocks AT RANDOM does as well or better.")
        A("     The MAE improvement seen in patch6 is therefore a sample-shrinkage")
        A("     artefact, not evidence of discrimination: any procedure that discards")
        A("     a third of blocks lowers the MAE of what remains.")
        A("")
        A("     This is the single most important negative result in the study and it")
        A("     must be reported. It does NOT invalidate the classifier's internal")
        A("     metrics -- it shows those metrics do not translate into downstream")
        A("     selection value on this cohort and this estimator.")
    else:
        A("  => THE GATE BEATS COVERAGE-MATCHED RANDOM SELECTION.")
        A(f"     Empirical p = {p_perm:.4f}. Its block choice carries real information;")
        A("     the improvement is not merely sample shrinkage.")
        if beats_heuristics is False:
            A("")
            A("     HOWEVER: a model-free heuristic matches or beats it (see head-to-head")
            A("     above). Report both. A Conv-Transformer that ties a one-line J-J")
            A("     statistic is a finding in itself, and a referee will run this")
            A("     comparison mentally whether or not you provide it.")
        elif beats_heuristics:
            A("     It also beats the model-free J-J heuristics, so the learned")
            A("     representation adds value over trivial signal statistics.")
    A("")
    A("  SCOPE: this tests selection quality at ONE coverage level, on ONE")
    A("  downstream estimator. It says nothing about the classifier's agreement")
    A("  with the human annotator, which is a separate question the internal")
    A("  evaluation already answers.")
    A("=" * 80)

    summary = "\n".join(L)
    (args.out_dir / "patch8_selection_value_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

    # ---------------- figure ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1.hist(perm_cohort, bins=50, color="lightsteelblue",
             edgecolor="white", label=f"Random selection null ({args.n_perm:,} perms)")
    ax1.axvline(obs_gate, color="crimson", lw=2.2, label=f"Gate ({obs_gate:.3f})")
    ax1.axvline(obs_all, color="black", ls="--", lw=1.4, label=f"Keep all ({obs_all:.3f})")
    if jj_col and np.isfinite(res["mae_jjcount"].mean()):
        ax1.axvline(res["mae_jjcount"].mean(), color="seagreen", ls=":", lw=1.8,
                    label=f"JJ-count ({res['mae_jjcount'].mean():.3f})")
    if cv_col and np.isfinite(res["mae_jjcv"].mean()):
        ax1.axvline(res["mae_jjcv"].mean(), color="darkorange", ls=":", lw=1.8,
                    label=f"JJ-CV ({res['mae_jjcv'].mean():.3f})")
    ax1.set_xlabel("Cohort mean per-subject MAE (bpm)")
    ax1.set_ylabel("Permutations")
    ax1.set_title(f"Gate vs coverage-matched random\nempirical p = {p_perm:.4f}")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    lo = min(res["mae_random_mean"].min(), res["mae_gate"].min()) - 1
    hi = max(res["mae_random_mean"].max(), res["mae_gate"].max()) + 1
    ax2.scatter(res["mae_random_mean"], res["mae_gate"], s=34,
                color="steelblue", alpha=0.85, edgecolors="white", linewidths=0.5)
    ax2.plot([lo, hi], [lo, hi], "k--", lw=1.0)
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.set_xlabel("MAE under random selection (bpm)")
    ax2.set_ylabel("MAE under gate selection (bpm)")
    ax2.set_title("Per subject: below the line = gate beats random")
    ax2.grid(alpha=0.3)

    fig.suptitle("Is the gate's block selection better than dropping blocks at random?", y=1.03)
    fig.tight_layout()
    fig.savefig(args.out_dir / "patch8_selection_value.png", dpi=200, bbox_inches="tight")
    fig.savefig(args.out_dir / "patch8_selection_value.pdf", bbox_inches="tight")

    print()
    for f in ["patch8_selection_value.csv", "patch8_selection_value_summary.txt",
              "patch8_selection_value.png"]:
        print(f"Wrote: {args.out_dir / f}")


if __name__ == "__main__":
    main()
