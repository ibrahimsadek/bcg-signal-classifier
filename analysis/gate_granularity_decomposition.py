#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
patch6_common_blocks_decomposition.py
======================================
THE DECISIVE EXPERIMENT.

patch1 showed that gating at the manuscript's own operating point (beta=0.50)
makes downstream HR estimation SIGNIFICANTLY WORSE than not gating at all
(+0.73 bpm, p < 0.0001, only 3/46 subjects improved). This script works out WHY,
by separating the two things `--disable_gate` switches off simultaneously:

  (A) BLOCK-level gating      -- whole 30-s blocks rejected when
                                 bcg_fraction < beta
  (B) INTERVAL-level restriction -- within a RETAINED block, J-peak detection is
                                 restricted to the seconds the gate called usable
                                 (per patch4, the gate accepts only ~71.9% of
                                 seconds, so ~28% are stripped even inside
                                 blocks that pass)

The patch1 comparison confounds these. At beta=0.50 block gating removes only
~0.7% of blocks, which cannot plausibly move MAE by 0.73 bpm -- so (B) is the
prime suspect. This script tests that directly.

METHOD
------
Restrict BOTH arms to the INTERSECTION of blocks that are paired-and-retained in
the ungated run AND in the gated run. On that common set, block-level gating is
identical by construction, so any residual difference in error is attributable
purely to interval-level restriction. This also removes the selection confound
that makes beta=0.90 look falsely good (lower MAE on an easier subset of blocks
is selection, not gate value).

Decomposition reported:
    total effect      = gated MAE  - ungated MAE   (all blocks, patch1's number)
    interval effect   = gated MAE  - ungated MAE   (common blocks only)
    selection effect  = total - interval           (attributable to which blocks
                                                    each arm chose to score)

Mechanism evidence: the same common blocks are compared on
n_seconds_with_j_peak and valid J-J interval counts, which is what the
interval-level restriction actually removes.

USAGE (from the repo root, after patch1 has run):

    python patch6_common_blocks_decomposition.py
    python patch6_common_blocks_decomposition.py --gated beta0.70

OUTPUTS (into --out_dir):
    patch6_common_blocks.csv            per-subject decomposition
    patch6_common_blocks_summary.txt    verdict + mechanism readout
    patch6_common_blocks.png/.pdf       figure
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
        raise SystemExit(f"{p} not found -- has patch1_run_gate_ablation.bat completed?")
    df = pd.read_csv(p)
    for need in ("patient_id", "block_idx"):
        if need not in df.columns:
            raise SystemExit(f"{p} lacks required column '{need}'. Columns: {list(df.columns)}")
    return df


def mark_scored(df: pd.DataFrame) -> pd.DataFrame:
    """Flag blocks that actually contributed a paired BCG/ECG comparison."""
    df = df.copy()
    if "paired_hr_available" in df.columns:
        scored = df["paired_hr_available"].astype(bool)
    else:
        hb = df["hr_bcg_bpm"] if "hr_bcg_bpm" in df.columns else df.get("base_hr_bcg_bpm")
        scored = hb.notna() & df["hr_ecg_bpm"].notna()
    df["scored"] = scored
    return df


def hr_cols(df: pd.DataFrame) -> tuple[str, str]:
    bcg = "hr_bcg_bpm" if "hr_bcg_bpm" in df.columns else "base_hr_bcg_bpm"
    return bcg, "hr_ecg_bpm"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Separate block-level gating from interval-level restriction")
    ap.add_argument("--ablation_root", type=Path, default=Path("gate_ablation"))
    ap.add_argument("--ungated", type=str, default="ungated")
    ap.add_argument("--gated", type=str, default="beta0.50")
    ap.add_argument("--out_dir", type=Path, default=Path("figures"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    u = mark_scored(load_blocks(args.ablation_root / args.ungated))
    g = mark_scored(load_blocks(args.ablation_root / args.gated))

    ub, ue = hr_cols(u)
    gb, ge = hr_cols(g)
    u["abs_err"] = (u[ub] - u[ue]).abs()
    g["abs_err"] = (g[gb] - g[ge]).abs()

    key = ["patient_id", "block_idx"]

    extra = [c for c in ["n_seconds_with_j_peak", "diagnostic_valid_jj_intervals",
                         "dense_valid_jj_intervals", "base_valid_jj_intervals",
                         "bcg_fraction", "n_bcg_seconds", "n_seconds_in_block"]
             if c in u.columns and c in g.columns]

    uu = u[key + ["scored", "abs_err"] + extra].add_suffix("_u").rename(
        columns={f"{k}_u": k for k in key})
    gg = g[key + ["scored", "abs_err"] + extra].add_suffix("_g").rename(
        columns={f"{k}_g": k for k in key})
    m = uu.merge(gg, on=key, how="outer")

    n_blocks_total = len(m)
    common = m[(m["scored_u"] == True) & (m["scored_g"] == True)].copy()  # noqa: E712
    only_u = m[(m["scored_u"] == True) & (m["scored_g"] != True)]         # noqa: E712
    only_g = m[(m["scored_u"] != True) & (m["scored_g"] == True)]         # noqa: E712

    if common.empty:
        raise SystemExit("No blocks scored in both arms -- cannot decompose.")

    # ---------- per-subject aggregation (patient = unit of inference) ----------
    per_sub = common.groupby("patient_id").agg(
        n_common_blocks=("abs_err_u", "size"),
        mae_ungated_common=("abs_err_u", "mean"),
        mae_gated_common=("abs_err_g", "mean"),
    ).reset_index()
    per_sub["d_mae_common"] = per_sub["mae_gated_common"] - per_sub["mae_ungated_common"]

    all_u = u[u["scored"]].groupby("patient_id")["abs_err"].mean().rename("mae_ungated_all")
    all_g = g[g["scored"]].groupby("patient_id")["abs_err"].mean().rename("mae_gated_all")
    per_sub = per_sub.merge(all_u, on="patient_id", how="left").merge(
        all_g, on="patient_id", how="left")
    per_sub["d_mae_all"] = per_sub["mae_gated_all"] - per_sub["mae_ungated_all"]
    per_sub["selection_effect"] = per_sub["d_mae_all"] - per_sub["d_mae_common"]
    per_sub.to_csv(args.out_dir / "patch6_common_blocks.csv", index=False)

    d_all = float(per_sub["d_mae_all"].mean())
    d_common = float(per_sub["d_mae_common"].mean())
    d_sel = d_all - d_common

    def wtest(a, b):
        if st is None or len(a) < 3:
            return np.nan
        try:
            return float(st.wilcoxon(a, b).pvalue)
        except ValueError:
            return np.nan

    p_common = wtest(per_sub["mae_gated_common"], per_sub["mae_ungated_common"])
    p_all = wtest(per_sub["mae_gated_all"], per_sub["mae_ungated_all"])

    # ---------- mechanism: what interval restriction removes ----------
    mech = []
    for c in extra:
        if c in ("bcg_fraction", "n_seconds_in_block"):
            continue
        cu, cg = f"{c}_u", f"{c}_g"
        if cu in common.columns and cg in common.columns:
            vu = common[cu].astype(float)
            vg = common[cg].astype(float)
            ok = vu.notna() & vg.notna()
            if ok.sum() == 0:
                continue
            mech.append(dict(
                quantity=c,
                ungated_mean=float(vu[ok].mean()),
                gated_mean=float(vg[ok].mean()),
                mean_delta=float((vg[ok] - vu[ok]).mean()),
                pct_change=100.0 * float((vg[ok] - vu[ok]).mean()) / float(vu[ok].mean())
                if vu[ok].mean() else np.nan,
                blocks_reduced=int((vg[ok] < vu[ok]).sum()),
                n=int(ok.sum()),
            ))
    mech_df = pd.DataFrame(mech)

    # correlation: does losing more seconds cost more error, block by block?
    corr_txt = "n/a"
    if "n_seconds_with_j_peak_u" in common.columns and st is not None:
        lost = (common["n_seconds_with_j_peak_u"].astype(float)
                - common["n_seconds_with_j_peak_g"].astype(float))
        derr = common["abs_err_g"].astype(float) - common["abs_err_u"].astype(float)
        ok = lost.notna() & derr.notna()
        if ok.sum() > 10 and lost[ok].std() > 0:
            r, p = st.spearmanr(lost[ok], derr[ok])
            corr_txt = f"Spearman rho = {r:+.3f} (p = {p:.2g}, n = {int(ok.sum())} blocks)"
            common["seconds_lost"] = lost
            common["d_abs_err"] = derr

    # ---------- summary ----------
    L = []
    A = L.append
    A("=" * 80)
    A("COMMON-BLOCK DECOMPOSITION: block-level gating vs interval-level restriction")
    A(f"Comparing '{args.gated}' against '{args.ungated}'")
    A("=" * 80)
    A("")
    A("BLOCK BOOKKEEPING")
    A("-" * 80)
    A(f"  Blocks present in either arm              : {n_blocks_total:,}")
    A(f"  Scored in BOTH arms (common set)          : {len(common):,}")
    A(f"  Scored only in ungated (lost to gating)   : {len(only_u):,}")
    A(f"  Scored only in gated (gained)             : {len(only_g):,}")
    A("")
    A("MAE DECOMPOSITION (per-subject means, n = %d subjects)" % len(per_sub))
    A("-" * 80)
    A(f"  Ungated MAE, all scored blocks            : {per_sub['mae_ungated_all'].mean():8.3f} bpm")
    A(f"  Gated   MAE, all scored blocks            : {per_sub['mae_gated_all'].mean():8.3f} bpm")
    A(f"  TOTAL effect (patch1's number)            : {d_all:+8.3f} bpm   (Wilcoxon p = {p_all:.2g})")
    A("")
    A(f"  Ungated MAE, common blocks only           : {per_sub['mae_ungated_common'].mean():8.3f} bpm")
    A(f"  Gated   MAE, common blocks only           : {per_sub['mae_gated_common'].mean():8.3f} bpm")
    A(f"  INTERVAL-level effect                     : {d_common:+8.3f} bpm   (Wilcoxon p = {p_common:.2g})")
    A("")
    A(f"  SELECTION effect (which blocks scored)    : {d_sel:+8.3f} bpm")
    A("")
    share = (100.0 * d_common / d_all) if d_all != 0 else np.nan
    A(f"  Share of the total effect due to interval-level restriction: {share:.1f}%")
    A("  Subjects where gating hurts on common blocks: "
      f"{int((per_sub['d_mae_common'] > 0).sum())}/{len(per_sub)}")
    A("")
    A("MECHANISM: what interval-level restriction removes (common blocks only)")
    A("-" * 80)
    if not mech_df.empty:
        A(f"{'Quantity':<32}{'Ungated':>10}{'Gated':>10}{'Delta':>10}{'%':>9}{'Blocks down':>13}")
        for _, r in mech_df.iterrows():
            A(f"{r['quantity']:<32}{r['ungated_mean']:>10.2f}{r['gated_mean']:>10.2f}"
              f"{r['mean_delta']:>10.2f}{r['pct_change']:>8.1f}%{r['blocks_reduced']:>13,}")
    else:
        A("  (no comparable count columns found in both arms)")
    A("")
    A("  Block-level correlation, seconds lost vs error increase:")
    A(f"    {corr_txt}")
    A("")
    A("VERDICT")
    A("-" * 80)
    if d_common > 0.05 and (np.isnan(p_common) or p_common < 0.05):
        A("  => INTERVAL-LEVEL RESTRICTION IS THE CULPRIT.")
        A("")
        A("     On blocks scored identically by both arms -- where block-level")
        A("     gating makes no difference by construction -- the gated pipeline is")
        A(f"     still {d_common:.3f} bpm worse. The harm therefore comes from")
        A("     restricting J-peak detection to gate-accepted seconds INSIDE")
        A("     retained blocks, not from rejecting whole blocks.")
        A("")
        A("     Mechanism: stripping ~28% of seconds fragments the J-J interval")
        A("     sequence, leaving fewer and more scattered intervals to average,")
        A("     so the block HR estimate gets noisier even though the seconds")
        A("     removed were individually lower-quality.")
        A("")
        A("     ACTIONABLE: test a BLOCK-GATING-ONLY variant -- use the gate to")
        A("     accept/reject whole blocks, but run J-peak detection over the")
        A("     entire retained block. That keeps the quality gate's purpose while")
        A("     removing the mechanism that hurts. If that variant beats ungated,")
        A("     you have a positive result AND an explanation for the negative one.")
    elif abs(d_common) <= 0.05:
        A("  => INTERVAL-LEVEL RESTRICTION IS NEUTRAL.")
        A("")
        A("     On common blocks the two arms are effectively identical, so the")
        A(f"     total effect ({d_all:+.3f} bpm) is essentially all SELECTION: the")
        A("     arms differ because they score different block sets, not because")
        A("     gating changes the estimate on a given block.")
        A("")
        A("     This means the gate is REJECTING BLOCKS THAT WOULD HAVE BEEN")
        A("     ESTIMATED WELL -- it is discarding usable signal. Check the error")
        A("     distribution of the blocks in 'scored only in ungated'.")
    else:
        A("  => INTERVAL-LEVEL RESTRICTION HELPS on common blocks")
        A(f"     ({d_common:+.3f} bpm), so the overall harm is driven by selection.")
        A("     The gate improves per-block estimates but rejects blocks that would")
        A("     have scored well. Re-examine the beta threshold rather than the")
        A("     interval masking.")
    A("")
    A("  NOTE ON beta=0.90: its apparently low MAE in patch1 is NOT gate value.")
    A("  At 3.3% coverage the arms score entirely different block subsets, so that")
    A("  comparison is selection, not performance. Only the common-block numbers")
    A("  above support a like-for-like claim.")
    A("=" * 80)

    summary = "\n".join(L)
    (args.out_dir / "patch6_common_blocks_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

    # ---------- figure ----------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6))

    lim_lo = min(per_sub["mae_ungated_common"].min(), per_sub["mae_gated_common"].min()) - 1
    lim_hi = max(per_sub["mae_ungated_common"].max(), per_sub["mae_gated_common"].max()) + 1
    ax1.scatter(per_sub["mae_ungated_common"], per_sub["mae_gated_common"],
                s=34, color="steelblue", alpha=0.8, edgecolors="white", linewidths=0.5)
    ax1.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1.0)
    ax1.set_xlim(lim_lo, lim_hi)
    ax1.set_ylim(lim_lo, lim_hi)
    ax1.set_xlabel("Ungated MAE on common blocks (bpm)")
    ax1.set_ylabel("Gated MAE on common blocks (bpm)")
    ax1.set_title("Like-for-like, same blocks\npoints above the line = gating hurts")
    ax1.grid(alpha=0.3)

    ax2.bar(["Total\n(patch1)", "Interval-level\n(common blocks)", "Selection\n(block choice)"],
            [d_all, d_common, d_sel],
            color=["#888888", "crimson", "steelblue"], alpha=0.9)
    ax2.axhline(0, color="black", lw=1.0)
    for i, v in enumerate([d_all, d_common, d_sel]):
        ax2.text(i, v, f"{v:+.3f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=10, fontweight="bold")
    ax2.set_ylabel("$\\Delta$MAE vs ungated (bpm)")
    ax2.set_title("Decomposition of the gate's harm\n(positive = worse than ungated)")
    ax2.grid(alpha=0.3, axis="y")

    if "seconds_lost" in common.columns:
        s = common.sample(min(len(common), 30000), random_state=0)
        ax3.scatter(s["seconds_lost"], s["d_abs_err"], s=3, alpha=0.08,
                    color="crimson", edgecolors="none")
        ax3.axhline(0, color="black", lw=1.0)
        ax3.set_xlabel("Seconds with J-peak lost to gating (per block)")
        ax3.set_ylabel("Increase in |error| (bpm)")
        ax3.set_title(f"Does losing seconds cost accuracy?\n{corr_txt}", fontsize=9)
        ax3.grid(alpha=0.3)
    else:
        ax3.set_visible(False)

    fig.suptitle(f"Isolating why gating hurts: {args.gated} vs {args.ungated}", y=1.03)
    fig.tight_layout()
    fig.savefig(args.out_dir / "patch6_common_blocks.png", dpi=200, bbox_inches="tight")
    fig.savefig(args.out_dir / "patch6_common_blocks.pd", bbox_inches="tight")

    print()
    for f in ["patch6_common_blocks.csv", "patch6_common_blocks_summary.txt",
              "patch6_common_blocks.png"]:
        print(f"Wrote: {args.out_dir / f}")


if __name__ == "__main__":
    main()
