# Bland-Altman and Pearson correlation plots for AF vs sinus rhythm groups.
# Co-authored with CoCo
"""
Generates per-group analysis:
  - Bland-Altman plot (difference vs mean)
  - Pearson regression/correlation plot (with r, p-value)
  - Summary statistics table

Requires: misc/per_30s_hr.csv, Overall_info.xlsx in the same directory.

Usage:
  python plot_af_vs_sinus_analysis.py
  python plot_af_vs_sinus_analysis.py --output_dir ./figures
  python plot_af_vs_sinus_analysis.py --per_30s_hr path/to/per_30s_hr.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def load_data(per_30s_path: Path, info_path: Path) -> pd.DataFrame:
    """Load per-30s HR data and merge with clinical AF/sinus classification."""
    hr = pd.read_csv(per_30s_path)
    hr = hr[hr["paired_hr_available"] == True].copy()

    info = pd.read_excel(info_path)
    # Build AF lookup: subject index -> has_af
    af_set = set()
    for _, row in info.iterrows():
        conclusion = str(row["Conclusion"]).lower()
        if "atrial fibrillation" in conclusion:
            af_set.add(int(row["Idx"]))

    # Extract subject number from patient_id (e.g. "Sub01_aligned_data_ecg" -> 1)
    import re
    hr["sub_num"] = hr["patient_id"].apply(
        lambda x: int(re.search(r"Sub(\d+)", x).group(1))
    )
    hr["group"] = hr["sub_num"].apply(lambda x: "AF" if x in af_set else "Sinus")

    return hr


def bland_altman_plot(
    ax: plt.Axes,
    ecg_hr: np.ndarray,
    bcg_hr: np.ndarray,
    title: str,
    color: str = "steelblue",
) -> dict:
    """Draw Bland-Altman plot on given axes. Returns stats dict."""
    mean_hr = (ecg_hr + bcg_hr) / 2
    diff = bcg_hr - ecg_hr  # BCG - ECG

    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    ax.scatter(mean_hr, diff, alpha=0.15, s=4, color=color, edgecolors="none")
    ax.axhline(mean_diff, color="red", linestyle="-", linewidth=1.5, label=f"Bias: {mean_diff:.2f}")
    ax.axhline(loa_upper, color="orange", linestyle="--", linewidth=1,
               label=f"+1.96 SD: {loa_upper:.2f}")
    ax.axhline(loa_lower, color="orange", linestyle="--", linewidth=1,
               label=f"-1.96 SD: {loa_lower:.2f}")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.5)

    ax.set_xlabel("Mean of ECG & BCG HR (bpm)")
    ax.set_ylabel("BCG HR - ECG HR (bpm)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    return {
        "bias": mean_diff,
        "sd": std_diff,
        "loa_upper": loa_upper,
        "loa_lower": loa_lower,
        "n_windows": len(diff),
    }


def main():
    parser = argparse.ArgumentParser(description="AF vs Sinus: Bland-Altman + Correlation")
    parser.add_argument("--per_30s_hr", type=Path, default=None,
                        help="Path to per_30s_hr.csv (default: misc/per_30s_hr.csv)")
    parser.add_argument("--info_xlsx", type=Path, default=None,
                        help="Path to Overall_info.xlsx")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Directory to save figures (default: same as per_30s_hr)")
    args = parser.parse_args()

    # Auto-detect paths
    script_dir = Path(__file__).parent
    per_30s_path = args.per_30s_hr or script_dir / "misc" / "per_30s_hr.csv"
    info_path = args.info_xlsx or script_dir / "Overall_info.xlsx"
    output_dir = args.output_dir or per_30s_path.parent

    if not per_30s_path.exists():
        print(f"ERROR: {per_30s_path} not found")
        return
    if not info_path.exists():
        print(f"ERROR: {info_path} not found")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and classify
    hr = load_data(per_30s_path, info_path)
    print(f"Loaded {len(hr)} paired windows")
    print(f"  AF group: {(hr['group'] == 'AF').sum()} windows, "
          f"{hr[hr['group'] == 'AF']['sub_num'].nunique()} subjects")
    print(f"  Sinus group: {(hr['group'] == 'Sinus').sum()} windows, "
          f"{hr[hr['group'] == 'Sinus']['sub_num'].nunique()} subjects")

    groups = {"AF": hr[hr["group"] == "AF"], "Sinus": hr[hr["group"] == "Sinus"]}
    colors = {"AF": "crimson", "Sinus": "steelblue"}

    # ================================================================
    # Figure 1: Bland-Altman plots (side by side)
    # ================================================================
    fig_ba, axes_ba = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    ba_stats = {}
    for i, (grp_name, grp_df) in enumerate(groups.items()):
        ecg = grp_df["hr_ecg_bpm"].values
        bcg = grp_df["hr_bcg_bpm"].values
        ba_stats[grp_name] = bland_altman_plot(
            axes_ba[i], ecg, bcg,
            title=f"Bland-Altman: {grp_name} (n={len(grp_df)}, {grp_df['sub_num'].nunique()} subj)",
            color=colors[grp_name],
        )
    fig_ba.suptitle("Bland-Altman Analysis: BCG HR vs ECG HR", fontsize=13, y=1.02)
    fig_ba.tight_layout()
    ba_path = output_dir / "bland_altman_af_vs_sinus.png"
    fig_ba.savefig(ba_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {ba_path}")

    # ================================================================
    # Figure 2: Bland-Altman for All subjects combined
    # ================================================================
    fig_all, ax_all = plt.subplots(1, 1, figsize=(8, 5))
    ecg_all = hr["hr_ecg_bpm"].values
    bcg_all = hr["hr_bcg_bpm"].values
    ba_stats["All"] = bland_altman_plot(
        ax_all, ecg_all, bcg_all,
        title=f"Bland-Altman: All Subjects (n={len(hr)}, 46 subj)",
        color="gray",
    )
    fig_all.tight_layout()
    all_path = output_dir / "bland_altman_all_subjects.png"
    fig_all.savefig(all_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {all_path}")

    # ================================================================
    # Summary statistics table
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"\n{'Group':<8} {'N_win':<8} {'N_subj':<8} {'MAE':<8} {'Bias':<8} "
          f"{'SD':<8} {'LoA_lo':<8} {'LoA_hi':<8}")
    print("-" * 70)

    for grp_name in ["Sinus", "AF", "All"]:
        grp_df = hr if grp_name == "All" else groups[grp_name]
        mae = grp_df["abs_error_bpm"].mean()
        ba = ba_stats[grp_name]
        n_subj = grp_df["sub_num"].nunique() if grp_name != "All" else 46
        print(
            f"{grp_name:<8} {ba['n_windows']:<8} {n_subj:<8} {mae:<8.3f} "
            f"{ba['bias']:<8.3f} {ba['sd']:<8.3f} {ba['loa_lower']:<8.2f} "
            f"{ba['loa_upper']:<8.2f}"
        )

    # Per-subject MAE by group
    print("\n\nPer-subject MAE by group:")
    per_subj = hr.groupby(["group", "sub_num"])["abs_error_bpm"].mean().reset_index()
    for grp in ["Sinus", "AF"]:
        g = per_subj[per_subj["group"] == grp]["abs_error_bpm"]
        print(f"  {grp}: mean={g.mean():.3f}, median={g.median():.3f}, "
              f"std={g.std():.3f}, min={g.min():.3f}, max={g.max():.3f}")

    # Save summary CSV
    summary_path = output_dir / "af_vs_sinus_summary.csv"
    rows = []
    for grp_name in ["Sinus", "AF", "All"]:
        grp_df = hr if grp_name == "All" else groups[grp_name]
        ba = ba_stats[grp_name]
        rows.append({
            "group": grp_name,
            "n_windows": ba["n_windows"],
            "n_subjects": grp_df["sub_num"].nunique() if grp_name != "All" else 46,
            "mae_bpm": grp_df["abs_error_bpm"].mean(),
            "bias_bpm": ba["bias"],
            "sd_bpm": ba["sd"],
            "loa_lower": ba["loa_lower"],
            "loa_upper": ba["loa_upper"],
        })
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
