# Run improved grid search with Viterbi+3-phase peaks and analyze per-recording/LOSO results.
# Co-authored with CoCo
r"""
End-to-end script:
  1. Run grid search with --use_improved_peaks --refine_diagnostic_peaks
  2. Parse grid_search_results.csv and grid_search_loso.csv
  3. Identify which subjects/recordings drive high MAE

Usage (from your local machine):
  python run_improved_grid_and_analyze.py \
    --model_dir  D:\Ibrahim\bcgProject\bcg-test\cv_output_nested_v2\final_model \
    --input_dir  D:\Ibrahim\bcgProject\bcg-test\NewData_processed \
    --output_dir D:\Ibrahim\bcgProject\bcg-test\grid_search_improved \
    --glob "*_aligned_data_ecg.txt" \
    --fs 50 \
    --normalize_signal \
    --predict_batch_size 512
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def run_grid_search(args: argparse.Namespace) -> Path:
    """Step 1: Launch the improved grid search via run_hr_pipeline.py."""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(args.pipeline_script),
        "--model_dir", str(args.model_dir),
        "--input_dir", str(args.input_dir),
        "--output_dir", str(output_dir),
        "--glob", args.glob,
        "--normalize_signal",
        "--predict_batch_size", str(args.predict_batch_size),
    ]
    # Detection flags (conditionally added)
    flags_desc = []
    if args.use_improved_peaks:
        cmd.append("--use_improved_peaks")
        flags_desc.append("--use_improved_peaks")
    if args.refine_diagnostic_peaks:
        cmd.append("--refine_diagnostic_peaks")
        flags_desc.append("--refine_diagnostic_peaks")
    # Grid search params
    cmd.extend([
        "--grid_search",
        "--grid_objective", args.grid_objective,
        "--grid_coverage_floor", str(args.grid_coverage_floor),
        "--grid_prominence_coef", args.grid_prominence_coef,
        "--grid_bcg_fraction_threshold", args.grid_bcg_fraction_threshold,
        "--grid_min_valid_jj_intervals", args.grid_min_valid_jj_intervals,
    ])
    # Only search adaptive_thresh_mult if improved peaks is active
    if args.use_improved_peaks:
        cmd.extend(["--grid_adaptive_thresh_mult", args.grid_adaptive_thresh_mult])
    if args.fs is not None:
        cmd.extend(["--fs", str(args.fs)])

    print("=" * 70)
    print("STEP 1: Running grid search")
    print(f"  Flags: {' '.join(flags_desc) if flags_desc else '(baseline diagnostic peaks)'}")
    print(f"  Output: {output_dir}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 70)

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\nGrid search exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    return output_dir


def analyze_grid_results(output_dir: Path) -> pd.DataFrame:
    """Step 2: Parse and summarize grid_search_results.csv."""
    results_path = output_dir / "grid_search_results.csv"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Grid search may have failed.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(results_path)
    print("\n" + "=" * 70)
    print("STEP 2: Grid Search Results Summary")
    print("=" * 70)
    print(f"\nTotal combinations evaluated: {len(df)}")
    print(f"\nParameter ranges searched:")
    for col in ["prominence_coef", "adaptive_thresh_mult", "bcg_fraction_threshold", "min_valid_jj_intervals"]:
        if col in df.columns:
            vals = sorted(df[col].unique())
            print(f"  {col}: {vals}")

    # Best combos by MAE (with coverage >= floor)
    print("\n--- Top 5 combos by MAE (coverage >= 0.80) ---")
    eligible = df[df["mean_coverage"] >= 0.80].copy()
    if eligible.empty:
        print("  WARNING: No combo meets coverage >= 0.80!")
        eligible = df.copy()
        print("  Showing best by MAE regardless of coverage:")

    top5 = eligible.nsmallest(5, "mean_mae_bpm")
    for _, row in top5.iterrows():
        print(
            f"  prom={row['prominence_coef']:.3f}  at={row['adaptive_thresh_mult']:.2f}  "
            f"bt={row['bcg_fraction_threshold']:.2f}  mj={int(row['min_valid_jj_intervals'])}  "
            f"-> MAE={row['mean_mae_bpm']:.3f}  cov={row['mean_coverage']:.3f}  "
            f"paired={int(row['total_paired_windows'])}"
        )

    # Best combos by coverage
    print("\n--- Top 5 combos by coverage ---")
    top5_cov = df.nsmallest(5, "mean_mae_bpm").head(5) if eligible.empty else df.nlargest(5, "mean_coverage")
    for _, row in top5_cov.iterrows():
        print(
            f"  prom={row['prominence_coef']:.3f}  at={row['adaptive_thresh_mult']:.2f}  "
            f"bt={row['bcg_fraction_threshold']:.2f}  mj={int(row['min_valid_jj_intervals'])}  "
            f"-> MAE={row['mean_mae_bpm']:.3f}  cov={row['mean_coverage']:.3f}"
        )

    # Parameter sensitivity analysis
    print("\n--- Parameter Sensitivity (mean MAE across levels, coverage >= 0.80) ---")
    analysis_df = eligible if not eligible.empty else df
    for param in ["prominence_coef", "adaptive_thresh_mult", "bcg_fraction_threshold", "min_valid_jj_intervals"]:
        if param in analysis_df.columns:
            grouped = analysis_df.groupby(param)["mean_mae_bpm"].mean()
            print(f"\n  {param}:")
            for val, mae in grouped.items():
                print(f"    {val} -> mean MAE = {mae:.3f}")

    return df


def analyze_loso(output_dir: Path) -> pd.DataFrame | None:
    """Step 2b: Parse LOSO results to see per-subject held-out performance."""
    loso_path = output_dir / "grid_search_loso.csv"
    if not loso_path.exists():
        print("\n  LOSO file not found (only 1 subject?). Skipping LOSO analysis.")
        return None

    loso = pd.read_csv(loso_path)
    print("\n--- LOSO Cross-Validation (per held-out subject) ---")

    obj_col = [c for c in loso.columns if c.startswith("held_out_") and c.endswith("_bpm")][0] if any(
        c.startswith("held_out_") and c.endswith("_bpm") for c in loso.columns
    ) else None

    if obj_col:
        loso_sorted = loso.sort_values(obj_col, ascending=False)
        print(f"\n  Mean held-out {obj_col}: {loso[obj_col].mean():.3f}")
        print(f"  Median held-out {obj_col}: {loso[obj_col].median():.3f}")
        print(f"  Std held-out {obj_col}: {loso[obj_col].std():.3f}")
        print(f"\n  Per-subject held-out error (worst first):")
        for _, row in loso_sorted.iterrows():
            chosen = (
                f"prom={row['chosen_prominence_coef']:.3f} "
                f"at={row['chosen_adaptive_thresh_mult']:.2f} "
                f"bt={row['chosen_bcg_fraction_threshold']:.2f} "
                f"mj={int(row['chosen_min_valid_jj_intervals'])}"
            )
            print(
                f"    {row['held_out_subject']}: {obj_col}={row[obj_col]:.3f}  "
                f"cov={row.get('held_out_coverage', np.nan):.3f}  "
                f"(train-selected: {chosen})"
            )
    return loso


def identify_error_drivers(output_dir: Path, grid_results: pd.DataFrame) -> None:
    """Step 3: Identify which subjects/recordings drive high MAE.

    Uses the best config from the grid to run a quick re-evaluation breakdown.
    Since grid_search_results.csv only has per-combo means, we look at the
    grid_search_loso.csv for per-subject breakdown, and also examine any
    per-recording data available.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Identifying Error Drivers")
    print("=" * 70)

    # Load best config
    best_path = output_dir / "grid_search_best.json"
    if best_path.exists():
        import json
        best = json.loads(best_path.read_text(encoding="utf-8"))
        print(f"\n  Recommended config: {best.get('recommended_config', {})}")
        print(f"  In-sample metrics: {best.get('recommended_config_in_sample', {})}")
        print(f"  LOSO mean held-out: {best.get('loso_mean_held_out_objective', 'N/A')}")
        print(f"  N subjects: {best.get('n_subjects', '?')}")
        print(f"  N combos searched: {best.get('n_combos', '?')}")

    # LOSO analysis for subject-level drivers
    loso_path = output_dir / "grid_search_loso.csv"
    if loso_path.exists():
        loso = pd.read_csv(loso_path)
        obj_col = [c for c in loso.columns if c.startswith("held_out_") and c.endswith("_bpm")]
        if obj_col:
            obj_col = obj_col[0]
            loso_sorted = loso.sort_values(obj_col, ascending=False)

            # Identify outlier subjects (those > 1.5 IQR above Q3)
            q1 = loso[obj_col].quantile(0.25)
            q3 = loso[obj_col].quantile(0.75)
            iqr = q3 - q1
            outlier_thresh = q3 + 1.5 * iqr
            outliers = loso_sorted[loso_sorted[obj_col] > outlier_thresh]

            mean_obj = float(loso[obj_col].mean())
            median_obj = float(loso[obj_col].median())

            print(f"\n  Distribution of held-out {obj_col}:")
            print(f"    Mean:   {mean_obj:.3f}")
            print(f"    Median: {median_obj:.3f}")
            print(f"    Q1:     {q1:.3f}")
            print(f"    Q3:     {q3:.3f}")
            print(f"    IQR:    {iqr:.3f}")
            print(f"    Outlier threshold (Q3 + 1.5*IQR): {outlier_thresh:.3f}")

            if not outliers.empty:
                print(f"\n  OUTLIER SUBJECTS (driving high mean MAE):")
                for _, row in outliers.iterrows():
                    contribution = (row[obj_col] - median_obj) / len(loso)
                    print(
                        f"    {row['held_out_subject']}: {obj_col}={row[obj_col]:.3f} "
                        f"(+{row[obj_col] - median_obj:.3f} above median, "
                        f"contributes ~{contribution:.3f} to mean)"
                    )
                # Impact estimate
                without_outliers = loso[~loso["held_out_subject"].isin(outliers["held_out_subject"])]
                print(f"\n  Mean {obj_col} WITHOUT outliers: {without_outliers[obj_col].mean():.3f}")
                print(f"  Mean {obj_col} WITH outliers:    {mean_obj:.3f}")
                print(f"  Removing {len(outliers)} outlier(s) would reduce mean by ~{mean_obj - without_outliers[obj_col].mean():.3f} bpm")
            else:
                print(f"\n  No statistical outliers detected (all within 1.5*IQR of Q3).")
                print("  Error is distributed relatively evenly across subjects.")

            # Subjects where coverage is low (potential data quality issues)
            if "held_out_coverage" in loso.columns:
                low_cov = loso_sorted[loso_sorted["held_out_coverage"] < 0.70]
                if not low_cov.empty:
                    print(f"\n  LOW-COVERAGE SUBJECTS (< 70% — possible data quality issues):")
                    for _, row in low_cov.iterrows():
                        print(
                            f"    {row['held_out_subject']}: coverage={row['held_out_coverage']:.3f}  "
                            f"{obj_col}={row[obj_col]:.3f}"
                        )

    # Comparison with baseline grid search (if available)
    print("\n" + "-" * 50)
    print("  COMPARISON: Improved vs Baseline")
    print("-" * 50)
    print("  (Compare these results with your previous grid search at")
    print("   D:\\Ibrahim\\bcgProject\\bcg-test\\grid_search\\grid_search_results.csv)")
    print("  Baseline best:  MAE=14.632, coverage=0.992 (diagnostic peaks only)")
    print("  If improved MAE is lower -> the 3-phase detector + Viterbi helps.")
    print("  If similar/worse -> the error source is upstream (filtering, model, ECG ref).")

    # Recommendations
    print("\n" + "-" * 50)
    print("  RECOMMENDATIONS")
    print("-" * 50)
    if loso_path.exists():
        loso = pd.read_csv(loso_path)
        obj_col = [c for c in loso.columns if c.startswith("held_out_") and c.endswith("_bpm")]
        if obj_col:
            mean_val = loso[obj_col[0]].mean()
            if mean_val < 12.0:
                print("  Result: Significant improvement. The improved detector helps.")
                print("  Next: Run full inference with the recommended config.")
            elif mean_val < 14.0:
                print("  Result: Modest improvement (~1-2 bpm). Gains are real but limited.")
                print("  Next: Investigate worst subjects for data quality / ECG ref errors.")
            else:
                print("  Result: No meaningful improvement over baseline.")
                print("  Next: The bottleneck is likely NOT peak detection. Investigate:")
                print("    - ECG reference accuracy (heartpy double-counting?)")
                print("    - Signal quality (motion artifacts missed by the gate)")
                print("    - Bandpass filter settings (try --hp_cutoff_hz 0.5 --lp_cutoff_hz 10)")
                print("    - Whether the model's BCG/non-BCG classification is correct")


def write_analysis_report(output_dir: Path) -> None:
    """Write a consolidated text report."""
    report_path = output_dir / "improved_grid_analysis_report.txt"
    # The print statements go to stdout; we also capture key findings to file
    lines = [
        "Improved Grid Search Analysis Report",
        "=" * 40,
        f"Output directory: {output_dir}",
        "",
        "Files generated:",
        "  - grid_search_results.csv     (all combos, mean metrics)",
        "  - grid_search_loso.csv        (per-subject held-out performance)",
        "  - grid_search_best.json       (recommended config)",
        "  - improved_grid_analysis_report.txt  (this file)",
        "",
        "Key differences from baseline grid search:",
        "  - --use_improved_peaks: 3-phase detector (adaptive envelope + template + I/J/K)",
        "  - --refine_diagnostic_peaks: Viterbi path optimization for rhythm consistency",
        "  - adaptive_thresh_mult is now ACTIVE and searched over [1.0, 1.2, 1.5]",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report saved: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run improved grid search + analyze results"
    )
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--glob", type=str, default="*_aligned_data_ecg.txt")
    parser.add_argument("--fs", type=float, default=None)
    parser.add_argument("--normalize_signal", action="store_true")
    parser.add_argument("--predict_batch_size", type=int, default=512)
    parser.add_argument(
        "--pipeline_script", type=Path, default=None,
        help="Path to run_hr_pipeline.py (auto-detected if in same directory)"
    )
    # Detection algorithm flags
    parser.add_argument("--use_improved_peaks", action="store_true",
                        help="Enable 3-phase J-peak detector (adaptive envelope + template + I/J/K)")
    parser.add_argument("--refine_diagnostic_peaks", action="store_true",
                        help="Apply Viterbi path optimization to per-second peaks")
    # Grid search params
    parser.add_argument("--grid_objective", type=str, default="mae", choices=["mae", "rmse"])
    parser.add_argument("--grid_coverage_floor", type=float, default=0.80)
    parser.add_argument("--grid_prominence_coef", type=str, default="0.03,0.05,0.08,0.12")
    parser.add_argument("--grid_adaptive_thresh_mult", type=str, default="1.0,1.2,1.5")
    parser.add_argument("--grid_bcg_fraction_threshold", type=str, default="0.30,0.50,0.70,0.90")
    parser.add_argument("--grid_min_valid_jj_intervals", type=str, default="1,3")
    # Skip grid search if results already exist
    parser.add_argument(
        "--skip_grid", action="store_true",
        help="Skip grid search and only run analysis (assumes results already exist in output_dir)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Auto-detect pipeline script location
    if args.pipeline_script is None:
        candidates = [
            Path(__file__).parent / "run_hr_pipeline.py",
            args.output_dir.parent / "run_hr_pipeline.py",
        ]
        for c in candidates:
            if c.exists():
                args.pipeline_script = c
                break
        if args.pipeline_script is None:
            print("ERROR: Could not find run_hr_pipeline.py. Pass --pipeline_script explicitly.", file=sys.stderr)
            sys.exit(1)

    # Step 1: Run grid search
    if not args.skip_grid:
        run_grid_search(args)
    else:
        print("Skipping grid search (--skip_grid). Using existing results.")
        if not (args.output_dir / "grid_search_results.csv").exists():
            print(f"ERROR: {args.output_dir / 'grid_search_results.csv'} not found.", file=sys.stderr)
            sys.exit(1)

    # Step 2: Analyze grid results
    grid_results = analyze_grid_results(args.output_dir)
    analyze_loso(args.output_dir)

    # Step 3: Identify error drivers
    identify_error_drivers(args.output_dir, grid_results)

    # Write report
    write_analysis_report(args.output_dir)

    print("\n" + "=" * 70)
    print("DONE. All steps complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
