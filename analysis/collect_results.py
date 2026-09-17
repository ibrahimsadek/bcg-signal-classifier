#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
collect_final_results.py
=========================
Gathers every artifact produced during this analysis into a single
`final_results\\` folder, organised by topic, with a MANIFEST describing what
each file is and which finding it supports.

Copies (never moves) -- your existing folders are left untouched.

USAGE (from D:\\Ibrahim\\bcgProject\\bcg-test):

    python collect_final_results.py
    python collect_final_results.py --include_large     (also copies per-second
                                                         prediction tables, ~GBs)
    python collect_final_results.py --dest my_folder
    python collect_final_results.py --zip               (also make final_results.zip)

WHAT IT COLLECTS
    01_internal_classification   nested CV fold reports, summary, significance tests,
                                 minority-class metrics, corrected CIs
    02_baselines_ablation        majority/threshold baselines, CNN ablation table
    03_external_validation       original grid-search HR results, per-subject MAE,
                                 AF vs sinus, half-rate diagnostic
    04_gate_ablation             gated vs ungated sweep, per-condition summaries
    05_gate_decomposition        block-level vs interval-level decomposition
    06_selection_value           permutation tests, coverage curves, heuristics
    07_figures                   every figure (png/pdf) produced
    08_source_scripts            the patch scripts, so results are reproducible
    99_raw_block_tables          per-30s block tables per condition (moderate size)
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

# (destination subfolder, source glob, note for the manifest)
RULES = [
    # ---- 01 internal classification -------------------------------------
    ("01_internal_classification", "cv_output_nested_v2/cv_summary_nested_transformer.tsv",
     "Per-fold nested-CV metrics (source of Table 2)"),
    ("01_internal_classification", "cv_output_nested_v2/cv_summary_nested_transformer_stats.txt",
     "Mean/SD across the 10 outer folds"),
    ("01_internal_classification", "cv_output_nested_v2/fold_*/fold_report.txt",
     "Per-fold report: hyperparameters, temperature, balancing counts, confusion matrix"),
    ("01_internal_classification", "figures/fold_metrics_detailed.csv",
     "Parsed per-fold metrics table"),
    ("01_internal_classification", "figures/patch3_minority_class_metrics.csv",
     "Per-fold minority-class (Non-BCG) performance + minority sample sizes"),
    ("01_internal_classification", "figures/patch3_minority_class_metrics.tex",
     "LaTeX table, drop-in for the manuscript"),
    ("01_internal_classification", "figures/patch3_corrected_ci.csv",
     "Nadeau-Bengio corrected CIs INCLUDING balanced accuracy"),
    ("01_internal_classification", "figures/patch3_summary.txt",
     "Readable summary: 128:1 minority-count spread, 5/10 folds near-blind"),
    ("01_internal_classification", "report.txt",
     "Significance-test battery (from fold_metrics_detailed.csv)"),
    ("01_internal_classification", "significance_tests_results.txt",
     "Significance-test battery (parsed from fold_report.txt) - independent re-derivation"),

    # ---- 02 baselines / ablation ----------------------------------------
    ("02_baselines_ablation", "ablation_results.tsv",
     "CNN ablation: 6 conditions x 5 splits (source of Table 4 and Table S9)"),
    ("02_baselines_ablation", "exp1_majority_baseline.csv",
     "Majority-class baseline per fold; also confirms 29,666/4,280 class counts"),
    ("02_baselines_ablation", "exp2_threshold_baselines.csv",
     "Handcrafted feature baselines (RMS, ZCR, spectral entropy, peak-to-peak, LR)"),

    # ---- 03 external validation -----------------------------------------
    ("03_external_validation", "Overall_info.xlsx",
     "Clinical metadata: AF/sinus labels, sex, age per subject"),
    ("03_external_validation", "grid_search/grid_search_results.csv",
     "32-combination grid search (source of Table S5)"),
    ("03_external_validation", "grid_search/grid_search_loso.csv",
     "LOSO per-subject held-out MAE (source of Table S4)"),
    ("03_external_validation", "grid_search/patient_summary.csv",
     "Per-subject coverage and MAE"),
    ("03_external_validation", "grid_search/patient_analysis.csv",
     "Per-subject analysis incl. bias and ranks"),
    ("03_external_validation", "grid_search/overall_metrics.csv",
     "Cohort totals"),
    ("03_external_validation", "grid_search/af_vs_sinus_summary.csv",
     "AF vs sinus: MAE, bias, limits of agreement (source of Table 6)"),
    ("03_external_validation", "grid_search/window_diagnostics.csv",
     "Per-recording window diagnostics"),
    ("03_external_validation", "grid_search/bad_patient_peak_strategy.csv",
     "Peak-strategy diagnostics for high-MAE recordings"),
    ("03_external_validation", "figures/patch2_half_rate_stats.csv",
     "Half-rate diagnostic: band counts and bias decomposition"),
    ("03_external_validation", "figures/patch2_half_rate_summary.txt",
     "VERDICT: half-rate PARTIALLY explains the negative bias (31.9% of it)"),
    ("03_external_validation", "figures/patch4_calibration_external.csv",
     "Calibrated vs uncalibrated gate decisions, per subject"),
    ("03_external_validation", "figures/patch4_calibration_summary.txt",
     "VERDICT: 100% decision agreement - temperature scaling cannot change argmax"),
    ("03_external_validation", "figures/patch5_selected_segments.csv",
     "Which 30-s example blocks were plotted, and by what rule"),
    ("03_external_validation", "figures/patch5_summary.txt",
     "Selection rule for the example-windows figure (for the caption)"),

    # ---- 04 gate ablation ------------------------------------------------
    ("04_gate_ablation", "figures/patch1_gate_ablation.csv",
     "Gated vs ungated: 5 conditions, MAE/coverage"),
    ("04_gate_ablation", "figures/patch1_gate_ablation.tex",
     "LaTeX table, drop-in"),
    ("04_gate_ablation", "figures/patch1_gate_ablation_summary.txt",
     "KEY RESULT: gating at beta=0.50 is +0.725 bpm WORSE than ungated (p=2.6e-11)"),

    # ---- 05 decomposition -------------------------------------------------
    ("05_gate_decomposition", "figures/patch6_common_blocks.csv",
     "Common-block decomposition vs beta=0.50, per subject"),
    ("05_gate_decomposition", "figures/patch6_common_blocks_summary.txt",
     "KEY RESULT: 99.1% of harm is INTERVAL-level (+0.718), selection is +0.007"),
    ("05_gate_decomposition", "figures_blockonly/patch6_common_blocks.csv",
     "Block-only beta=0.50 vs ungated, per subject"),
    ("05_gate_decomposition", "figures_blockonly/patch6_common_blocks_summary.txt",
     "Block-only beta=0.50: +0.007 bpm (ns) - block gating alone is neutral"),
    ("05_gate_decomposition", "figures_blockonly70/patch6_common_blocks.csv",
     "Block-only beta=0.70 vs ungated, per subject"),
    ("05_gate_decomposition", "figures_blockonly70/patch6_common_blocks_summary.txt",
     "Block-only beta=0.70: -0.239 bpm (p=0.035) - block gating helps at 66% coverage"),

    # ---- 06 selection value ----------------------------------------------
    ("06_selection_value", "figures_sel70/patch8_selection_value.csv",
     "Gate vs random/heuristics at 66% coverage, per subject"),
    ("06_selection_value", "figures_sel70/patch8_selection_value_summary.txt",
     "KEY RESULT: gate beats coverage-matched random (p<0.0005); JJ-CV ties it"),
    ("06_selection_value", "figures_sel50/patch8_selection_value.csv",
     "Same at 99% coverage (control), per subject"),
    ("06_selection_value", "figures_sel50/patch8_selection_value_summary.txt",
     "CONTROL: at 99% coverage the gate carries no selection information (p=0.957)"),
    ("06_selection_value", "figures_curves/patch9_coverage_curves.csv",
     "MAE vs coverage, 14 levels, for JJ-CV / JJ-count / random"),
    ("06_selection_value", "figures_curves/patch9_coverage_curves_summary.txt",
     "KEY RESULT: JJ-CV outperforms the gate at every coverage level; "
     "JJ-count is worse than random everywhere"),

    # ---- 08 source scripts ------------------------------------------------
    ("08_source_scripts", "patch*.py", "Analysis scripts (reproducible)"),
    ("08_source_scripts", "patch*.bat", "Run drivers"),
    ("08_source_scripts", "run_hr_pipeline.py", "Original downstream HR pipeline"),
    ("08_source_scripts", "run_hr_pipeline_blockgate.py",
     "Block-gating-only variant (adds is_bcg_peak + --block_gate_only)"),
    ("08_source_scripts", "plot_af_vs_sinus_analysis.py", "AF vs sinus plotting"),
    ("08_source_scripts", "exp7_figures_and_diagnostics.py", "Figure/diagnostic generator"),
    ("08_source_scripts", "run_improved_grid_and_analyze.py", "Grid search driver"),
]

# figures: copy every png/pdf from these folders into 07_figures
FIGURE_DIRS = ["figures", "figures_blockonly", "figures_blockonly70",
               "figures_sel50", "figures_sel70", "figures_curves",
               "figures_blockonly/patch5_blind", "figures/patch5_blind"]

# per-condition block tables
GATE_CONDITIONS = ["ungated", "beta0.30", "beta0.50", "beta0.70", "beta0.90",
                   "blockonly0.50", "blockonly0.70"]

LARGE_PATTERNS = ["per_second_predictions.csv"]


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}GB"


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect all analysis artifacts into one folder")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--dest", type=Path, default=Path("final_results"))
    ap.add_argument("--include_large", action="store_true",
                    help="Also copy per_second_predictions.csv files (very large)")
    ap.add_argument("--zip", action="store_true", help="Also produce final_results.zip")
    args = ap.parse_args()

    root: Path = args.root.resolve()
    dest: Path = (root / args.dest) if not args.dest.is_absolute() else args.dest
    dest.mkdir(parents=True, exist_ok=True)

    catalog: dict[str, tuple[str, int, str]] = {}  # dest_rel -> (src_rel, size, note)
    copied: list[str] = []                         # dest_rel of files actually written
    missing: list[tuple[str, str]] = []
    disambiguated: set[tuple[str, str]] = set()

    def copy_one(src: Path, sub: str, note: str) -> None:
        if not src.is_file():
            return
        out_dir = dest / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        # keep fold_NN prefix so the 10 fold_report.txt files don't collide
        name = src.name
        if name == "fold_report.txt":
            name = f"{src.parent.name}_fold_report.txt"
        # same filename arriving from a DIFFERENT source folder would silently
        # overwrite (e.g. patch6_common_blocks.csv exists in figures/,
        # figures_blockonly/ and figures_blockonly70/). Disambiguate by parent.
        rel = f"{sub}/{name}"
        srel = str(src.relative_to(root))
        # once a base name has been disambiguated in this subfolder, every
        # later arrival with that name must be prefixed too
        if (sub, name) in disambiguated:
            name = f"{src.parent.name}__{name}"
            rel = f"{sub}/{name}"
        elif rel in catalog and catalog[rel][0] != srel:
            disambiguated.add((sub, src.name))
            name = f"{src.parent.name}__{name}"
            rel = f"{sub}/{name}"
            # retro-rename the earlier one too, so neither is ambiguous
            old_rel = f"{sub}/{src.name}"
            if old_rel in catalog:
                o_src, o_size, o_note = catalog.pop(old_rel)
                o_parent = Path(root / o_src).parent.name
                new_old = f"{sub}/{o_parent}__{src.name}"
                catalog[new_old] = (o_src, o_size, o_note)
                op = out_dir / src.name
                if op.exists():
                    op.rename(out_dir / f"{o_parent}__{src.name}")
        target = out_dir / name
        catalog[rel] = (srel, src.stat().st_size, note)
        if target.exists() and target.stat().st_size == src.stat().st_size:
            return
        shutil.copy2(src, target)
        copied.append(rel)

    print("=" * 76)
    print("COLLECTING FINAL RESULTS")
    print("=" * 76)
    print(f"  source : {root}")
    print(f"  dest   : {dest}")
    print()

    # rule-driven copies
    for sub, pattern, note in RULES:
        hits = sorted(root.glob(pattern))
        if not hits:
            missing.append((pattern, note))
            continue
        for h in hits:
            copy_one(h, sub, note)

    # every figure
    for fd in FIGURE_DIRS:
        d = root / fd
        if not d.is_dir():
            continue
        for ext in ("*.png", "*.pdf"):
            for f in sorted(d.glob(ext)):
                tag = fd.replace("/", "_").replace("figures", "fig")
                out_dir = dest / "07_figures"
                out_dir.mkdir(parents=True, exist_ok=True)
                target = out_dir / f"{tag}__{f.name}"
                rel = f"07_figures/{target.name}"
                catalog[rel] = (str(f.relative_to(root)), f.stat().st_size, "figure")
                if not (target.exists() and target.stat().st_size == f.stat().st_size):
                    shutil.copy2(f, target)
                    copied.append(rel)

    # per-condition block tables
    for cond in GATE_CONDITIONS:
        cdir = root / "gate_ablation" / cond
        if not cdir.is_dir():
            continue
        for fname, note in [
            ("per_30s_hr.csv", f"[{cond}] per-30s block table (HR, coverage, J-J counts)"),
            ("patient_summary.csv", f"[{cond}] per-subject summary"),
            ("overall_metrics.csv", f"[{cond}] cohort totals"),
        ]:
            src = cdir / fname
            if src.is_file():
                out_dir = dest / "99_raw_block_tables"
                out_dir.mkdir(parents=True, exist_ok=True)
                target = out_dir / f"{cond}__{fname}"
                rel = f"99_raw_block_tables/{target.name}"
                catalog[rel] = (str(src.relative_to(root)), src.stat().st_size, note)
                if not (target.exists() and target.stat().st_size == src.stat().st_size):
                    shutil.copy2(src, target)
                    copied.append(rel)
        # analysis subfolder csvs
        adir = cdir / "analysis"
        if adir.is_dir():
            for f in sorted(adir.glob("*.csv")):
                out_dir = dest / "99_raw_block_tables"
                out_dir.mkdir(parents=True, exist_ok=True)
                target = out_dir / f"{cond}__analysis__{f.name}"
                rel = f"99_raw_block_tables/{target.name}"
                catalog[rel] = (str(f.relative_to(root)), f.stat().st_size,
                                f"[{cond}] analysis output")
                if not (target.exists() and target.stat().st_size == f.stat().st_size):
                    shutil.copy2(f, target)
                    copied.append(rel)

    # large files, opt-in
    if args.include_large:
        for cond in GATE_CONDITIONS:
            for pat in LARGE_PATTERNS:
                src = root / "gate_ablation" / cond / pat
                if src.is_file():
                    out_dir = dest / "99_raw_block_tables" / "large"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    target = out_dir / f"{cond}__{pat}"
                    rel = f"99_raw_block_tables/large/{target.name}"
                    catalog[rel] = (str(src.relative_to(root)), src.stat().st_size,
                                    f"[{cond}] per-second predictions")
                    if not target.exists():
                        print(f"  copying LARGE {src.relative_to(root)} "
                              f"({human(src.stat().st_size)})...")
                        shutil.copy2(src, target)
                        copied.append(rel)

    # ---------------- manifest ----------------
    total = sum(v[1] for v in catalog.values())
    lines = []
    A = lines.append
    A("FINAL RESULTS - MANIFEST")
    A("=" * 76)
    A(f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}")
    A(f"Source    : {root}")
    A(f"Files     : {len(catalog)}")
    A(f"Total size: {human(total)}")
    A("")
    A("HEADLINE FINDINGS AND WHERE THEY LIVE")
    A("-" * 76)
    A("  1. Subject-level leakage inflates benchmarks by 4-8 points")
    A("       -> 02_baselines_ablation/ablation_results.tsv")
    A("  2. The gate's minority-class detection is weak and highly heterogeneous")
    A("       (5/10 folds near-blind; 128:1 spread in minority sample size)")
    A("       -> 01_internal_classification/patch3_summary.txt")
    A("  3. Gating as published makes downstream HR WORSE (+0.725 bpm, p=2.6e-11)")
    A("       -> 04_gate_ablation/patch1_gate_ablation_summary.txt")
    A("  4. 99.1% of that harm is interval-level masking, not block rejection")
    A("       (strips 27.8% of seconds -> loses 41.6% of J-J intervals)")
    A("       -> 05_gate_decomposition/patch6_common_blocks_summary.txt")
    A("  5. Block-gating-only at beta=0.70 helps (-0.239 bpm, p=0.035)")
    A("       -> 05_gate_decomposition/figures_blockonly70 summary")
    A("  6. That gain is real, not sample shrinkage: beats coverage-matched")
    A("       random (p<0.0005); random selection is flat across all coverages")
    A("       -> 06_selection_value/figures_sel70 summary")
    A("  7. A J-J regularity statistic outperforms the gate at every coverage,")
    A("       BUT it is a post-hoc precision measure, not a pre-hoc quality gate")
    A("       (it needs the peaks the gate exists to protect; it also separates")
    A("        blocks when no quality difference exists at all)")
    A("       -> 06_selection_value/patch9_coverage_curves_summary.txt")
    A("  8. Calibrated vs uncalibrated gates agree on 100% of windows")
    A("       (temperature scaling is monotonic; cannot alter argmax)")
    A("       -> 03_external_validation/patch4_calibration_summary.txt")
    A("")
    A("KNOWN CAVEATS TO CARRY INTO THE WRITE-UP")
    A("-" * 76)
    A("  * The 4-8 point leakage figure rests on a 5-split CNN testbed; balanced")
    A("    accuracy alone reaches only p=0.13. Strengthen or soften the claim.")
    A("  * Only 15/46 subjects individually beat their own random null at 66%")
    A("    coverage; the cohort effect is consistent direction, not strong effects.")
    A("  * beta=0.90 in the ablation is NOT gate value - at 3.3% coverage the arms")
    A("    score different block subsets, so that comparison is selection.")
    A("  * Table 2 fold 4/10 k* values and Table S2 after-balancing counts were")
    A("    corrected during this analysis (folds 4 and 10 were transposed).")
    A("")
    A("FILE INDEX")
    A("=" * 76)
    cur = None
    for dst in sorted(catalog):
        src, size, note = catalog[dst]
        sub = dst.split("/")[0]
        if sub != cur:
            cur = sub
            A("")
            A(f"[{sub}]")
        A(f"  {dst.split('/', 1)[1]:<52} {human(size):>9}")
        if note and note != "figure":
            A(f"      {note}")
    if missing:
        A("")
        A("NOT FOUND (may simply not have been generated)")
        A("-" * 76)
        for pat, note in missing:
            A(f"  {pat}")
            A(f"      {note}")
    if not args.include_large:
        A("")
        A("NOT COPIED: per_second_predictions.csv (very large).")
        A("Re-run with --include_large if you need the per-second tables.")

    (dest / "MANIFEST.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"  present : {len(catalog)} files ({human(total)})")
    print(f"  newly copied: {len(copied)}")
    if missing:
        print(f"  missing : {len(missing)} pattern(s) - see MANIFEST.txt")
    print(f"  manifest: {dest / 'MANIFEST.txt'}")

    if args.zip:
        zpath = root / f"{dest.name}.zip"
        print(f"  zipping -> {zpath} ...")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in dest.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(dest.parent))
        print(f"  zip size: {human(zpath.stat().st_size)}")

    print("=" * 76)
    print("DONE. Open MANIFEST.txt first -- it maps each finding to its files.")


if __name__ == "__main__":
    main()
