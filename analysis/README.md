# Downstream analysis and revision experiments

This directory evaluates the quality gate **end to end** -- by its effect on the
heart-rate estimator it is meant to protect -- rather than by classification
metrics alone. Every script here operates on a **frozen** classifier; nothing in
this directory retrains a model.

## What you need before running

| Requirement | Used by | Notes |
| --- | --- | --- |
| `cv_output_nested_v2/fold_*/` reports | `minority_class_metrics.py` | From `main.py` training (README s.3) |
| `cv_output_nested_v2/final_model/` | gate ablation runners | model, calibrator, inference config |
| External BCG recordings | all downstream scripts | Qiu et al., figshare 10.6084/m9.figshare.28643153 |
| `Overall_info.xlsx` | `half_rate_diagnostic.py` | Rhythm labels from that dataset |
| `grid_search/` outputs | `half_rate_diagnostic.py`, `example_windows.py` | per-30s and per-second tables |

**Run every script from the repository root**, not from inside `analysis/`. All
default paths are relative to the working directory. Each script takes explicit
`--` overrides if your layout differs; use `--help`.

## Order of execution

The gate ablation must run before the analyses that consume its output.

```
scripts/run_gate_ablation.bat                       # 5 conditions, several hours
python analysis/make_block_gate_only.py             # writes the block-gate-only variant
scripts/run_block_gate_only.bat                     # 1 condition, ~1.5 h
python analysis/gate_granularity_decomposition.py
python analysis/selection_value_permutation.py --gated blockonly0.70
python analysis/coverage_curves.py
```

These need no prior run and take seconds to minutes:

```
python analysis/minority_class_metrics.py
python analysis/calibration_decision_equivalence.py
python analysis/half_rate_diagnostic.py
python analysis/example_windows.py
python analysis/architecture_benchmark.py --inspect   # verify data parsing first
python analysis/architecture_benchmark.py             # then the full run, 1.5-3 h
```

Finally, to gather everything into one folder with a manifest:

```
python analysis/collect_results.py
```

## Script index

Output files are written to `figures/` unless `--out_dir` says otherwise.

| Script | Key outputs | What it establishes |
| --- | --- | --- |
| `minority_class_metrics.py` | `patch3_minority_class_metrics.csv`, `patch3_corrected_ci.csv`, `patch3_summary.txt` | Per-fold minority-class (Non-BCG) metrics, minority sample sizes with Wilson intervals, and Nadeau-Bengio corrected CIs including balanced accuracy. |
| `calibration_decision_equivalence.py` | `patch4_calibration_external.csv`, `patch4_calibration_summary.txt` | Verifies that calibrated and uncalibrated gates emit identical decisions, since temperature scaling is monotonic and cannot move a window across a fixed threshold. |
| `half_rate_diagnostic.py` | `patch2_half_rate_stats.csv`, `patch2_half_rate_histogram.pdf` | Tests whether the systematic negative heart-rate bias is explained by half-rate detection during tachycardia. |
| `example_windows.py` | `patch5_example_windows.pdf`, `patch5_selected_segments.csv` | Exports example 30-second blocks stratified by BCG fraction under a fixed, stated selection rule; --export_blind writes an unlabelled set plus a withheld key for a second-reader agreement study. |
| `gate_ablation_summary.py` | `patch1_gate_ablation.csv`, `patch1_gate_ablation.pdf` | Aggregates the gated-vs-ungated runs into the MAE/coverage comparison and figure. |
| `gate_granularity_decomposition.py` | `patch6_common_blocks.csv`, `patch6_common_blocks.pdf` | Common-block decomposition separating block-level rejection from interval-level restriction. |
| `make_block_gate_only.py` | `downstream/run_hr_pipeline_blockgate.py` | Non-destructive patcher that generates the block-gating-only pipeline variant by separating the two roles of the per-second usability flag. Verifies every edit and aborts rather than writing a partial patch. |
| `selection_value_permutation.py` | `patch8_selection_value.csv`, `patch8_selection_value.pdf` | Tests whether the gate's block selection beats coverage-matched random selection and model-free J-J statistics. |
| `coverage_curves.py` | `patch9_coverage_curves.csv`, `patch9_coverage_curves.pdf` | Sweeps every selection strategy across coverage levels. |
| `architecture_benchmark.py` | `patch11_per_fold.csv`, `patch11_benchmark.tex` | Controlled comparison of 1D CNN, ResNet-1D, BiLSTM and Conv-Transformer under one identical patient-wise protocol. |
| `collect_results.py` | `final_results/` + `MANIFEST.txt` | Gathers every analysis artifact into a single final_results/ folder with a manifest mapping findings to files. |

## A note on the block-gating-only variant

`make_block_gate_only.py` writes `downstream/run_hr_pipeline_blockgate.py` and
never modifies the original pipeline. It separates the two roles that the
per-second usability flag previously served: driving the block-level BCG
fraction (block gating), and restricting where J-peaks may be detected inside a
retained block (interval-level restriction). Conflating those two is what caused
the gate to degrade downstream accuracy; separating them is what removed the
harm. The patcher verifies every edit, checks that no J-peak selection path is
left unpatched, syntax-checks the generated file, and aborts without writing if
any check fails.

## Interpreting the results

Two results are easy to misread and are worth stating plainly:

* **Calibrated and uncalibrated gates agree on 100% of windows.** This is the
  expected outcome, not a bug. Temperature scaling divides logits by a single
  positive constant, which is monotonic and therefore cannot move a window
  across a fixed decision threshold in binary classification.
* **Lower MAE at a stricter threshold is not automatically evidence of gate
  value.** A stricter threshold scores fewer and different blocks. Only the
  coverage-matched comparisons in `selection_value_permutation.py` support a
  like-for-like claim; at 3.3% coverage the comparison is selection, not
  performance.
