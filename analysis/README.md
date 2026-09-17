# Downstream analysis and revision experiments

This directory holds the analyses added during revision. They evaluate the
quality gate **end to end** -- that is, by its effect on the heart-rate
estimator it is meant to protect -- rather than by classification metrics alone.

Everything here operates on a **frozen** classifier. No model is retrained by
any script in this directory.

## Order of execution

The gate ablation must run before the analyses that consume its output.

```
scripts/run_gate_ablation.bat                    # 5 conditions, hours
python analysis/make_block_gate_only.py          # generates the variant
scripts/run_block_gate_only.bat                  # 1 condition, ~1.5 h
python analysis/gate_granularity_decomposition.py
python analysis/selection_value_permutation.py --gated blockonly0.70
python analysis/coverage_curves.py
```

These require no prior run and take seconds:

```
python analysis/minority_class_metrics.py
python analysis/calibration_decision_equivalence.py
python analysis/half_rate_diagnostic.py
python analysis/example_windows.py
```

## Script index

| Script | Produces | Purpose |
| --- | --- | --- |
| `minority_class_metrics.py` | see manuscript | Per-fold minority-class (Non-BCG) metrics, minority sample sizes with Wilson intervals, and Nadeau-Bengio corrected CIs including balanced accuracy. Produces Table 5 and Supplementary Table S9. |
| `calibration_decision_equivalence.py` | see manuscript | Verifies that calibrated and uncalibrated gates emit identical decisions, since temperature scaling is monotonic and cannot move a window across a fixed threshold. |
| `half_rate_diagnostic.py` | see manuscript | Tests whether the systematic negative heart-rate bias is explained by half-rate detection during tachycardia. |
| `example_windows.py` | see manuscript | Exports example 30-second blocks stratified by BCG fraction under a fixed, stated selection rule; --export_blind writes an unlabelled set plus a withheld key for a second-reader agreement study. |
| `gate_ablation_summary.py` | see manuscript | Aggregates the gated-vs-ungated runs into the MAE/coverage comparison and figure. Produces Table 6. |
| `gate_granularity_decomposition.py` | see manuscript | Common-block decomposition separating block-level rejection from interval-level restriction. Produces Table 7 and Figure 5 -- the paper's central mechanism result. |
| `make_block_gate_only.py` | see manuscript | Non-destructive patcher that generates the block-gating-only pipeline variant by separating the two roles of the per-second usability flag. Verifies every edit and aborts rather than writing a partial patch. |
| `selection_value_permutation.py` | see manuscript | Tests whether the gate's block selection beats coverage-matched random selection and model-free J-J statistics. Produces Table 8. |
| `coverage_curves.py` | see manuscript | Sweeps every selection strategy across coverage levels. Produces Figure 6. |
| `architecture_benchmark.py` | see manuscript | Controlled comparison of 1D CNN, ResNet-1D, BiLSTM and Conv-Transformer under one identical patient-wise protocol. |
| `collect_results.py` | see manuscript | Gathers every analysis artifact into a single final_results/ folder with a manifest mapping findings to files. |

## A note on the block-gating-only variant

`make_block_gate_only.py` writes `downstream/run_hr_pipeline_blockgate.py` and
never modifies the original pipeline. It separates the two roles that the
per-second usability flag previously served: driving the block-level BCG
fraction (block gating) and restricting where J-peaks may be detected inside a
retained block (interval-level restriction). Conflating those two is what
caused the gate to degrade downstream accuracy; separating them is what fixed
it. The patcher verifies every edit, checks that no J-peak selection path is
left unpatched, syntax-checks the result, and aborts without writing if any
check fails.
