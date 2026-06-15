# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-06-14

### Initial release (manuscript submission version)

This is the version accompanying the SoftwareX manuscript submission:
"Explainable Ballistocardiogram Signal Classification with Transformers."

### Features
- Nested patient-wise cross-validation (10 outer, 3 inner folds) with runtime
  leakage assertion at every split.
- Capped augmentation (Option-B): bounded minority oversampling with noise,
  scaling, and circular shift transforms; optional majority downsampling.
- Two model architectures: 1D CNN and convolutional Transformer.
- Post-hoc probability calibration: temperature scaling and Platt scaling,
  evaluated with Brier score, ECE, and NLL.
- Integrated Gradients explainability on held-out test samples.
- Model persistence and inference mode (`--predict`) for deployment on new
  recordings (annotation-interval and sliding-window modes).
- Baseline/ablation experiments (`--ablation`) toggling split mode, augmentation
  strategy, and calibration method.
- CI pipeline: flake8 linting + pytest.
- Deterministic execution: all RNGs seeded (seed=42) with per-fold derivation.

### Repository structure
- `bcg_signal_classifier/` — core package (14 modules)
- `tests/` — unit tests for augmentation, calibration, models, persistence,
  and preprocessing
- `.github/workflows/ci.yml` — automated CI
