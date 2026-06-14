# Transformer-Based Ballistocardiogram Signal Classification with Explainable AI
**BCG vs Non-BCG** classification pipeline with **nested patient-wise cross-validation**, **train-only capped augmentation**, **probability calibration**, and **Integrated Gradients** explanations.

Main entry point: `main.py`

---

## 1) Project structure

```
bcg-signal-classifier/
├─ main.py                        # Entry point
├─ bcg_signal_classifier/         # Python package
│  ├─ __init__.py
│  ├─ config.py                   # Configuration dataclass
│  ├─ preprocessing.py            # Signal filtering & preprocessing
│  ├─ augmentation.py             # Data augmentation
│  ├─ models.py                   # CNN & Transformer models
│  ├─ calibration.py              # Probability calibration
│  ├─ xai.py                      # Integrated Gradients XAI
│  ├─ visualization.py            # Plotting utilities
│  ├─ dataset.py                  # Data loading
│  └─ pipeline.py                 # Main pipeline orchestration
├─ data/                          # Patient CSV files (not in repo)
│  ├─ 0001.csv
│  ├─ 0002.csv
│  └─ ...
└─ annotations/                   # Annotation files (not in repo)
   ├─ patient__0001__annotations.txt
   ├─ patient__0002__annotations.txt
   └─ ...
```

### 1.1 `data\<PID>.csv` format
Required columns:
- `epoch` (ms timestamp)
- `raw_data_sleepMat` (raw BCG channel)

Example:
```csv
epoch,raw_data_sleepMat,PulseOximeter_results
1473817327008,5502,115
1473817327028,5481,115
```

### 1.2 `annotations\patient__<PID>__annotations.txt` format
The script uses `pandas.read_csv(sep=None, engine="python")`, so the file can be TSV or CSV but must contain:

- `chunk_id`
- `categories` (labels: `BCG`, `NonBCG`, `Non-BCG`, `Non_BCG` ; case-insensitive)
- `start_time` (ms)
- `end_time` (ms)

Example (TSV):
```tsv
chunk_id	categories	start_time	end_time
1	NonBCG	1473817327008	1473817327988
2	BCG	1473817328008	1473817328988
```

---

## 2) Installation (Python 3.9+)

Create a virtual environment and install dependencies:

**Windows:**
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Verify TensorFlow backend
```bat
python -c "import tensorflow as tf; print('TF', tf.__version__); print('GPUs', tf.config.list_physical_devices('GPU'))"
```

If you use DirectML, you should see GPU devices listed as `DML`.

> If you ever see NumPy 2.x incompatibility errors with TF 2.10, ensure `numpy<2` is installed (pinned in `requirements.txt`).

---

## 3) How to train models

Training runs **nested patient-wise cross-validation** for unbiased evaluation
and, when finished, fits **one deployable model on all data** and saves it for
later inference (see Section 4).

**Step 1 — place your data.** Put patient recordings in `data/` and matching
annotations in `annotations/` (formats in Section 1). Running `main.py` creates
these folders automatically if missing and reports how many files it found.

**Step 2 — run training.**

Transformer (default):
```bash
python main.py --model transformer
```

CNN:
```bash
python main.py --model cnn
```

Custom output directory:
```bash
python main.py --model transformer --out_dir ./cv_output_nested
```

Faster smoke test (smaller grid, fewer epochs, no explainability):
```bash
python main.py --model cnn --inner_splits 3 --inner_epochs 6 --epochs 12 --hp_max_trials 6 --disable_xai
```

See all parameters:
```bash
python main.py --help
```

**Step 3 — collect results.** Cross-validation metrics, plots, and per-fold
reports are written to `--out_dir` (Section 6). The trained deployable model is
saved to `--out_dir/final_model/` unless you pass `--no_save_final_model`.

> Nested CV is intentionally thorough and can be slow. Use the smoke-test command
> above while validating your data, then run the full configuration.

---

## 4) How to apply pretrained models

After training, a ready-to-use model is stored in `final_model/` containing:

- `model.keras` — the trained network (logits output),
- `calibrator.json` — the fitted probability calibrator (temperature/Platt/none),
- `inference_config.json` — the exact preprocessing settings and label map.

Run inference on a **new recording** with `--predict`. Preprocessing
(filtering, z-scoring, resampling) is reproduced automatically from
`inference_config.json`, so you only provide the raw CSV.

**Option A — classify annotated intervals** (same unit as training; `categories`
optional — if present, accuracy is reported):
```bash
python main.py --predict \
  --model_dir ./cv_output_nested/final_model \
  --input ./new_patient.csv \
  --annotations ./new_patient_annotations.txt \
  --output ./predictions.csv
```

**Option B — annotation-free sliding window** (for deployment on raw streams):
```bash
python main.py --predict \
  --model_dir ./cv_output_nested/final_model \
  --input ./new_patient.csv \
  --window_sec 1.0 --stride_sec 0.5 \
  --output ./predictions.csv
```

The output CSV has one row per segment:

| Column | Meaning |
|---|---|
| `chunk_id` | Interval id (annotation mode) or window index |
| `start_time`, `end_time` | Segment bounds (ms epoch) |
| `pred_index` | Predicted class index (`0`=NonBCG, `1`=BCG) |
| `pred_label` | Human-readable predicted label |
| `prob_BCG` | **Calibrated** probability of the BCG class |
| `prob_BCG_uncal` | Uncalibrated (softmax) probability of the BCG class |
| `true_label`, `correct` | Only if `categories` were supplied |

You can also call inference directly: `python -m bcg_signal_classifier.inference --model_dir ... --input ...`.

---

## 5) What the script does (validated against the code)

### 5.1 Signal preprocessing (per patient)
Defaults are defined in the script `Config`:
- Sampling rate: `fs = 50 Hz`
- Chebyshev Type-I **high-pass**: order `2`, ripple `0.5 dB`, cutoff `2.5 Hz`
- Chebyshev Type-I **low-pass**: order `4`, ripple `0.5 dB`, cutoff `5.0 Hz`
- Zero-phase filtering: `scipy.signal.filtfilt`
- Fixed resample length: `target_len = 50`

### 5.2 Chunk extraction (uses **all** annotated intervals)
For each annotation interval `[start_time, end_time]`:
1. Find samples in the patient CSV where `epoch` lies within the interval
2. Extract the filtered BCG segment
3. Z-score normalize the segment
4. Resample to `target_len=50` points (linear interpolation)

The script logs:
- total annotation rows
- total chunks extracted
- number of empty intervals (no samples in epoch range)

### 5.3 Leakage prevention (patient-wise splitting everywhere)
The pipeline uses patient IDs as **groups** and enforces disjointness:

- **Outer CV:** `GroupKFold(n_splits=outer_splits)` (default `10`)
- **Inner CV (nested):** `GroupKFold(n_splits=inner_splits)` (default `3`) on *outer-train only*
- **Calibration split:** `GroupShuffleSplit(test_size=calib_frac)` (default `0.2`) on *outer-train only*

The script asserts no patient overlap between the relevant splits.

### 5.4 Hyperparameter selection (inner loop)
Grid search over:
- learning rate candidates: `--lrs` (default `1e-4,3e-4`)
- dropout candidates: `--dropouts` (default `0.1,0.2`)
- capped oversampling factor: `--max_oversample_factors` (default `2.0,3.0`)
- augmentation noise std: `--noise_stds` (default `0.03,0.05`)

The grid is truncated to `--hp_max_trials` (default `12`).
Selection criterion: best mean inner-fold **F1**.

### 5.5 Augmentation (Option B: capped, train-only)
Augmentation is applied **only** to the training subset:
- In inner-CV: inner-train only
- In outer-CV: train-fit only (never calibration or test)

Transforms (configured in `Config`):
- Gaussian noise (`noise_std` tuned)
- multiplicative scaling (`scale_min=0.90` to `scale_max=1.10`)
- circular roll/shift (`max_roll_frac=0.10` of `target_len`)
- optional majority downsampling (`downsample_majority=True`)

### 5.6 Calibration (post-hoc probability reliability)
Choice via `--calibration`:
- `temperature` (default): temperature scaling on logits (optimized with `calib_opt_steps=250`, `calib_opt_lr=0.05`)
- `platt`: logistic regression on logit margin
- `none`: skip calibration

Reported metrics **before and after** calibration:
- Brier score
- ECE (expected calibration error)
- NLL (negative log-likelihood)

Reliability diagrams are saved per fold.

### 5.7 Explainable AI (Integrated Gradients)
Enabled unless `--disable_xai` is passed.
Per fold:
- sample up to `--xai_examples_per_fold` test examples (default `32`)
- compute IG with `--ig_steps` (default `64`)
- save plots into `fold_XX/xai_ig/`

---

## 6) Outputs

`--out_dir` (default `./cv_output_nested`) will contain:

- `counts_before_overall.png`
- `counts_after_overall_<model>.png`
- `cv_summary_nested_<model>.tsv`
- `cv_summary_nested_<model>_stats.txt`
- `fold_01/`, `fold_02/`, ... per-fold artifacts:
  - `reliability_uncal.png`, `reliability_cal.png`
  - `xai_ig/` (if enabled)
  - fold report files (metrics + patient split lists)
- `final_model/` — the deployable model for inference (Section 4), unless
  `--no_save_final_model` is set:
  - `model.keras`, `calibrator.json`, `inference_config.json`

---

## 7) Parameters (defaults from the script)

| Flag | Default |
|---|---:|
| `--project_dir` | `.` |
| `--out_dir` | `./cv_output_nested` |
| `--model` | `transformer` (`cnn` or `transformer`) |
| `--outer_splits` | `10` |
| `--inner_splits` | `3` |
| `--epochs` | `20` |
| `--inner_epochs` | `8` |
| `--batch_size` | `128` |
| `--lrs` | `1e-4,3e-4` |
| `--dropouts` | `0.1,0.2` |
| `--max_oversample_factors` | `2.0,3.0` |
| `--noise_stds` | `0.03,0.05` |
| `--hp_max_trials` | `12` |
| `--calibration` | `temperature` (`temperature`, `platt`, `none`) |
| `--calib_frac` | `0.2` |
| `--calib_opt_steps` | `250` |
| `--calib_opt_lr` | `0.05` |
| `--disable_xai` | (flag) |
| `--xai_examples_per_fold` | `32` |
| `--ig_steps` | `64` |
| `--no_save_final_model` | (flag) skip saving the deployable model |
| `--final_model_dir` | `<out_dir>/final_model` |

**Inference flags (`--predict` mode, see Section 4):**

| Flag | Default | Meaning |
|---|---:|---|
| `--model_dir` | (required) | Saved model directory (`final_model/`) |
| `--input` | (required) | Patient CSV (`epoch`, `raw_data_sleepMat`) |
| `--annotations` | `""` | Annotation file with intervals to classify |
| `--window_sec` | `0.0` | Sliding-window length (s); annotation-free mode |
| `--stride_sec` | `1.0` | Sliding-window hop (s) |
| `--output` | `predictions.csv` | Output CSV path |

---

## 8) Troubleshooting

### Silence oneDNN informational message (if present)
```bat
set TF_ENABLE_ONEDNN_OPTS=0
```

### Speed
Integrated Gradients adds extra gradient passes; disable XAI for faster runs:
```bash
python main.py --model transformer --disable_xai
```

---

## 9) Baselines and ablation experiments

To quantify the benefit of the pipeline's design choices, `--ablation` runs a
**flat (non-nested) cross-validation with fixed hyperparameters** while toggling
one design choice at a time. Each run appends a summary row (mean ± std per
metric over folds) to `ablation_output/ablation_results.tsv`. The validated
nested pipeline is untouched, so the main reported results are unaffected.

Toggles:

| Flag | Choices | Contrast |
|---|---|---|
| `--split_mode` | `patient` (default), `record` | leakage-safe vs. leaked (random) splitting |
| `--aug_mode` | `capped` (default), `plain`, `none` | capped vs. plain oversampling vs. no augmentation |
| `--calibration` | `temperature` (default), `platt`, `none` | calibrated vs. uncalibrated probabilities |

Each run reports `acc`, `f1`, `auc`, `brier`, `ece`, `nll` — both uncalibrated
(`uncal_*`) and, when calibration is enabled, calibrated (`cal_*`).

Example comparison set (run on your data):
```bash
# A) Leakage-safe + capped aug + calibration (matches the proposed design)
python main.py --ablation --split_mode patient --aug_mode capped --calibration temperature --label proposed

# B) Leaked (record-wise) splitting — shows inflated, optimistic metrics
python main.py --ablation --split_mode record  --aug_mode capped --calibration temperature --label leaked_split

# C) No augmentation
python main.py --ablation --split_mode patient --aug_mode none   --calibration temperature --label no_aug

# D) Plain (uncapped) oversampling
python main.py --ablation --split_mode patient --aug_mode plain  --calibration temperature --label plain_oversample

# E) Uncalibrated (also reported as uncal_* in every run)
python main.py --ablation --split_mode patient --aug_mode capped --calibration none        --label uncalibrated
```

Other knobs: `--model` (`cnn`/`transformer`), `--splits` (folds, default `5`),
`--epochs`, `--lr`, `--dropout`, `--max_oversample_factor`, `--noise_std`,
`--out_dir`. See `python main.py --ablation --help`.
