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

## 3) Running the pipeline

Transformer:
```bash
python main.py --model transformer
```

CNN:
```bash
python main.py --model cnn
```

Custom output directory:
```bash
python main.py --out_dir ./cv_output_nested --model transformer
```

Disable explainability (faster):
```bash
python main.py --model transformer --disable_xai
```

See all parameters:
```bash
python main.py --help
```

---

## 4) What the script does (validated against the code)

### 4.1 Signal preprocessing (per patient)
Defaults are defined in the script `Config`:
- Sampling rate: `fs = 50 Hz`
- Chebyshev Type-I **high-pass**: order `2`, ripple `0.5 dB`, cutoff `2.5 Hz`
- Chebyshev Type-I **low-pass**: order `4`, ripple `0.5 dB`, cutoff `5.0 Hz`
- Zero-phase filtering: `scipy.signal.filtfilt`
- Fixed resample length: `target_len = 50`

### 4.2 Chunk extraction (uses **all** annotated intervals)
For each annotation interval `[start_time, end_time]`:
1. Find samples in the patient CSV where `epoch` lies within the interval
2. Extract the filtered BCG segment
3. Z-score normalize the segment
4. Resample to `target_len=50` points (linear interpolation)

The script logs:
- total annotation rows
- total chunks extracted
- number of empty intervals (no samples in epoch range)

### 4.3 Leakage prevention (patient-wise splitting everywhere)
The pipeline uses patient IDs as **groups** and enforces disjointness:

- **Outer CV:** `GroupKFold(n_splits=outer_splits)` (default `10`)
- **Inner CV (nested):** `GroupKFold(n_splits=inner_splits)` (default `3`) on *outer-train only*
- **Calibration split:** `GroupShuffleSplit(test_size=calib_frac)` (default `0.2`) on *outer-train only*

The script asserts no patient overlap between the relevant splits.

### 4.4 Hyperparameter selection (inner loop)
Grid search over:
- learning rate candidates: `--lrs` (default `1e-4,3e-4`)
- dropout candidates: `--dropouts` (default `0.1,0.2`)
- capped oversampling factor: `--max_oversample_factors` (default `2.0,3.0`)
- augmentation noise std: `--noise_stds` (default `0.03,0.05`)

The grid is truncated to `--hp_max_trials` (default `12`).
Selection criterion: best mean inner-fold **F1**.

### 4.5 Augmentation (Option B: capped, train-only)
Augmentation is applied **only** to the training subset:
- In inner-CV: inner-train only
- In outer-CV: train-fit only (never calibration or test)

Transforms (configured in `Config`):
- Gaussian noise (`noise_std` tuned)
- multiplicative scaling (`scale_min=0.90` to `scale_max=1.10`)
- circular roll/shift (`max_roll_frac=0.10` of `target_len`)
- optional majority downsampling (`downsample_majority=True`)

### 4.6 Calibration (post-hoc probability reliability)
Choice via `--calibration`:
- `temperature` (default): temperature scaling on logits (optimized with `calib_opt_steps=250`, `calib_opt_lr=0.05`)
- `platt`: logistic regression on logit margin
- `none`: skip calibration

Reported metrics **before and after** calibration:
- Brier score
- ECE (expected calibration error)
- NLL (negative log-likelihood)

Reliability diagrams are saved per fold.

### 4.7 Explainable AI (Integrated Gradients)
Enabled unless `--disable_xai` is passed.
Per fold:
- sample up to `--xai_examples_per_fold` test examples (default `32`)
- compute IG with `--ig_steps` (default `64`)
- save plots into `fold_XX/xai_ig/`

---

## 5) Outputs

`--out_dir` (default `./cv_output_nested`) will contain:

- `counts_before_overall.png`
- `counts_after_overall_<model>.png`
- `cv_summary_nested_<model>.tsv`
- `cv_summary_nested_<model>_stats.txt`
- `fold_01/`, `fold_02/`, ... per-fold artifacts:
  - `reliability_uncal.png`, `reliability_cal.png`
  - `xai_ig/` (if enabled)
  - fold report files (metrics + patient split lists)

---

## 6) Parameters (defaults from the script)

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

---

## 7) Troubleshooting

### Silence oneDNN informational message (if present)
```bat
set TF_ENABLE_ONEDNN_OPTS=0
```

### Speed
Integrated Gradients adds extra gradient passes; disable XAI for faster runs:
```bash
python main.py --model transformer --disable_xai
```
