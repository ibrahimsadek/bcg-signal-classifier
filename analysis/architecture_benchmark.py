#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
patch11_architecture_benchmark.py
==================================
Controlled comparison of FOUR architectures under ONE identical protocol.

WHY THIS EXISTS
---------------
Reviewer 2 asked for benchmarking against additional methods. Equally important,
the manuscript's own Limitation 8 concedes that the existing CNN-vs-Transformer
comparison is NOT controlled: the CNN used flat 5-fold CV with fixed
hyperparameters while the Transformer used nested 10-fold CV with per-fold
tuning, so the reported gap conflates architecture with optimisation budget.

Adding more models under yet another protocol would compound that problem. This
script instead runs ALL FOUR architectures -- including a fresh Transformer --
under matched folds, matched hyperparameters, matched augmentation and matched
seeds, so that the only difference between rows is the architecture.

    1D CNN              the manuscript's existing baseline
    ResNet-1D           deeper convolutional control: isolates whether attention
                        adds anything beyond depth
    BiLSTM              recurrent sequence model; the family Geng et al. (2024)
                        apply to BCG J-wave recognition
    Conv-Transformer    the proposed model, re-run under the same budget

IMPORTANT -- HOW TO READ THE TRANSFORMER ROW
--------------------------------------------
The Transformer number produced here will NOT equal the nested 10-fold value
reported in the manuscript (balanced accuracy 0.631). It cannot: that figure
came from per-fold hyperparameter tuning, and this protocol deliberately fixes
hyperparameters for all models. Report this table as a CONTROLLED ARCHITECTURAL
COMPARISON and the nested result as the tuned performance of the selected model.
Do not present this Transformer row as a reproduction of the nested result.

Preprocessing is reimplemented from the manuscript's Methods section
(Chebyshev-I 2.5-5.0 Hz zero-phase, per-chunk z-score, 50-point resample).
Absolute values may therefore differ slightly from the archived run if any
preprocessing detail is not captured in that text. This is harmless for an
architecture comparison, because all four models receive byte-identical input --
but state it in the paper rather than letting the row read as a reproduction.

USAGE (from the repo root)
--------------------------
  STEP 1 -- always do this first, it is fast and catches parsing problems:

      python patch11_architecture_benchmark.py --inspect

  STEP 2 -- the full run (10 folds x 4 models):

      python patch11_architecture_benchmark.py
      python patch11_architecture_benchmark.py --models cnn resnet bilstm transformer
      python patch11_architecture_benchmark.py --epochs 40 --folds 10

OUTPUTS (into --out_dir)
    patch11_per_fold.csv          every model x fold metric
    patch11_summary.csv           cohort means +/- SD per model
    patch11_benchmark.tex         LaTeX table, drop-in
    patch11_summary.txt           readable summary + paired tests vs Transformer
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SEED = 42
WINDOW_LEN = 50
FS = 50.0

# Fixed hyperparameters for ALL models: the modal selections of the nested run
# (dropout 0.10 in 6/10 folds, oversampling cap k=3 in 6/10, noise sigma=0.05 in
# 7/10; learning rate split 5/5, so the larger of the two is used).
HP = dict(lr=3e-4, dropout=0.10, k_cap=3.0, noise_std=0.05, batch=128)


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def read_table(path: Path) -> pd.DataFrame:
    last = None
    for sep in [",", "\t", r"\s+", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception as e:      # noqa: BLE001
            last = e
    raise SystemExit(f"Could not parse {path} ({last})")


def find_col(df: pd.DataFrame, *cands, contains=None):
    low = {c.lower().strip(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    if contains:
        for lc, orig in low.items():
            if any(t in lc for t in contains):
                return orig
    return None


def load_patient(sig_path: Path, ann_path: Path, verbose=False):
    """Return (X, y) for one patient: X (n,50) float32, y (n,) int."""
    sig = read_table(sig_path)
    ann = read_table(ann_path)

    t_col = find_col(sig, "epoch", "timestamp", "time", contains=["epoch", "time"])
    v_col = find_col(sig, "raw_data_sleepMat", "raw", "bcg", "value", "signal")
    if v_col is None:
        nums = [c for c in sig.select_dtypes("number").columns if c != t_col]
        v_col = nums[0] if nums else None
    if t_col is None or v_col is None:
        raise SystemExit(f"{sig_path.name}: could not identify time/value columns "
                         f"(found {list(sig.columns)})")

    s_col = find_col(ann, "start_time", "start", contains=["start"])
    e_col = find_col(ann, "end_time", "end", contains=["end"])
    c_col = find_col(ann, "categories", "category", "label", contains=["categ", "label", "class"])
    if None in (s_col, e_col, c_col):
        raise SystemExit(f"{ann_path.name}: could not identify start/end/category columns "
                         f"(found {list(ann.columns)})")

    t = sig[t_col].to_numpy(dtype=float)
    x = sig[v_col].to_numpy(dtype=float)

    # bandpass per Methods: Chebyshev-I, zero-phase, 2.5-5.0 Hz
    from scipy.signal import cheby1, filtfilt
    bh, ah = cheby1(2, 0.5, 2.5 / (FS / 2), btype="highpass")
    bl, al = cheby1(4, 0.5, 5.0 / (FS / 2), btype="lowpass")
    xf = filtfilt(bl, al, filtfilt(bh, ah, x))

    starts = ann[s_col].to_numpy(dtype=float)
    ends = ann[e_col].to_numpy(dtype=float)
    cats = ann[c_col].astype(str).str.strip().str.lower().to_numpy()

    # label: 1 = usable BCG, 0 = noise-dominated
    y_all = np.array([0 if ("non" in c or "noise" in c) else 1 for c in cats], dtype=int)

    order = np.argsort(t)
    t, xf = t[order], xf[order]

    X, Y, skipped = [], [], 0
    for s0, e0, lab in zip(starts, ends, y_all):
        i0, i1 = np.searchsorted(t, s0, "left"), np.searchsorted(t, e0, "left")
        seg = xf[i0:i1]
        if seg.size < 2:
            skipped += 1
            continue
        seg = (seg - seg.mean()) / (seg.std() + 1e-8)
        if seg.size != WINDOW_LEN:
            seg = np.interp(np.linspace(0, seg.size - 1, WINDOW_LEN),
                            np.arange(seg.size), seg)
        X.append(seg.astype(np.float32))
        Y.append(int(lab))
    if verbose:
        print(f"    {sig_path.name}: {len(ann)} annotations -> {len(X)} windows "
              f"({skipped} skipped), BCG={int(np.sum(Y))} NonBCG={len(Y)-int(np.sum(Y))}")
    if not X:
        raise SystemExit(f"{sig_path.name}: no usable windows extracted -- check that "
                         f"annotation times and signal timestamps share units.")
    return np.stack(X), np.asarray(Y, dtype=int)


def load_all(data_dir: Path, ann_dir: Path, verbose=False):
    sigs = sorted(data_dir.glob("*.csv"))
    if not sigs:
        raise SystemExit(f"No patient CSVs in {data_dir}")
    Xs, ys, gs = [], [], []
    for sp in sigs:
        pid = sp.stem
        cands = list(ann_dir.glob(f"*{pid}*")) or list(ann_dir.glob(f"*{pid.lstrip('0')}*"))
        if not cands:
            print(f"    [skip] no annotation file for {pid}")
            continue
        X, y = load_patient(sp, cands[0], verbose)
        Xs.append(X); ys.append(y); gs.append(np.full(len(y), pid))
    if not Xs:
        raise SystemExit("No patients loaded.")
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


# ----------------------------------------------------------------------
# Augmentation (capped, train-only) -- per Methods
# ----------------------------------------------------------------------
def capped_augment(X, y, k_cap, noise_std, rng):
    out_X, out_y = [], []
    counts = {c: int((y == c).sum()) for c in np.unique(y)}
    n_maj = max(counts.values())
    n_min = min(counts.values())
    T = int(min(n_maj, np.ceil(k_cap * n_min)))
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) >= T:
            sel = rng.choice(idx, size=T, replace=False)
            out_X.append(X[sel]); out_y.append(np.full(T, c))
            continue
        out_X.append(X[idx]); out_y.append(np.full(len(idx), c))
        need = T - len(idx)
        src = rng.choice(idx, size=need, replace=True)
        syn = X[src].copy()
        mode = rng.integers(0, 3, size=need)
        syn[mode == 0] += rng.normal(0, noise_std, size=syn[mode == 0].shape).astype(np.float32)
        if np.any(mode == 1):
            syn[mode == 1] *= rng.uniform(0.9, 1.1, size=(int((mode == 1).sum()), 1)).astype(np.float32)
        for j in np.where(mode == 2)[0]:
            syn[j] = np.roll(syn[j], int(rng.integers(-5, 6)))
        out_X.append(syn); out_y.append(np.full(need, c))
    Xa = np.concatenate(out_X); ya = np.concatenate(out_y)
    p = rng.permutation(len(ya))
    return Xa[p], ya[p]


# ----------------------------------------------------------------------
# Architectures
# ----------------------------------------------------------------------
def build(name, drop):
    import tensorflow as tf
    from tensorflow.keras import layers as L, Model, Input

    inp = Input(shape=(WINDOW_LEN, 1))

    if name == "cnn":
        x = inp
        for f, ks in [(64, 7), (64, 5), (64, 3)]:
            x = L.Conv1D(f, ks, padding="same", activation="relu")(x)
            x = L.MaxPooling1D(2, padding="same")(x)
            x = L.Dropout(drop)(x)
        x = L.GlobalAveragePooling1D()(x)
        x = L.Dense(64, activation="relu")(x)

    elif name == "resnet":
        x = L.Conv1D(64, 7, padding="same", activation="relu")(inp)
        for f in (64, 128, 128):
            sc = L.Conv1D(f, 1, padding="same")(x)
            h = L.Conv1D(f, 5, padding="same")(x)
            h = L.BatchNormalization()(h); h = L.Activation("relu")(h)
            h = L.Conv1D(f, 3, padding="same")(h)
            h = L.BatchNormalization()(h)
            x = L.Activation("relu")(L.Add()([sc, h]))
            x = L.Dropout(drop)(x)
        x = L.GlobalAveragePooling1D()(x)
        x = L.Dense(64, activation="relu")(x)

    elif name == "bilstm":
        # unroll=True disqualifies the fused cuDNN kernel and uses the generic
        # (still GPU-resident) implementation. Some TensorFlow builds have no
        # CudnnRNN kernel registered, which otherwise raises:
        #   "No OpKernel was registered to support Op 'CudnnRNN'".
        # Unrolling is safe and cheap here because the sequence is only 50 steps.
        x = L.Conv1D(64, 5, padding="same", activation="relu")(inp)
        x = L.Bidirectional(L.LSTM(64, return_sequences=True, unroll=True))(x)
        x = L.Dropout(drop)(x)
        x = L.Bidirectional(L.LSTM(32, return_sequences=True, unroll=True))(x)
        x = L.Dropout(drop)(x)
        x = L.GlobalAveragePooling1D()(x)
        x = L.Dense(64, activation="relu")(x)

    elif name == "transformer":
        x = L.Conv1D(64, 5, padding="same", activation="relu")(inp)
        x = L.Conv1D(64, 3, padding="same", activation="relu")(x)
        pos = L.Embedding(WINDOW_LEN, 64)(tf.range(WINDOW_LEN))
        x = x + pos
        for _ in range(2):
            a = L.MultiHeadAttention(num_heads=4, key_dim=16, dropout=drop)(x, x)
            x = L.LayerNormalization(epsilon=1e-6)(L.Add()([x, a]))
            f = L.Dense(128, activation="relu")(x)
            f = L.Dropout(drop)(f)
            f = L.Dense(64)(f)
            x = L.LayerNormalization(epsilon=1e-6)(L.Add()([x, f]))
        x = L.GlobalAveragePooling1D()(x)
        x = L.Dense(64, activation="relu")(x)
    else:
        raise ValueError(name)

    out = L.Dense(2)(x)
    return Model(inp, out)


def evaluate(y_true, logits):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, roc_auc_score, confusion_matrix)
    p = np.argmax(logits, axis=1)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob = (e / e.sum(axis=1, keepdims=True))[:, 1]
    try:
        auc = roc_auc_score(y_true, prob)
    except ValueError:
        auc = np.nan
    cm = confusion_matrix(y_true, p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return dict(
        acc=accuracy_score(y_true, p),
        bal_acc=balanced_accuracy_score(y_true, p),
        macro_f1=f1_score(y_true, p, average="macro", zero_division=0),
        f1_bcg=f1_score(y_true, p, pos_label=1, zero_division=0),
        f1_nonbcg=f1_score(y_true, p, pos_label=0, zero_division=0),
        auc=auc, TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Controlled four-architecture benchmark")
    ap.add_argument("--data_dir", type=Path, default=Path("data"))
    ap.add_argument("--ann_dir", type=Path, default=Path("annotations"))
    ap.add_argument("--out_dir", type=Path, default=Path("figures"))
    ap.add_argument("--models", nargs="*",
                    default=["cnn", "resnet", "bilstm", "transformer"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--inspect", action="store_true",
                    help="Load and report the data, then exit WITHOUT training")
    ap.add_argument("--resume", action="store_true",
                    help="Skip models already present in patch11_per_fold.csv")
    ap.add_argument("--cpu_fallback", action="store_true",
                    help="Run models on CPU if a GPU kernel is unavailable")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("CONTROLLED ARCHITECTURE BENCHMARK")
    print("=" * 76)
    print(f"  data        : {args.data_dir}")
    print(f"  annotations : {args.ann_dir}")
    print()
    X, y, g = load_all(args.data_dir, args.ann_dir, verbose=True)
    print()
    print(f"  windows  : {len(y):,}   shape {X.shape}")
    print(f"  patients : {len(np.unique(g))}  -> {sorted(np.unique(g))}")
    print(f"  BCG      : {int(y.sum()):,} ({100*y.mean():.1f}%)")
    print(f"  Non-BCG  : {int((1-y).sum()):,} ({100*(1-y).mean():.1f}%)")
    print()

    if args.inspect:
        print("  EXPECTED (from the manuscript): 33,946 windows; 29,666 BCG (87.4%);")
        print("                                  4,280 Non-BCG; 10 patients.")
        print()
        print("  If the counts above differ materially, the loader has mis-parsed the")
        print("  annotation or signal files -- fix that BEFORE running the benchmark,")
        print("  otherwise every model is trained on the wrong data. Send me the output.")
        print("=" * 76)
        return

    import tensorflow as tf
    from sklearn.model_selection import GroupKFold
    tf.get_logger().setLevel("ERROR")

    Xc = X[..., None]
    gkf = GroupKFold(n_splits=args.folds)

    # Results are checkpointed after EVERY model so a crash late in the run does
    # not discard the models that already completed.
    ckpt = args.out_dir / "patch11_per_fold.csv"
    rows = []
    done = set()
    if args.resume and ckpt.exists():
        prev = pd.read_csv(ckpt)
        rows = prev.to_dict("records")
        done = set(prev["model"].unique())
        print(f"  resuming: {sorted(done)} already complete ({len(prev)} fold rows)")

    for mi, mname in enumerate(args.models, 1):
        if mname in done:
            print(f"[{mi}/{len(args.models)}] {mname} -- already done, skipping")
            continue
        print(f"[{mi}/{len(args.models)}] {mname}")
        for fi, (tr, te) in enumerate(gkf.split(Xc, y, groups=g), 1):
            tf.keras.utils.set_random_seed(SEED + fi)
            rng = np.random.default_rng(SEED + fi)
            Xtr, ytr = capped_augment(X[tr], y[tr], HP["k_cap"], HP["noise_std"], rng)
            def _train_and_eval(device=None):
                ctx = tf.device(device) if device else contextlib.nullcontext()
                with ctx:
                    mdl = build(mname, HP["dropout"])
                    mdl.compile(
                        optimizer=tf.keras.optimizers.Adam(HP["lr"]),
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
                    mdl.fit(Xtr[..., None], ytr, epochs=args.epochs,
                            batch_size=HP["batch"], verbose=0)
                    return evaluate(y[te], mdl.predict(Xc[te], verbose=0))
            try:
                m = _train_and_eval()
            except tf.errors.InvalidArgumentError as exc:
                if not args.cpu_fallback:
                    raise
                print(f"    GPU kernel unavailable ({str(exc)[:70]}...), retrying on CPU")
                m = _train_and_eval("/CPU:0")
            m.update(model=mname, fold=fi, test_patient=str(np.unique(g[te])[0]),
                     n_test=len(te))
            rows.append(m)
            print(f"    fold {fi:2d} ({m['test_patient']}): bal_acc={m['bal_acc']:.3f} "
                  f"macroF1={m['macro_f1']:.3f} AUC={m['auc']:.3f}")
            tf.keras.backend.clear_session()
        # checkpoint after each completed model
        pd.DataFrame(rows).to_csv(ckpt, index=False)
        print(f"    [checkpoint] {ckpt} now holds {len(rows)} fold rows")

    df = pd.DataFrame(rows)
    df.to_csv(ckpt, index=False)
    args.models = [m for m in df["model"].unique()]   # report only what actually ran

    metrics = ["acc", "bal_acc", "macro_f1", "f1_bcg", "f1_nonbcg", "auc"]
    summ = df.groupby("model")[metrics].agg(["mean", "std"]).round(4)
    summ.to_csv(args.out_dir / "patch11_summary.csv")

    # paired tests vs the Transformer, on matched folds
    from scipy import stats as st
    lines, A = [], None
    L = []
    A = L.append
    A("=" * 78)
    A("CONTROLLED ARCHITECTURE BENCHMARK")
    A("All models: identical folds, hyperparameters, augmentation and seeds.")
    A("=" * 78)
    A("")
    A(f"  Protocol : {args.folds}-fold patient-wise GroupKFold, {args.epochs} epochs")
    A(f"  Fixed HP : lr={HP['lr']}, dropout={HP['dropout']}, "
      f"k_cap={HP['k_cap']}, noise_std={HP['noise_std']}")
    A(f"  Windows  : {len(y):,} from {len(np.unique(g))} patients "
      f"({100*y.mean():.1f}% positive)")
    A("")
    A(f"{'Model':<16}{'Acc':>9}{'Bal.Acc':>10}{'MacroF1':>10}"
      f"{'F1_BCG':>9}{'F1_non':>9}{'AUC':>9}")
    A("-" * 72)
    for m in args.models:
        d = df[df["model"] == m]
        A(f"{m:<16}" + "".join(
            f"{d[k].mean():>9.3f}" if k == 'acc' else f"{d[k].mean():>{10 if k in ('bal_acc','macro_f1') else 9}.3f}"
            for k in metrics))
        A(f"{'  +/- SD':<16}" + "".join(
            f"{d[k].std():>9.3f}" if k == 'acc' else f"{d[k].std():>{10 if k in ('bal_acc','macro_f1') else 9}.3f}"
            for k in metrics))
    A("")
    if "transformer" in args.models:
        A("PAIRED COMPARISON vs CONV-TRANSFORMER (Wilcoxon, matched folds)")
        A("-" * 78)
        A(f"{'Model':<16}{'dBalAcc':>10}{'p':>10}{'dMacroF1':>11}{'p':>10}{'dAUC':>9}{'p':>10}")
        tr = df[df["model"] == "transformer"].sort_values("fold")
        for m in args.models:
            if m == "transformer":
                continue
            o = df[df["model"] == m].sort_values("fold")
            cells = ""
            for k in ("bal_acc", "macro_f1", "auc"):
                a, b = o[k].to_numpy(), tr[k].to_numpy()
                ok = np.isfinite(a) & np.isfinite(b)
                try:
                    p = st.wilcoxon(a[ok], b[ok]).pvalue
                except ValueError:
                    p = np.nan
                cells += f"{np.mean(a[ok]-b[ok]):>{10 if k!='macro_f1' else 11}.3f}{p:>10.4f}"
            A(f"{m:<16}{cells}")
        A("")
        A("  Negative delta = the comparator is WORSE than the Transformer.")
    A("")
    A("HOW TO REPORT THIS")
    A("-" * 78)
    A("  * This table is a CONTROLLED ARCHITECTURAL comparison: one protocol, one")
    A("    hyperparameter set, matched folds. It retires the manuscript's")
    A("    Limitation 8 (CNN/Transformer confounded by optimisation budget).")
    A("  * The Transformer row here is NOT the nested 10-fold result (bal_acc")
    A("    0.631). That figure came from per-fold tuning; this protocol fixes")
    A("    hyperparameters for fairness. Report both, clearly labelled.")
    A("  * Preprocessing was reimplemented from the Methods text, so absolute")
    A("    values may differ slightly from the archived run. All four models")
    A("    received byte-identical input, so the COMPARISON is unaffected --")
    A("    but say so in the paper.")
    A("=" * 78)
    (args.out_dir / "patch11_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print()
    print("\n".join(L))

    # LaTeX
    tex = [r"% Auto-generated by patch11_architecture_benchmark.py",
           r"\begin{table}[!ht]", r"\centering", r"\small",
           r"\caption{\textbf{Controlled architectural comparison.} All four architectures "
           r"were trained and evaluated under an identical patient-wise protocol: the same "
           rf"{args.folds} \texttt{{GroupKFold}} splits, the same fixed hyperparameters "
           r"(learning rate $3\times10^{-4}$, dropout 0.10, oversampling cap $k=3$, "
           r"augmentation noise $\sigma=0.05$), the same capped train-only augmentation and "
           r"the same random seeds, so that the only difference between rows is the network. "
           r"Values are means $\pm$ SD over folds on the naturally imbalanced outer-test "
           r"partitions. Because hyperparameters are fixed for fairness, the Conv-Transformer "
           r"row here is not identical to the nested tuned result reported in "
           r"Table~\ref{tab:baselines}; the two answer different questions.}",
           r"\label{tab:architectures}", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{lcccccc}", r"\toprule",
           r"Architecture & Accuracy & Bal.\ Acc & Macro-F1 & F1$_{\text{BCG}}$ "
           r"& F1$_{\text{Non-BCG}}$ & AUC \\", r"\midrule"]
    pretty = {"cnn": "1D CNN", "resnet": "ResNet-1D", "bilstm": "BiLSTM",
              "transformer": r"\textbf{Conv-Transformer}"}
    for m in args.models:
        d = df[df["model"] == m]
        tex.append(f"{pretty.get(m, m)} & " + " & ".join(
            f"{d[k].mean():.3f} $\\pm$ {d[k].std():.3f}" for k in metrics) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (args.out_dir / "patch11_benchmark.tex").write_text("\n".join(tex), encoding="utf-8")

    print()
    for f in ["patch11_per_fold.csv", "patch11_summary.csv",
              "patch11_benchmark.tex", "patch11_summary.txt"]:
        print(f"Wrote: {args.out_dir / f}")


if __name__ == "__main__":
    main()
