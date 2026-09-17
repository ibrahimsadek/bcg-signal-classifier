# Generate publication figures: IG heatmaps, confusion matrices, ROC/PR curves, training curves
# Co-authored with CoCo
"""
EXP-7: Publication Figures & Diagnostic Plots
==============================================
Generates publication figures for the Frontiers in Physiology revision:
  1. Per-fold confusion matrices (2x5 grid)
  2. ROC and Precision-Recall curves (composited from per-fold images)
  3. Training/validation curves (if available)
  4. Integrated Gradients (IG) heatmap (composited from per-fold xai_ig/*.png)
  5. Reliability diagrams (2-row layout: uncal top, cal bottom)
  6. Calibration comparison bar chart

Usage:
    python exp7_figures_and_diagnostics.py --cv_dir cv_output_nested_v2 --out_dir figures
    python exp7_figures_and_diagnostics.py --cv_dir cv_output_nested_v2 --out_dir figures --only rel
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pathlib import Path


# =============================================================================
# SECTION 1: PARSE EXISTING FOLD REPORTS
# =============================================================================

def parse_fold_report(fold_report_path):
    with open(fold_report_path, "r") as f:
        text = f.read()

    result = {}

    for key in ["acc", "prec", "rec", "f1",
                "brier_uncal", "ece_uncal", "nll_uncal",
                "brier_cal", "ece_cal", "nll_cal",
                "temperature_T"]:
        match = re.search(rf"^{key}:\s*([\d.]+)", text, re.MULTILINE)
        if match:
            result[key] = float(match.group(1))

    cm_match = re.search(
        r"confusion_matrix:\s*\r?\n\[\[\s*(\d+)\s+(\d+)\]\s*\r?\n?\s*\[\s*(\d+)\s+(\d+)\]\]",
        text
    )
    if cm_match:
        tn, fp, fn, tp = int(cm_match.group(1)), int(cm_match.group(2)), \
                         int(cm_match.group(3)), int(cm_match.group(4))
        result["cm"] = np.array([[tn, fp], [fn, tp]])
        result["tn"], result["fp"], result["fn"], result["tp"] = tn, fp, fn, tp
    else:
        cm_section = re.search(r"confusion_matrix:\s*\r?\n(.*?\]\])", text, re.DOTALL)
        if cm_section:
            nums = [int(x) for x in re.findall(r"\d+", cm_section.group(1))]
            if len(nums) == 4:
                tn, fp, fn, tp = nums
                result["cm"] = np.array([[tn, fp], [fn, tp]])
                result["tn"], result["fp"], result["fn"], result["tp"] = tn, fp, fn, tp

    pat_match = re.search(r"test_patients:\s*\['(\d+)'\]", text)
    if pat_match:
        result["test_patient"] = pat_match.group(1)

    hp_match = re.search(r"best_hp:\s*\{(.+?)\}", text)
    if hp_match:
        result["best_hp"] = hp_match.group(1)

    return result


def load_all_fold_reports(cv_dir):
    cv_path = Path(cv_dir)
    fold_dirs = sorted(cv_path.glob("fold_*"))
    reports = []
    for fold_dir in fold_dirs:
        report_path = fold_dir / "fold_report.txt"
        if report_path.exists():
            report = parse_fold_report(report_path)
            report["fold"] = int(fold_dir.name.split("_")[1])
            report["fold_dir"] = fold_dir
            reports.append(report)
    return reports


# =============================================================================
# SECTION 2: CONFUSION MATRICES
# =============================================================================

def plot_confusion_matrices(reports, out_dir, class_names=("Non-BCG", "BCG")):
    n_folds = len(reports)
    n_cols = 5
    n_rows = (n_folds + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for i, report in enumerate(reports):
        ax = axes_flat[i]
        if "cm" not in report:
            ax.set_visible(False)
            continue

        cm = report["cm"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

        for row in range(2):
            for col in range(2):
                count = cm[row, col]
                pct = cm_norm[row, col] * 100
                ax.text(col, row, f"{count}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=9,
                        color="white" if cm_norm[row, col] > 0.6 else "black")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names, fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)

        patient = report.get("test_patient", "?")
        f1 = report.get("f1", 0)
        ax.set_title(f"Fold {report['fold']} (Patient {patient})\nF1={f1:.3f}",
                     fontsize=9, fontweight="bold")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Per-Fold Confusion Matrices \u2014 Nested 10-Fold Patient-Wise CV\n"
                 "(Rows: True class, Columns: Predicted class)",
                 fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    save_path = Path(out_dir) / "confusion_matrices.pdf"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    save_png = Path(out_dir) / "confusion_matrices.png"
    fig.savefig(save_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {save_path}")
    print(f"  [SAVED] {save_png}")
    return save_path


# =============================================================================
# SECTION 3: SUMMARY TABLE
# =============================================================================

def generate_summary_table(cv_dir, out_dir, reports):
    cv_path = Path(cv_dir)
    tsv_path = cv_path / "  .tsv"

    if tsv_path.exists():
        df = pd.read_csv(tsv_path, sep="\t")
    else:
        rows = []
        for r in reports:
            rows.append({
                "fold": r["fold"],
                "acc": r.get("acc", np.nan),
                "prec": r.get("prec", np.nan),
                "rec": r.get("rec", np.nan),
                "f1": r.get("f1", np.nan),
                "test_patients": r.get("test_patient", ""),
            })
        df = pd.DataFrame(rows)

    for r in reports:
        fold_mask = df["fold"] == r["fold"]
        if "cm" in r:
            df.loc[fold_mask, "TP"] = r["tp"]
            df.loc[fold_mask, "FP"] = r["fp"]
            df.loc[fold_mask, "TN"] = r["tn"]
            df.loc[fold_mask, "FN"] = r["fn"]
            df.loc[fold_mask, "N_test"] = r["tp"] + r["fp"] + r["tn"] + r["fn"]

    if "TP" in df.columns:
        df["bal_acc"] = 0.5 * (df["TP"] / (df["TP"] + df["FN"]) +
                               df["TN"] / (df["TN"] + df["FP"]))

    print("\n" + "=" * 90)
    print("PER-FOLD METRICS (from existing fold reports)")
    print("=" * 90)
    display_cols = ["fold", "test_patients", "N_test", "TP", "FP", "TN", "FN",
                    "acc", "bal_acc", "prec", "rec", "f1"]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string(index=False, float_format="%.4f"))

    pm_sd = "\u00b1 SD"
    print(f"\n{'Metric':<12} {'Mean':>10} {pm_sd:>10} {'Min':>10} {'Max':>10}")
    print("-" * 45)
    for col in ["acc", "bal_acc", "prec", "rec", "f1"]:
        if col in df.columns:
            m, s = df[col].mean(), df[col].std()
            mn, mx = df[col].min(), df[col].max()
            print(f"{col:<12} {m:>10.4f} {s:>10.4f} {mn:>10.4f} {mx:>10.4f}")

    csv_path = Path(out_dir) / "fold_metrics_detailed.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  [SAVED] {csv_path}")

    latex_path = Path(out_dir) / "fold_metrics_table.tex"
    with open(latex_path, "w") as f:
        f.write("% Auto-generated per-fold metrics table\n")
        f.write("\\begin{table}[!ht]\n\\centering\\small\n")
        f.write("\\caption{Per-fold metrics. Each fold holds out one patient.}\n")
        f.write("\\label{tab:detailed_metrics}\n")
        f.write("\\begin{tabular}{clrrrrrcccccc}\n\\toprule\n")
        f.write("Fold & Patient & $N$ & TP & FP & TN & FN & "
                "Acc & Bal.Acc & Prec & Rec & F1 \\\\\n")
        f.write("\\midrule\n")
        for _, row in df.iterrows():
            patient = row.get("test_patients", "?")
            n_test = int(row["N_test"]) if "N_test" in row and not pd.isna(row.get("N_test")) else "--"
            tp = int(row["TP"]) if "TP" in row and not pd.isna(row.get("TP")) else "--"
            fp = int(row["FP"]) if "FP" in row and not pd.isna(row.get("FP")) else "--"
            tn = int(row["TN"]) if "TN" in row and not pd.isna(row.get("TN")) else "--"
            fn = int(row["FN"]) if "FN" in row and not pd.isna(row.get("FN")) else "--"
            f.write(f"  {int(row['fold'])} & {patient} & {n_test} & "
                    f"{tp} & {fp} & {tn} & {fn} & "
                    f"{row['acc']:.3f} & {row.get('bal_acc', np.nan):.3f} & "
                    f"{row['prec']:.3f} & {row['rec']:.3f} & "
                    f"{row['f1']:.3f} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"  \\textbf{{Mean}} & -- & -- & -- & -- & -- & -- & "
                f"\\textbf{{{df['acc'].mean():.3f}}} & "
                f"\\textbf{{{df.get('bal_acc', pd.Series([0])).mean():.3f}}} & "
                f"\\textbf{{{df['prec'].mean():.3f}}} & "
                f"\\textbf{{{df['rec'].mean():.3f}}} & "
                f"\\textbf{{{df['f1'].mean():.3f}}} \\\\\n")
        f.write("\\botrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  [SAVED] {latex_path}")
    return df


# =============================================================================
# SECTION 4: ROC AND PR CURVES (from existing images)
# =============================================================================

def plot_roc_and_pr_curves(cv_dir, out_dir):
    fold_dirs = sorted(Path(cv_dir).glob("fold_*"))

    roc_images = []
    pr_images = []
    fold_labels = []

    for fold_dir in fold_dirs[:10]:
        roc_path = fold_dir / "roc_curve.png"
        pr_path = fold_dir / "precision_recall_curve.png"
        if roc_path.exists():
            roc_images.append(roc_path)
        if pr_path.exists():
            pr_images.append(pr_path)
        if roc_path.exists() or pr_path.exists():
            fold_num = int(fold_dir.name.split("_")[1])
            fold_labels.append(f"Fold {fold_num}")

    if not roc_images and not pr_images:
        print("  [SKIP] No per-fold roc_curve.png or precision_recall_curve.png found")
        return None

    if roc_images:
        n = len(roc_images)
        n_cols = 5
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 4.5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()
        for i, img_path in enumerate(roc_images):
            ax = axes_flat[i]
            img = imread(str(img_path))
            ax.imshow(img, aspect="auto")
            ax.set_title(fold_labels[i], fontsize=10, fontweight="bold")
            ax.axis("off")
        for j in range(len(roc_images), len(axes_flat)):
            axes_flat[j].set_visible(False)
        plt.suptitle("Per-Fold ROC Curves", fontsize=13, fontweight="bold", y=0.98)
        plt.subplots_adjust(wspace=0.02, hspace=0.15)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        save_path = Path(out_dir) / "roc_curves_grid.pdf"
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        fig.savefig(Path(out_dir) / "roc_curves_grid.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [SAVED] {save_path}")

    if pr_images:
        n = len(pr_images)
        n_cols = 5
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 4.5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()
        for i, img_path in enumerate(pr_images):
            ax = axes_flat[i]
            img = imread(str(img_path))
            ax.imshow(img, aspect="auto")
            ax.set_title(fold_labels[i], fontsize=10, fontweight="bold")
            ax.axis("off")
        for j in range(len(pr_images), len(axes_flat)):
            axes_flat[j].set_visible(False)
        plt.suptitle("Per-Fold Precision-Recall Curves", fontsize=13, fontweight="bold", y=0.98)
        plt.subplots_adjust(wspace=0.02, hspace=0.15)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        save_path = Path(out_dir) / "pr_curves_grid.pdf"
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        fig.savefig(Path(out_dir) / "pr_curves_grid.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [SAVED] {save_path}")

    return save_path


# =============================================================================
# SECTION 5: RELIABILITY DIAGRAMS (2-row layout: uncal top, cal bottom)
# =============================================================================

def composite_reliability_diagrams(cv_dir, out_dir):
    fold_dirs = sorted(Path(cv_dir).glob("fold_*"))

    uncal_images = []
    cal_images = []
    fold_labels = []

    for fold_dir in fold_dirs[:10]:
        uncal_path = fold_dir / "reliability_uncal.png"
        cal_path = fold_dir / "reliability_cal.png"
        if uncal_path.exists() and cal_path.exists():
            uncal_images.append(uncal_path)
            cal_images.append(cal_path)
            fold_num = int(fold_dir.name.split("_")[1])
            fold_labels.append(f"Fold {fold_num}")

    if not uncal_images:
        print("  [SKIP] No reliability_uncal.png / reliability_cal.png found")
        return None

    n = len(uncal_images)
    n_cols = 5

    def _make_rel_figure(uncal_imgs, cal_imgs, labels, suffix=""):
        n_f = len(uncal_imgs)
        fig, axes = plt.subplots(2, n_cols, figsize=(22, 8))

        for i in range(n_f):
            ax = axes[0, i]
            img = imread(str(uncal_imgs[i]))
            ax.imshow(img, aspect="auto")
            ax.set_title(labels[i], fontsize=10, fontweight="bold")
            ax.axis("off")
        for j in range(n_f, n_cols):
            axes[0, j].set_visible(False)

        for i in range(n_f):
            ax = axes[1, i]
            img = imread(str(cal_imgs[i]))
            ax.imshow(img, aspect="auto")
            ax.set_title(labels[i], fontsize=10, fontweight="bold")
            ax.axis("off")
        for j in range(n_f, n_cols):
            axes[1, j].set_visible(False)

        axes[0, 0].text(-0.05, 0.5, "Uncalibrated", transform=axes[0, 0].transAxes,
                        fontsize=11, fontweight="bold", va="center", ha="right",
                        rotation=90)
        axes[1, 0].text(-0.05, 0.5, "Calibrated\n(T-scaling)", transform=axes[1, 0].transAxes,
                        fontsize=11, fontweight="bold", va="center", ha="right",
                        rotation=90)

        plt.suptitle("Reliability Diagrams \u2014 Uncalibrated vs Calibrated" + suffix,
                     fontsize=13, fontweight="bold", y=0.98)
        plt.subplots_adjust(wspace=0.03, hspace=0.12)
        plt.tight_layout(rect=[0.03, 0, 1, 0.94])
        return fig

    if n <= 5:
        fig = _make_rel_figure(uncal_images, cal_images, fold_labels)
        save_path = Path(out_dir) / "reliability_diagrams_comparison.pdf"
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        fig.savefig(Path(out_dir) / "reliability_diagrams_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [SAVED] {save_path}")
    else:
        fig1 = _make_rel_figure(uncal_images[:5], cal_images[:5],
                                fold_labels[:5], " (Folds 1\u20135)")
        save_path = Path(out_dir) / "reliability_diagrams_comparison_1.pdf"
        fig1.savefig(save_path, dpi=200, bbox_inches="tight")
        fig1.savefig(Path(out_dir) / "reliability_diagrams_comparison_1.png", dpi=150, bbox_inches="tight")
        plt.close(fig1)
        print(f"  [SAVED] {save_path}")

        fig2 = _make_rel_figure(uncal_images[5:], cal_images[5:],
                                fold_labels[5:], " (Folds 6\u201310)")
        save_path2 = Path(out_dir) / "reliability_diagrams_comparison_2.pdf"
        fig2.savefig(save_path2, dpi=200, bbox_inches="tight")
        fig2.savefig(Path(out_dir) / "reliability_diagrams_comparison_2.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  [SAVED] {save_path2}")

    return save_path


# =============================================================================
# SECTION 6: TRAINING CURVES
# =============================================================================

def plot_training_curves(cv_dir, out_dir):
    fold_dirs = sorted(Path(cv_dir).glob("fold_*"))
    histories = []

    for fold_dir in fold_dirs[:10]:
        hist_path = fold_dir / "training_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                histories.append((fold_dir.name, json.load(f)))

    if not histories:
        # Try image fallback
        train_images = []
        fold_labels = []
        for fold_dir in fold_dirs[:10]:
            for name in ["training_curve.png", "loss_curve.png", "training_curves.png"]:
                p = fold_dir / name
                if p.exists():
                    train_images.append(p)
                    fold_labels.append(f"Fold {int(fold_dir.name.split('_')[1])}")
                    break
        if not train_images:
            print("  [SKIP] No training_history.json or training curve images found")
            return None

        n = len(train_images)
        n_cols = 5
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 4.5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()
        for i, img_path in enumerate(train_images):
            ax = axes_flat[i]
            img = imread(str(img_path))
            ax.imshow(img, aspect="auto")
            ax.set_title(fold_labels[i], fontsize=10, fontweight="bold")
            ax.axis("off")
        for j in range(len(train_images), len(axes_flat)):
            axes_flat[j].set_visible(False)
        plt.suptitle("Per-Fold Training Curves", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        save_path = Path(out_dir) / "training_curves_grid.pdf"
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  [SAVED] {save_path}")
        return save_path

    # JSON-based training curves
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, (fold_name, hist) in enumerate(histories):
        epochs = range(1, len(hist.get("loss", [])) + 1)
        c = colors[i]
        fold_num = int(fold_name.split("_")[1])
        if "loss" in hist:
            ax_loss.plot(epochs, hist["loss"], color=c, alpha=0.35, linewidth=0.8)
        if "val_loss" in hist:
            ax_loss.plot(epochs, hist["val_loss"], color=c, alpha=0.85,
                         linewidth=1.3, linestyle="--", label=f"Fold {fold_num}")
        acc_key = next((k for k in hist if "accuracy" in k and "val" not in k), None)
        val_acc_key = next((k for k in hist if "val" in k and "accuracy" in k), None)
        if acc_key:
            ax_acc.plot(epochs, hist[acc_key], color=c, alpha=0.35, linewidth=0.8)
        if val_acc_key:
            ax_acc.plot(epochs, hist[val_acc_key], color=c, alpha=0.85,
                        linewidth=1.3, linestyle="--", label=f"Fold {fold_num}")

    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training & Validation Loss", fontweight="bold")
    ax_loss.legend(fontsize=7.5, ncol=2); ax_loss.grid(True, alpha=0.3)
    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Training & Validation Accuracy", fontweight="bold")
    ax_acc.legend(fontsize=7.5, ncol=2); ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(out_dir) / "training_curves.pdf"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {save_path}")
    return save_path


# =============================================================================
# SECTION 7: INTEGRATED GRADIENTS (from existing xai_ig images)
# =============================================================================

def plot_ig_heatmap(cv_dir, out_dir, n_examples=8, fold_idx=8):
    fold_dirs = sorted(Path(cv_dir).glob("fold_*"))
    ig_data = []

    for fold_dir in fold_dirs[:10]:
        ig_dir = fold_dir / "xai_ig"
        if not ig_dir.exists():
            continue
        ig_pngs = sorted(ig_dir.glob("*.png"))
        if ig_pngs:
            fold_num = int(fold_dir.name.split("_")[1])
            ig_data.append((fold_num, ig_pngs))

    if not ig_data:
        print("  [SKIP] No xai_ig/*.png images found in any fold")
        return None

    # Pick fold 9 (best F1) if available
    target_fold = None
    for fold_num, pngs in ig_data:
        if fold_num == 9:
            target_fold = (fold_num, pngs)
            break
    if target_fold is None:
        target_fold = max(ig_data, key=lambda x: len(x[1]))

    best_fold_num, best_fold_pngs = target_fold

    n_show = min(n_examples, len(best_fold_pngs))
    step = max(1, len(best_fold_pngs) // n_show)
    selected_pngs = best_fold_pngs[::step][:n_show]

    # 2-column grid
    n_cols = 2
    n_rows = (n_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
    axes_flat = axes.flatten()

    for i, img_path in enumerate(selected_pngs):
        ax = axes_flat[i]
        img = imread(str(img_path))
        ax.imshow(img)
        ax.set_title(img_path.stem, fontsize=8)
        ax.axis("off")
    for j in range(len(selected_pngs), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle(f"Integrated Gradients \u2014 Fold {best_fold_num} (Representative Examples)\n"
                 "Red = supports predicted class | Blue = opposes",
                 fontsize=11, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    save_path = Path(out_dir) / "ig_heatmap_composite.pdf"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    fig.savefig(Path(out_dir) / "ig_heatmap_composite.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {save_path}")

    # Multi-fold overview
    if len(ig_data) > 1:
        n_folds_ig = len(ig_data)
        n_cols_ov = 5
        n_rows_ov = (n_folds_ig + n_cols_ov - 1) // n_cols_ov
        fig3, axes3 = plt.subplots(n_rows_ov, n_cols_ov, figsize=(20, 4 * n_rows_ov))
        if n_rows_ov == 1:
            axes3 = axes3.reshape(1, -1)
        axes3_flat = axes3.flatten()
        for i, (fold_num, pngs) in enumerate(ig_data[:10]):
            ax = axes3_flat[i]
            img = imread(str(pngs[0]))
            ax.imshow(img)
            ax.set_title(f"Fold {fold_num}", fontsize=9, fontweight="bold")
            ax.axis("off")
        for j in range(len(ig_data), len(axes3_flat)):
            axes3_flat[j].set_visible(False)
        plt.suptitle("IG Heatmap \u2014 One Example Per Fold",
                     fontsize=12, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        overview_path = Path(out_dir) / "ig_heatmap_per_fold.pdf"
        fig3.savefig(overview_path, dpi=200, bbox_inches="tight")
        plt.close(fig3)
        print(f"  [SAVED] {overview_path}")

    return save_path


# =============================================================================
# SECTION 8: CALIBRATION COMPARISON BAR CHART
# =============================================================================

def plot_calibration_comparison(reports, out_dir):
    folds = [r["fold"] for r in reports]
    ece_uncal = [r.get("ece_uncal", 0) for r in reports]
    ece_cal = [r.get("ece_cal", 0) for r in reports]
    brier_uncal = [r.get("brier_uncal", 0) for r in reports]
    brier_cal = [r.get("brier_cal", 0) for r in reports]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(folds))
    width = 0.35

    ax1.bar(x - width/2, ece_uncal, width, label="Uncalibrated", color="#4472C4", alpha=0.8)
    ax1.bar(x + width/2, ece_cal, width, label="Calibrated (T-scaling)", color="#ED7D31", alpha=0.8)
    ax1.set_xlabel("Fold"); ax1.set_ylabel("ECE")
    ax1.set_title("Expected Calibration Error", fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(folds)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x - width/2, brier_uncal, width, label="Uncalibrated", color="#4472C4", alpha=0.8)
    ax2.bar(x + width/2, brier_cal, width, label="Calibrated (T-scaling)", color="#ED7D31", alpha=0.8)
    ax2.set_xlabel("Fold"); ax2.set_ylabel("Brier Score")
    ax2.set_title("Brier Score", fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(folds)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = Path(out_dir) / "calibration_comparison.pdf"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {save_path}")
    return save_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EXP-7: Generate publication figures for BCG quality classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cv_dir", default="./cv_output_nested_v2")
    parser.add_argument("--out_dir", default="./figures")
    parser.add_argument("--mode", choices=["existing", "full"], default="existing")
    parser.add_argument("--only", choices=["cm", "roc", "loss", "ig", "table", "cal", "rel"])
    parser.add_argument("--ig_fold", type=int, default=8)
    parser.add_argument("--ig_n", type=int, default=8)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("EXP-7: PUBLICATION FIGURES & DIAGNOSTICS")
    print("=" * 70)
    print(f"  CV directory : {args.cv_dir}")
    print(f"  Output dir   : {args.out_dir}")
    print(f"  Mode         : {args.mode}")
    print()

    reports = load_all_fold_reports(args.cv_dir)
    if not reports:
        print("[ERROR] No fold_report.txt found in any fold directory.")
        print(f"        Checked: {args.cv_dir}/fold_*/fold_report.txt")
        sys.exit(1)

    print(f"  Found {len(reports)} fold reports.\n")

    if args.only:
        targets = [args.only]
    else:
        targets = ["table", "cm", "cal", "rel", "roc", "loss", "ig"]

    if "table" in targets:
        print("[1] Generating per-fold summary table...")
        generate_summary_table(args.cv_dir, args.out_dir, reports)
        print()

    if "cm" in targets:
        print("[2] Generating confusion matrices...")
        plot_confusion_matrices(reports, args.out_dir)
        print()

    if "cal" in targets:
        print("[3] Generating calibration comparison (bar chart)...")
        plot_calibration_comparison(reports, args.out_dir)
        print()

    if "rel" in targets:
        print("[4] Generating reliability diagrams (uncal vs cal)...")
        composite_reliability_diagrams(args.cv_dir, args.out_dir)
        print()

    if "roc" in targets:
        print("[5] Generating ROC and Precision-Recall curves...")
        plot_roc_and_pr_curves(args.cv_dir, args.out_dir)
        print()

    if "loss" in targets:
        print("[6] Generating training/validation curves...")
        plot_training_curves(args.cv_dir, args.out_dir)
        print()

    if "ig" in targets:
        print("[7] Generating Integrated Gradients heatmap...")
        plot_ig_heatmap(args.cv_dir, args.out_dir,
                        n_examples=args.ig_n, fold_idx=args.ig_fold)
        print()

    print("=" * 70)
    print("DONE")
    print("=" * 70)
    generated = [f for f in Path(args.out_dir).glob("*") if f.is_file()]
    for f in sorted(generated):
        print(f"  {f}")


if __name__ == "__main__":
    main()
