#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Entry point for BCG signal classification pipeline."""

import os
from pathlib import Path
from bcg_signal_classifier.pipeline import main

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO + WARNING logs


def ensure_data_dirs():
    """Create data and annotations directories if they don't exist."""
    project_dir = Path(".")
    data_dir = project_dir / "data"
    ann_dir = project_dir / "annotations"

    for d in (data_dir, ann_dir):
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Created missing directory: {d.resolve()}")
        else:
            print(f"[OK] Found directory: {d.resolve()}")

    # Check if data dir has any CSV files
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"[WARNING] No CSV files found in {data_dir.resolve()}")
        print("          Place your patient files (e.g. 0001.csv, 0002.csv, ...) in the data/ folder.")
    else:
        valid = [f for f in csv_files if f.stem.isdigit() and len(f.stem) == 4]
        print(f"[OK] Found {len(valid)} patient CSV file(s) in data/")

    ann_files = list(ann_dir.glob("*.txt"))
    if not ann_files:
        print(f"[WARNING] No annotation files found in {ann_dir.resolve()}")
        print("          Place your annotation files in the annotations/ folder.")
    else:
        print(f"[OK] Found {len(ann_files)} annotation file(s) in annotations/")

    return len(csv_files) > 0 and len(ann_files) > 0


if __name__ == "__main__":
    ready = ensure_data_dirs()
    if not ready:
        print("\n[!] Please add your data files before running the pipeline.")
        print("    Then re-run: python main.py --model cnn")
    else:
        print("\nStarting pipeline...\n")
        main()
