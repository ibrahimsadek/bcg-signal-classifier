# -*- coding: utf-8 -*-
"""Visualization functions for BCG signal classification."""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def plot_counts(counts: Dict[str, int], title: str, out_path: Path) -> None:
    """Plot and save class counts as a bar chart.

    Args:
        counts: Dictionary mapping class names to counts.
        title: Plot title.
        out_path: Path to save the plot.
    """
    fig, ax = plt.subplots()
    ax.bar(list(counts.keys()), list(counts.values()))
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Number of chunks")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
