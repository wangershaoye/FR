"""Manuscript correlation figures (Figures 3 and 4), without face images.

The plotting code lives in :mod:`manuscript_figures`, the single source of
truth shared with ``notebooks/figures.ipynb`` so the released figures and the
inline notebook figures are identical and match the manuscript house style.
"""
import pandas as pd

from common import FIGURE_DIR, TABLE_DIR, ensure_output_dirs
import manuscript_figures as mf


def run():
    ensure_output_dirs()
    corr_path = TABLE_DIR / "participant_zero_order_spearman.csv"
    partial_path = TABLE_DIR / "participant_partial_spearman.csv"
    if not corr_path.exists() or not partial_path.exists():
        from correlations import run as run_correlations

        run_correlations()

    corr = pd.read_csv(corr_path)
    partial = pd.read_csv(partial_path)
    mf.save_figure(mf.figure_zero_order(corr), "participant_spearman_distributions", FIGURE_DIR)
    mf.save_figure(mf.figure_partial_joint(partial), "partial_correlation_joint_distribution", FIGURE_DIR)
    print(f"Figures written to {FIGURE_DIR}")


if __name__ == "__main__":
    run()
