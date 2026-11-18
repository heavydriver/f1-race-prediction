"""Shared helpers for the F1 race prediction notebooks.

All notebooks import from here so they use the same seeds, splits, metrics,
and plot theme.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
SEEDS = [1, 7, 42, 99, 2026]

TRAIN_YEARS = list(range(2014, 2024))  # 2014-2023
VAL_YEAR = 2024
TEST_YEAR = 2025

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "driver_race_features.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"

TARGET = "finish_position"

# ----------------------------------------------------------------------------
# Plot theme (light background, Formula 1 reds)
# ----------------------------------------------------------------------------
F1_RED = "#E10600"
F1_CARBON = "#15151E"
F1_PALETTE = [F1_RED, "#FF5A4F", "#B30500", "#FF9A94", "#6E0300", F1_CARBON]
SPLIT_COLORS = {"train": "#FF9A94", "val": "#FF5A4F", "test": F1_RED}
F1_CMAP = "Reds"


def set_f1_theme():
    """Apply the shared light F1-red plot theme."""
    sns.set_theme(style="whitegrid", palette=F1_PALETTE)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#FFF8F7",
        "axes.edgecolor": "#E5C9C7",
        "grid.color": "#F2DCDA",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelcolor": F1_CARBON,
        "text.color": F1_CARBON,
        "xtick.color": F1_CARBON,
        "ytick.color": F1_CARBON,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def save_fig(fig, notebook_name, file_name):
    """Save a figure to figures/<notebook_name>/<file_name>.png."""
    out_dir = FIGURES_DIR / notebook_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{file_name}.png")


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_data(path=DATA_PATH):
    """Load the driver-race feature table."""
    return pd.read_csv(path)


def split_by_year(df):
    """Chronological split: train 2014-2023, val 2024, test 2025."""
    train = df[df["year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    val = df[df["year"] == VAL_YEAR].reset_index(drop=True)
    test = df[df["year"] == TEST_YEAR].reset_index(drop=True)
    return train, val, test


# ----------------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------------
def to_race_ranks(df, score_col, tiebreak_col="grid"):
    """Turn a score (lower = better) into a finishing order 1..N per race.

    Works for the grid baseline and for model regression outputs alike.
    Ties are broken by `tiebreak_col` (missing values go last).
    """
    order = df.sort_values(
        ["raceId", score_col, tiebreak_col], na_position="last", kind="mergesort"
    )
    ranks = order.groupby("raceId").cumcount() + 1
    return ranks.reindex(df.index).astype(int)


# ----------------------------------------------------------------------------
# Metrics (computed per race, then averaged over races)
# ----------------------------------------------------------------------------
def _race_scores(actual, pred):
    actual = np.asarray(actual)
    pred = np.asarray(pred)
    pred_podium = set(np.where(pred <= 3)[0])
    true_podium = set(np.where(actual <= 3)[0])
    return {
        "accuracy": np.mean(actual == pred),
        "mae": np.mean(np.abs(actual - pred)),
        "podium_recall": len(pred_podium & true_podium) / 3,
        "spearman": spearmanr(actual, pred).statistic,
    }


def per_race_metrics(df, pred_col, target_col=TARGET):
    """One row of metrics per race (useful for plots)."""
    rows = []
    for race_id, race in df.groupby("raceId"):
        scores = _race_scores(race[target_col], race[pred_col])
        scores.update({
            "raceId": race_id,
            "year": race["year"].iloc[0],
            "round": race["round"].iloc[0],
            "race_name": race["race_name"].iloc[0],
        })
        rows.append(scores)
    cols = ["raceId", "year", "round", "race_name",
            "accuracy", "mae", "podium_recall", "spearman"]
    return pd.DataFrame(rows)[cols].sort_values(["year", "round"]).reset_index(drop=True)


def evaluate(df, pred_col, target_col=TARGET):
    """Average of the per-race metrics -> dict."""
    races = per_race_metrics(df, pred_col, target_col)
    return races[["accuracy", "mae", "podium_recall", "spearman"]].mean().to_dict()


def summarize_seeds(results_list):
    """Mean and std of each metric across seed runs (list of dicts)."""
    results = pd.DataFrame(results_list)
    return results.agg(["mean", "std"]).T
