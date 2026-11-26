from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "trial_level_ratings.csv"
EXPRESSION_FEATURE_PATH = (
    ROOT / "data" / "processed" / "stimulus_expression_features.csv"
)
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"

RATING_COLUMNS = ["Artificiality", "Attractiveness", "Trustworthiness"]


def ensure_output_dirs():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    required = {
        "participant_id",
        "stimulus_id",
        "session",
        "Artificiality",
        "Attractiveness",
        "Trustworthiness",
        "response_time_sec",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for column in RATING_COLUMNS + ["response_time_sec"]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    return df


def validate_data(df):
    errors = []
    if len(df) != 50_000:
        errors.append(f"Expected 50,000 rows, found {len(df):,}.")
    if df["participant_id"].nunique() != 100:
        errors.append("Expected 100 participants.")
    if df["stimulus_id"].nunique() != 500:
        errors.append("Expected 500 stimuli.")
    if df.duplicated(["participant_id", "stimulus_id"]).any():
        errors.append("Participant-stimulus pairs are not unique.")
    if df.isna().any().any():
        errors.append("The analysis dataset contains missing values.")

    participant_counts = df.groupby("participant_id").size()
    stimulus_counts = df.groupby("stimulus_id").size()
    if not (participant_counts == 500).all():
        errors.append("Every participant must have exactly 500 trials.")
    if not (stimulus_counts == 100).all():
        errors.append("Every stimulus must have exactly 100 ratings.")

    for column in RATING_COLUMNS:
        values = df[column].to_numpy()
        if not np.isin(values, [1, 2, 3, 4, 5]).all():
            errors.append(f"{column} contains values outside integer levels 1-5.")

    if errors:
        raise ValueError("\n".join(errors))
    return participant_counts, stimulus_counts

