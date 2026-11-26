from common import DATA_PATH, load_data, validate_data


def main():
    df = load_data()
    participant_counts, stimulus_counts = validate_data(df)

    print(f"Validated: {DATA_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Participants: {df['participant_id'].nunique()}")
    print(f"Stimuli: {df['stimulus_id'].nunique()}")
    print(
        "Trials per participant (min/median/max):",
        int(participant_counts.min()),
        float(participant_counts.median()),
        int(participant_counts.max()),
    )
    print(
        "Ratings per stimulus (min/median/max):",
        int(stimulus_counts.min()),
        float(stimulus_counts.median()),
        int(stimulus_counts.max()),
    )


if __name__ == "__main__":
    main()

