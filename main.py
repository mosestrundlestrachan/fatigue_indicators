from data_load import get_player_id, get_pitch_data, build_game_data_dict, build_training_dataframe, clean_game_dict
from mlearning import train_fatigue_classifier
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless for scripts
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # === HARDCODED PITCHERS AND SEASONS ===
    pitcher_names = [
        "Logan Gilbert", 
        "George Kirby", "Luis Castillo", "Justin Verlander", 
        "Clayton Kershaw", "Max Scherzer", "Gerrit Cole", "Zac Gallen", 
        "Aaron Nola", "Corbin Burnes"
    ]
    seasons = [2020, 2021, 2022, 2023, 2024]

    all_game_data = {}

    for pitcher_name in pitcher_names:
        pitcher_id = get_player_id(pitcher_name)
        if pitcher_id is None:
            print(f"⚠️ Skipping {pitcher_name}: could not find ID.")
            continue

        for season in seasons:
            print(f"Processing {pitcher_name} - {season}...")
            try:
                pitch_data = get_pitch_data(pitcher_id, season)
                if pitch_data.empty:
                    print(f"⚠️ No pitch data for {pitcher_name} in {season}")
                    continue

                # Check required columns
                missing_cols = {"inning", "estimated_woba_using_speedangle"} - set(pitch_data.columns)
                if missing_cols:
                    print(f"⚠️ Skipping {pitcher_name} {season}: missing {missing_cols}")
                    continue

                raw_game_data = build_game_data_dict(pitch_data)

                # Inject pitcher name before cleaning
                for game_df in raw_game_data.values():
                    game_df["pitcher_name"] = pitcher_name

                game_data = clean_game_dict(raw_game_data)

                print(f"✅ {pitcher_name} {season}: {len(game_data)} valid games")
                all_game_data.update(game_data)

            except Exception as e:
                print(f"❌ Failed to load data for {pitcher_name} in {season}: {e}")

    if not all_game_data:
        print("❌ No valid game data collected.")
        return

    print(f"✅ Finished gathering game data: {len(all_game_data)} total games")

    df = build_training_dataframe(all_game_data)
    if df.empty:
        print("❌ Training DataFrame is empty after merging and cleaning. Exiting.")
        return

    # Map pitcher names from raw game data
    df["pitcher_name"] = df["game_date"].map(
        lambda date: all_game_data[date]["pitcher_name"].iloc[0] if date in all_game_data else "Unknown"
    )

    # Show breakdown of fatigue labels
    print("📊 Fatigue Label Distribution in Final Training Set:")
    print(df["fatigued"].value_counts())

    # Train classifier
    model = train_fatigue_classifier(df, threshold=0.35)

    # Predict and preview
    feature_cols = ["early_ff_velocity", "early_ff_spin", "early_strike_ratio", "early_pitch_count"]
    df["fatigue_pred"] = model.predict(df[feature_cols])

    print(df[["game_date", "pitcher_name", "fatigued", "fatigue_pred"]].head(50))

    # Optional: Save results
    # df.to_csv("fatigue_predictions.csv", index=False)

if __name__ == "__main__":
    main()
