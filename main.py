from data_load import get_player_id, get_pitch_data, build_game_data_dict, build_training_dataframe, combine_game_data, clean_game_dict
from mlearning import train_fatigue_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    pitcher_name = input("Enter the pitcher's name (First Last): ")
    seasons = input("Enter comma-separated MLB season years (e.g., 2022,2023,2024): ")

    pitcher_id = get_player_id(pitcher_name)
    if pitcher_id is None:
        return

    all_game_data = {}

    for season in map(int, seasons.split(",")):
        print(f"Processing {season}...")
        pitch_data = get_pitch_data(pitcher_id, season)
        game_data = build_game_data_dict(pitch_data)
        game_data = clean_game_dict(game_data)
        all_game_data.update(game_data)  # Merge into one dictionary

    df = build_training_dataframe(all_game_data)

    drops = df["velocity_drop_ff"].dropna()
    threshold = np.percentile(drops, 75)
    df["fatigued"] = df["velocity_drop_ff"] >= threshold


    model = train_fatigue_model(df)

    # make predictions
    X = df[["early_ff_velocity", "early_ff_spin", "early_strike_ratio", "early_pitch_count"]]
    df["fatigue_pred"] = model.predict(X)

    print(df.head(49))

    # Save to CSV
    #filename = f"{pitcher_name.replace(' ', '_').lower()}_fatigue_predictions.csv"
    #df.to_csv(filename, index=False)

    #for saving to csv
    #df.to_csv("logan_gilbert_fatigue_training_data.csv", index=False)

    #df = velocity_trends_by_pitch(game_data)
    #sns.lineplot(data=df, x="inning", y="avg_velocity", hue="pitch_type")
    #plt.title("Velocity by Inning and Pitch Type")
    #plt.show()

    #print(f"Collected {len(games_and_pitches)} games for {pitcher_name} in {season}.")
    #first_game_date = list(games_and_pitches.keys())[0]
    #test_game_date = list(games_and_pitches.keys())[-1]
    #print(f"This game's data (date: {first_game_date}):")
    #print(games_and_pitches[first_game_date].head()) #for testing
    #pd.set_option("display.max_columns", None)
    #print(pitch_data.tail())
    #print(pitch_data.info()) testing

if __name__ == "__main__":
    main()