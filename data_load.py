import pandas as pd
import numpy as np
from pybaseball import statcast_pitcher, playerid_lookup
from constants import OPENING_DAY
import os


def cached_get_pitch_data(pitcher_id: str, season: int) -> pd.DataFrame:
    filename = f"cache/{pitcher_id}_{season}.csv"
    if os.path.exists(filename):
        print(f"Loading cached data for {season}...")
        return pd.read_csv(filename)
    else:
        print(f"Fetching data for {season}...")
        df = get_pitch_data(pitcher_id, season)
        os.makedirs("cache", exist_ok=True)
        df.to_csv(filename, index=False)
        return df


def get_player_id(pitcher_name: str) -> int:
    """
    Take a pitcher's name in the format "First Last" and return their MLBAM ID.
    """
    parts = pitcher_name.split()
    if len(parts) < 2:
        print("Please enter both first and last name.")
        return None
    first, last = parts[0], parts[1]
    result = playerid_lookup(last, first)
    if result.empty:
        print(f"Could not find player ID for {pitcher_name}")
        return None
    return int(result.iloc[0]["key_mlbam"])

def get_pitch_data(pitcher_id: int, season: int) -> pd.DataFrame:
    """
    Returns all pitch-by-pitch data for the given pitcher and season from Statcast.
    """
    start_date = f"{season}-03-20"
    end_date = f"{season}-10-05"
    df = statcast_pitcher(start_date, end_date, pitcher_id)

    if season in OPENING_DAY:
        df = df[df["game_date"] >= OPENING_DAY[season]]

    return df


def build_game_data_dict(pitch_data: pd.DataFrame) -> dict:
    """
    Splits pitch data into a dictionary of game DataFrames keyed by game date.
    """
    game_dict = {}
    for game_date, group in pitch_data.groupby("game_date"):
        game_df = group.reset_index(drop=True)
        game_df.index += 1
        game_df.index.name = "pitch_number"
        game_df["game_id"] = game_date
        game_dict[game_date] = game_df
    return game_dict


def clean_game_dict(game_dict: dict) -> dict:
    """
    Ensures all entries in game_dict are valid pandas DataFrames 
    with required columns like 'inning'. Removes any malformed games.
    """
    cleaned = {}
    for date, df in game_dict.items():
        if isinstance(df, pd.DataFrame) and not df.empty and "inning" in df.columns:
            cleaned[date] = df
    return cleaned


def combine_game_data(game_dict: dict) -> pd.DataFrame:
    """
    Combines all individual game DataFrames in a game_dict into a single DataFrame.

    Adds a 'game_date' column if not already present and resets the index.
    
    Parameters:
    - game_dict: dict of {game_date: game_df}
    
    Returns:
    - A single combined DataFrame with all pitches.
    """
    all_pitches = []

    for game_date, df in game_dict.items():
        df = df.copy()
        if "game_date" not in df.columns:
            df["game_date"] = game_date
        all_pitches.append(df)

    combined_df = pd.concat(all_pitches, ignore_index=True)
    return combined_df


def avg_velocity_by_pitch_and_inning(game_df):
    """
    Returns a DataFrame showing average velocity grouped by inning and pitch type.
    """
    return (
        game_df.groupby(["inning", "pitch_type"])["release_speed"]
        .mean()
        .unstack()  # makes pitch types into columns
        .sort_index()
    )

# not currently using
def velocity_trends_by_pitch(game_dict):
    """
    Builds a long-form DataFrame of average velocity for each pitch type per inning per game.
    """
    rows = []
    for game_date, df in game_dict.items():
        grouped = df.groupby(["inning", "pitch_type"])["release_speed"].mean()
        for (inning, pitch_type), avg_vel in grouped.items():
            rows.append({
                "game_date": game_date,
                "inning": inning,
                "pitch_type": pitch_type,
                "avg_velocity": avg_vel
            })
    return pd.DataFrame(rows)


def spin_rate_drop_by_pitch(game_df, early_innings=[1, 2, 3], late_innings=[5, 6, 7, 8, 9]):
    early = game_df[game_df["inning"].isin(early_innings)]
    late = game_df[game_df["inning"].isin(late_innings)]

    early_avg = early.groupby("pitch_type")["release_spin_rate"].mean()
    late_avg = late.groupby("pitch_type")["release_spin_rate"].mean()

    common_types = early_avg.index.intersection(late_avg.index)
    drop = (early_avg[common_types] - late_avg[common_types])
    return drop.to_dict()

def strike_to_ball_ratio(game_df):
    strike_keywords = ["strike", "foul"]
    is_strike = game_df["description"].str.contains("|".join(strike_keywords), case=False, na=False)
    is_ball = game_df["description"].str.lower() == "ball"

    num_strikes = is_strike.sum()
    num_balls = is_ball.sum()

    if num_balls == 0:
        if num_strikes == 0:
            return 0.0
        return float("inf")
    return num_strikes / num_balls

#for choosing what xwoba to use for fatigue target
def average_xwoba_for_two_run_innings(game_dict: dict) -> float:
    inning_xwobas = []

    for game_df in game_dict.values():
        if "inning" not in game_df.columns or "estimated_woba_using_speedangle" not in game_df.columns:
            continue

        # Optional: create a column for runs per plate appearance if needed
        if "runs_scored" not in game_df.columns:
            if "events" in game_df.columns:
                game_df["runs_scored"] = game_df["events"].apply(lambda x: 1 if x in ["home_run", "score", "sac_fly"] else 0)
            else:
                continue  # skip if we can’t infer runs

        # Group by inning and calculate runs
        grouped = game_df.groupby("inning")
        for inning, df_inning in grouped:
            total_runs = df_inning["runs_scored"].sum()
            xwoba_vals = df_inning["estimated_woba_using_speedangle"].dropna()

            if total_runs == 2 and not xwoba_vals.empty:
                inning_avg = xwoba_vals.mean()
                inning_xwobas.append(inning_avg)

    if not inning_xwobas:
        print("⚠️ No qualifying innings found with exactly 2 runs scored.")
        return None

    return sum(inning_xwobas) / len(inning_xwobas)


def build_fatigue_metrics_drop_in(game_dict: dict, fatigue_xwoba_threshold: float) -> pd.DataFrame:
    """
    For each game, compute fatigue indicators:
    - Avg velocity drop (FF only, if available)
    - Spin rate drop by pitch type
    - Strike-to-ball ratio
    - delta xwOBA (late - early), batted balls only

    Returns:
    - A DataFrame with one row per game and fatigue metrics as columns.
    """
    records = []

    for date, game_df in game_dict.items():
        if "inning" not in game_df.columns or "estimated_woba_using_speedangle" not in game_df.columns:
            continue

        fatigued = False
        late_innings = game_df[game_df["inning"] > 3]

        if not late_innings.empty:
            for inning in late_innings["inning"].unique():
                batted = late_innings[
                    (late_innings["inning"] == inning) &
                    (late_innings["estimated_woba_using_speedangle"].notna())
                ]
                if not batted.empty:
                    inning_xwoba = batted["estimated_woba_using_speedangle"].mean()
                    if inning_xwoba > .320:
                        fatigued = True
                        break

        ff_df = game_df[game_df["pitch_type"] == "FF"]
        early_vel = ff_df[ff_df["inning"].isin([1, 2, 3])]["release_speed"].mean()
        late_vel = ff_df[ff_df["inning"] >= 5]["release_speed"].mean()
        velocity_drop = early_vel - late_vel if pd.notna(early_vel) and pd.notna(late_vel) else None

        spin_drop_dict = spin_rate_drop_by_pitch(game_df)
        spin_drop_ff = spin_drop_dict.get("FF", None)

        ratio = strike_to_ball_ratio(game_df)

        records.append({
            "game_date": date,
            "velocity_drop_ff": velocity_drop,
            "spin_rate_drop_ff": spin_drop_ff,
            "strike_to_ball_ratio": ratio,
            "fatigued": int(fatigued)
        })
    return pd.DataFrame(records)



def build_early_game_feature_df(game_dict: dict) -> pd.DataFrame:
    """
    Builds a DataFrame of early-game features (innings 1,2, and 3) for each game.
    Features include:
    - avg FF velocity
    - avg FF spin rate
    - strike-to-ball ratio
    - pitch count

    Parameters:
    - game_dict: dict of DataFrames, one per game keyed by game_date

    Returns:
    - DataFrame with one row per game and early-game features as columns
    """
    records = []

    for game_date, game_df in game_dict.items():
        early_df = game_df[game_df["inning"].isin([1, 2, 3])]
        ff_df = early_df[early_df["pitch_type"] == "FF"]

        # Compute early FF velocity and spin
        avg_ff_vel = ff_df["release_speed"].mean() if not ff_df.empty else None
        avg_ff_spin = ff_df["release_spin_rate"].mean() if not ff_df.empty else None

        # Strike to ball ratio
        strike_keywords = ["strike", "foul"]
        is_strike = early_df["description"].str.contains("|".join(strike_keywords), case=False, na=False)
        is_ball = early_df["description"].str.lower() == "ball"
        num_strikes = is_strike.sum()
        num_balls = is_ball.sum()
        if num_balls == 0:
            if num_strikes == 0:
                strike_ratio = 0.0
            else:
                strike_ratio = float("inf")
        else:
            strike_ratio = num_strikes / num_balls

        # Pitch count
        pitch_count = len(early_df)

        records.append({
            "game_date": game_date,
            "early_ff_velocity": avg_ff_vel,
            "early_ff_spin": avg_ff_spin,
            "early_strike_ratio": strike_ratio,
            "early_pitch_count": pitch_count
        })

    return pd.DataFrame(records)

def build_target_dataframe(game_dict: dict, innings=[1, 2, 3]) -> pd.DataFrame:
    """
    Builds a DataFrame of target features for each game.
    Features include:
    - avg FF velocity
    - avg FF spin rate
    - strike-to-ball ratio
    - pitch count

    Parameters:
    - game_dict: dict of DataFrames, one per game keyed by game_date
    - innings: list of innings to consider for early-game features

    Returns:
    - DataFrame with one row per game and target features as columns
    """
    records = []

    for game_date, game_df in game_dict.items():
        # ADD THIS TO DEBUG
        if not isinstance(game_df, pd.DataFrame):
            print(f"❌ Skipping {game_date}: Not a DataFrame, found {type(game_df)}")
            continue
        if "inning" not in game_df.columns:
            print(f"❌ Skipping {game_date}: Missing 'inning' column")
            continue

    for game_date, game_df in game_dict.items():
        early_df = game_df[game_df["inning"].isin(innings)]
        ff_df = early_df[early_df["pitch_type"] == "FF"]

        # Compute early FF velocity and spin
        avg_ff_vel = ff_df["release_speed"].mean() if not ff_df.empty else None
        avg_ff_spin = ff_df["release_spin_rate"].mean() if not ff_df.empty else None

        # Strike to ball ratio
        strike_keywords = ["strike", "foul"]
        is_strike = early_df["description"].str.contains("|".join(strike_keywords), case=False, na=False)
        is_ball = early_df["description"].str.lower() == "ball"
        num_strikes = is_strike.sum()
        num_balls = is_ball.sum()
        if num_balls == 0:
            if num_strikes == 0:
                strike_ratio = 0.0
            else:
                strike_ratio = float("inf")
        else:
            strike_ratio = num_strikes / num_balls

        # Pitch count
        pitch_count = len(early_df)

        records.append({
            "game_date": game_date,
            "early_ff_velocity": avg_ff_vel,
            "early_ff_spin": avg_ff_spin,
            "early_strike_ratio": strike_ratio,
            "early_pitch_count": pitch_count
        })

    return pd.DataFrame(records)


def build_training_dataframe(game_dict: dict) -> pd.DataFrame:
    """
    Builds a combined training DataFrame using:
    - Early game features (innings 1–3)
    - Fatigue targets (velocity/spin drop from early to late innings)

    Returns:
    - Merged DataFrame with one row per game and all features + targets
    """
    # Compute early-game features
    features_df = build_target_dataframe(game_dict, innings=[1, 2, 3])

    # Compute data-driven fatigue threshold once
    fatigue_threshold = average_xwoba_for_two_run_innings(game_dict)

    if fatigue_threshold is None:
        print("❌ Could not compute fatigue threshold. No training data generated.")
        return pd.DataFrame()

    # Compute fatigue targets with threshold
    fatigue_df = build_fatigue_metrics_drop_in(game_dict, fatigue_threshold)

    # Merge on game_date
    merged_df = pd.merge(features_df, fatigue_df, on="game_date")

    # Drop rows with missing values
    merged_df = merged_df.dropna()

    # Optional: print stats
    print("📊 Fatigue Label Distribution:")
    print(merged_df["fatigued"].value_counts())

    return merged_df
