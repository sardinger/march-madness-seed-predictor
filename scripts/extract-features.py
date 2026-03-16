from dotenv import load_dotenv
from pymongo import MongoClient
from pathlib import Path
import os
import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import convert_team_name


def log_unmatched(df_left, df_right, left_col, right_col, label):
    """Log if a team name is in the left df but not the right and vv."""
    unmatched_left = set(df_left[left_col]) - set(df_right[right_col])
    unmatched_right = set(df_right[right_col]) - set(df_left[left_col])
    if unmatched_left:
        print(f"[UNMATCHED] {label} - in left but not right: {sorted(unmatched_left)}")
    if unmatched_right:
        print(f"[UNMATCHED] {label} - in right but not left: {sorted(unmatched_right)}")


def main():
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    client = MongoClient(uri)

    db = client["march-madness"]
    container = db["team-stats"]

    cursor = container.find(
        {},
        {
            "team": 1,
            "fg2a": 1,
            "fg2_pct": 1,
            "fg3a": 1,
            "fg3_pct": 1,
            "fta": 1,
            "ft_pct": 1,
            "orb": 1,
            "drb": 1,
            "ast": 1,
            "stl": 1,
            "blk": 1,
            "tov": 1,
            "pts": 1,
            "_id": 0,
        },
    )
    df_stats = pd.DataFrame(list(cursor))
    print(df_stats.head())

    container = db["season-ratings-2026"]
    cursor = container.find(
        {},
        {
            "school": 1,
            "ranker": 1,
            "ap_rank": 1,
            "wins": 1,
            "losses": 1,
            "opp_pts_per_g": 1,
            "sos": 1,
            "srs": 1,
            "off_rtg": 1,
            "def_rtg": 1,
            "_id": 0,
        },
    )
    df_ratings = pd.DataFrame(list(cursor))

    # Convert school names to lowercase and ap rank to ints
    df_ratings["school"] = df_ratings["school"].str.lower().apply(convert_team_name)
    df_ratings["ap_rank"] = pd.to_numeric(
        df_ratings["ap_rank"], errors="coerce"
    ).astype("Int64")

    log_unmatched(
        df_stats, df_ratings, "team", "school", "team-stats vs season-ratings"
    )
    df = pd.merge(
        df_stats, df_ratings, left_on="team", right_on="school"
    )  # TODO: Only teams in both df will merge into 1 so make sure they all match
    df.drop(columns=["school"], inplace=True)

    print(df.head())

    container = db["rolling-stats"]
    cursor = container.find(
        {},
        {
            "team": 1,
            "game_location": 1,
            "srs": 1,
            "game_result": 1,
            "pts": 1,
            "opp_pts": 1,
            "_id": 0,
        },
    )
    df_rolling = pd.DataFrame(list(cursor))

    # Convert game location to 0 for away game and 1 for home
    # TODO: what about neutral?
    df_rolling["game_location"] = df_rolling["game_location"].apply(
        lambda x: 0 if x == "@" else 1
    )

    df_rolling = (
        df_rolling.groupby("team")
        .head(5)
        .groupby("team")
        .agg(
            avg_pts=("pts", "mean"),
            avg_opp_pts=("opp_pts", "mean"),
            avg_srs=("srs", "mean"),
            num_home_games=("game_location", lambda x: (x == 1).sum()),
            num_away_games=("game_location", lambda x: (x == 0).sum()),
            wins_last5=("game_result", "sum"),
        )
        .reset_index()
    )

    log_unmatched(df, df_rolling, "team", "team", "merged-df vs rolling-stats")
    df = pd.merge(df, df_rolling, left_on="team", right_on="team")
    print(df.head())

    output_path = (
        Path(__file__).resolve().parent.parent / "data" / "processed-dataset.csv"
    )
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
