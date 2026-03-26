"""
Predicts tournament seeds (1–16) from team stats.
Seed distribution: 4 teams per seed 1–10, 6 teams for seed 11, 6 teams for seed 16.

Usage:
    python seed_predictor.py --data your_data.csv
    python seed_predictor.py --data your_data.csv --evaluate   # if 'seed' column exists
"""

import warnings
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ── Seed distribution ──────────────────────────────────────────────────────────
# Seeds 1–10: 4 teams each = 40 teams
# Seed 11:    6 teams
# Seeds 12–15: 4 teams each = 16 teams
# Seed 16:    6 teams
# Total: 40 + 6 + 16 + 6 = 68 teams
SEED_COUNTS = {
    1: 4, 2: 4, 3: 4, 4: 4, 5: 4,
    6: 4, 7: 4, 8: 4, 9: 4, 10: 4,
    11: 6,
    12: 4, 13: 4, 14: 4, 15: 4,
    16: 6,
}
TOTAL_TEAMS = sum(SEED_COUNTS.values())  # 68


def build_score(df: pd.DataFrame) -> pd.Series:
    """
    Composite score combining the most predictive features.
    Higher score → better team → lower (better) seed number.
    """
    score = pd.Series(0.0, index=df.index)

    # Offensive quality
    if "off_rtg" in df: score += df["off_rtg"].fillna(df["off_rtg"].median()) * 2.5
    if "avg_pts" in df: score += df["avg_pts"].fillna(df["avg_pts"].median()) * 0.8
    if "pts"    in df: score += df["pts"].fillna(df["pts"].median()) * 0.01

    # Defensive quality (lower opp pts = better)
    if "def_rtg"      in df: score -= df["def_rtg"].fillna(df["def_rtg"].median()) * 2.5
    if "avg_opp_pts"  in df: score -= df["avg_opp_pts"].fillna(df["avg_opp_pts"].median()) * 0.8
    if "opp_pts_per_g"in df: score -= df["opp_pts_per_g"].fillna(df["opp_pts_per_g"].median()) * 0.8

    # Strength of schedule & overall rating
    if "srs"      in df: score += df["srs"].fillna(df["srs"].median()) * 3.0
    if "avg_srs"  in df: score += df["avg_srs"].fillna(df["avg_srs"].median()) * 1.5
    if "sos"      in df: score += df["sos"].fillna(df["sos"].median()) * 1.0

    # Win/loss record
    if "wins"   in df: score += df["wins"].fillna(0) * 1.2
    if "losses" in df: score -= df["losses"].fillna(0) * 0.8
    if "ranker" in df:
        score -= df["ranker"].fillna(df["ranker"].max()) * 0.5
    if "ap_rank" in df:
        valid_ap = df["ap_rank"].fillna(df["ap_rank"].max() + 1)
        score -= valid_ap * 0.3

    # Recent form
    if "wins_last5" in df: score += df["wins_last5"].fillna(df["wins_last5"].median()) * 2.0

    # Shooting efficiency
    if "fg3_pct" in df: score += df["fg3_pct"].fillna(df["fg3_pct"].median()) * 15
    if "fg2_pct" in df: score += df["fg2_pct"].fillna(df["fg2_pct"].median()) * 10
    if "ft_pct"  in df: score += df["ft_pct"].fillna(df["ft_pct"].median()) * 5

    # Other stats
    if "ast" in df: score += df["ast"].fillna(df["ast"].median()) * 0.3
    if "stl" in df: score += df["stl"].fillna(df["stl"].median()) * 0.4
    if "blk" in df: score += df["blk"].fillna(df["blk"].median()) * 0.3
    if "tov" in df: score -= df["tov"].fillna(df["tov"].median()) * 0.5
    if "orb" in df: score += df["orb"].fillna(df["orb"].median()) * 0.2
    if "drb" in df: score += df["drb"].fillna(df["drb"].median()) * 0.15

    return score


def assign_seeds(score_series: pd.Series) -> pd.Series:
    """
    Rank teams by composite score (descending) and assign seeds
    according to SEED_COUNTS distribution.
    """
    n = len(score_series)
    if n > TOTAL_TEAMS:
        raise ValueError(
            f"Dataset has {n} teams but seed structure supports max {TOTAL_TEAMS}. "
            "Filter to tournament-eligible teams first."
        )

    # Adjust distribution if fewer teams
    seed_counts = dict(SEED_COUNTS)
    assigned = 0
    seeds_used = []
    for s in sorted(seed_counts):
        if assigned >= n:
            break
        take = min(seed_counts[s], n - assigned)
        seeds_used.extend([s] * take)
        assigned += take

    # Sort teams best → worst
    ranked_idx = score_series.sort_values(ascending=False).index
    seed_map = pd.Series(seeds_used, index=ranked_idx)
    return seed_map.reindex(score_series.index)


def ml_score(df: pd.DataFrame, features: list[str]) -> pd.Series:
    """
    Use a GradientBoosting model to learn team quality from features.
    Since we have no seed labels, we train it to reproduce the composite score
    (self-supervised), then use it to produce a smooth, regularised ranking.
    """
    X = df[features].copy()
    y = build_score(df)

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("model",  GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        )),
    ])
    pipe.fit(X, y)
    return pd.Series(pipe.predict(X), index=df.index)


def predict_seeds(df: pd.DataFrame, use_ml: bool = True) -> pd.DataFrame:
    FEATURE_COLS = [c for c in df.columns if c not in ("team",)]

    # Composite score (always computed)
    composite = build_score(df)

    if use_ml and len(FEATURE_COLS) >= 3:
        ml = ml_score(df, FEATURE_COLS)
        # Blend: 60 % ML, 40 % hand-crafted composite
        final_score = 0.6 * ml + 0.4 * (
            (composite - composite.mean()) / (composite.std() + 1e-9)
            * (ml.std() + 1e-9)
        )
    else:
        final_score = composite

    seed_col = assign_seeds(final_score)

    result = df[["team"]].copy() if "team" in df.columns else df.iloc[:, :1].copy()
    result["composite_score"] = final_score.values
    result["predicted_seed"]  = seed_col.values
    result = result.sort_values("predicted_seed").reset_index(drop=True)
    result["rank_overall"] = result.index + 1
    return result


# ── Evaluation (if true seed column present) ──────────────────────────────────

def evaluate(result: pd.DataFrame, true_seeds: pd.Series) -> None:
    result = result.copy()
    result["true_seed"] = true_seeds.values
    result["seed_error"] = (result["predicted_seed"] - result["true_seed"]).abs()

    mae   = result["seed_error"].mean()
    exact = (result["seed_error"] == 0).mean() * 100
    within1 = (result["seed_error"] <= 1).mean() * 100

    print("\n── Evaluation ─────────────────────────────────────")
    print(f"  Mean Absolute Seed Error : {mae:.2f}")
    print(f"  Exact seed accuracy      : {exact:.1f}%")
    print(f"  Within 1 seed accuracy   : {within1:.1f}%")

    print("\nSeed-level accuracy:")
    for seed in sorted(result["true_seed"].unique()):
        sub = result[result["true_seed"] == seed]
        acc = (sub["seed_error"] == 0).mean() * 100
        print(f"  Seed {seed:>2}: {acc:5.1f}% exact  (n={len(sub)})")


# ── Config ─────────────────────────────────────────────────────────────────────

DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "processed-dataset.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "predicted_seeds.csv")
USE_ML      = True   # Set False to use composite score only
EVALUATE    = True   # Set True to evaluate if a 'seed' column exists in the data


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} teams with {len(df.columns)} columns.")

    true_seeds = None
    if "seed" in df.columns:
        true_seeds = df["seed"].copy()
        df = df.drop(columns=["seed"])

    result = predict_seeds(df, use_ml=USE_ML)

    print("\n── Predicted Seeds ────────────────────────────────")
    print(result[["rank_overall", "team", "predicted_seed", "composite_score"]]
          .to_string(index=False))

    if EVALUATE and true_seeds is not None:
        evaluate(result, true_seeds)
    elif EVALUATE:
        print("\n[Note] EVALUATE=True but no 'seed' column found in data.")

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults saved to: {OUTPUT_PATH}")