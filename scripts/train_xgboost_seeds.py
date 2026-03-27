import pandas as pd
import numpy as np
import re
import unicodedata
import json
import joblib

from pathlib import Path
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt


# Load data
features_path = Path("/Users/mayankgudi/Desktop/MSU/Senior/CSE_482/marchmadness/march-madness-seed-predictor/data/processed-dataset.csv")
seeds_path = Path("/Users/mayankgudi/Desktop/MSU/Senior/CSE_482/marchmadness/march-madness-seed-predictor/data/2026_ncaa_mens_tournament_seeds.csv")

features_df = pd.read_csv(features_path)
seeds_df = pd.read_csv(seeds_path)



# normalize the team names

def normalize_team_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    name = name.lower().strip()
    name = name.replace("&", "and")
    name = re.sub(r"[^a-z0-9]+", "", name)

    alias_map = {
        "texasaandm": "texasam",
        "stjohns": "stjohnsny",
        "saintmarys": "saintmarysca",
        "mcneese": "mcneesestate",
        "ncstate": "northcarolinastate",
        "penn": "pennsylvania",
        "umbc": "marylandbaltimorecounty",
        "liu": "longislanduniversity",
        "prairieviewaandm": "prairieview",
        "queens": "queensnc",

        "byu": "brighamyoung",
        "ucf": "centralflorida",
        "uconn": "connecticut",
        "smu": "southernmethodist",
        "tcu": "texaschristian",
        "vcu": "virginiacommonwealth",
    }

    return alias_map.get(name, name)


features_df["team_key"] = features_df["team"].apply(normalize_team_name)
seeds_df["team_key"] = seeds_df["team"].apply(normalize_team_name)


# merge all data
df = features_df.merge(
    seeds_df[["team_key", "region", "regional_seed", "overall_seed", "first_four"]],
    on="team_key",
    how="inner"
)

print("Merged shape:", df.shape)

missing_from_merge = sorted(set(features_df["team_key"]) - set(df["team_key"]))
if missing_from_merge:
    print("Teams not matched after merge:", missing_from_merge)

# Select a target
# Ran the training regime for regional seeding as well as overall seeding
target_col = "overall_seed"

drop_cols = [
    "team",
    "team_key",
    "region",
    "regional_seed",
    "overall_seed",
    "first_four"
]

X = df.drop(columns=drop_cols)
y = df[target_col].astype(float)

# Fill missing numeric values
X = X.fillna(X.median(numeric_only=True))

print("Feature columns:")
print(X.columns.tolist())
print("Target:", target_col)


# Train and test splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Baseline model
baseline_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

baseline_model.fit(X_train, y_train)

baseline_preds = baseline_model.predict(X_test)
baseline_preds_round = np.clip(np.rint(baseline_preds), 1, 16)

baseline_mae = mean_absolute_error(y_test, baseline_preds)
baseline_exact_acc = np.mean(baseline_preds_round == y_test)
baseline_within1 = np.mean(np.abs(baseline_preds_round - y_test) <= 1)

print("\n" + "=" * 50)
print("BASELINE RESULTS")
print("=" * 50)
print(f"Baseline MAE: {baseline_mae:.4f}")
print(f"Baseline Exact Accuracy: {baseline_exact_acc:.4f}")
print(f"Baseline Within-1 Accuracy: {baseline_within1:.4f}")



# Hyperparameter tuning
# Used a small grid because its only 68 rows
param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_alpha": [0.0, 0.1],
    "reg_lambda": [0.5, 1.0, 1.5],
}

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_cv_mae = -grid_search.best_score_

best_preds = best_model.predict(X_test)
best_preds_round = np.clip(np.rint(best_preds), 1, 16)

best_mae = mean_absolute_error(y_test, best_preds)
best_exact_acc = np.mean(best_preds_round == y_test)
best_within1 = np.mean(np.abs(best_preds_round - y_test) <= 1)

print("\n" + "=" * 50)
print("BEST MODEL RESULTS")
print("=" * 50)
print("Best params:", best_params)
print(f"Best CV MAE: {best_cv_mae:.4f}")
print(f"Best Test MAE: {best_mae:.4f}")
print(f"Best Exact Accuracy: {best_exact_acc:.4f}")
print(f"Best Within-1 Accuracy: {best_within1:.4f}")


# Utilized AI to plot and help me save everything
plt.figure(figsize=(10, 8))
plot_importance(best_model, max_num_features=15)
plt.title(f"Top XGBoost Feature Importances for {target_col}")
plt.tight_layout()
plt.show()


model_dir = Path("xgb_seed_model")
model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, model_dir / f"best_model_{target_col}.pkl")

with open(model_dir / f"feature_names_{target_col}.json", "w") as f:
    json.dump(X.columns.tolist(), f)

metadata = {
    "target": target_col,
    "best_params": best_params,
    "best_cv_mae": float(best_cv_mae),
    "test_mae": float(best_mae),
    "test_exact_accuracy": float(best_exact_acc),
    "test_within_one_accuracy": float(best_within1),
    "num_rows": int(df.shape[0]),
    "num_features": int(X.shape[1]),
}

with open(model_dir / f"metadata_{target_col}.json", "w") as f:
    json.dump(metadata, f, indent=2)