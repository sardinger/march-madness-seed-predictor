import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os

# Config 
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed-dataset.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "visualizations")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")

# Load data 
df = pd.read_csv(DATA_PATH)

# All numeric features (drop team label)
FEATURE_COLS = [c for c in df.columns if c != "team"]

# Features we care most about for seed prediction (exclude ap_rank — too sparse)
KEY_FEATURES = [
    "ranker", "srs", "off_rtg", "def_rtg", "sos",
    "wins", "losses", "opp_pts_per_g",
    "fg3_pct", "fg2_pct", "ft_pct",
    "orb", "drb", "ast", "stl", "blk", "tov",
    "avg_pts", "avg_opp_pts", "avg_srs", "wins_last5",
]


# 1. Feature Distributions 
"""
Each subplot shows how the values for that feature are
distributed across all 68 tournament teams, with a KDE curve overlaid
"""
def plot_distributions():
    """Histogram + KDE for every numeric feature."""
    n_cols = 4
    n_rows = -(-len(KEY_FEATURES) // n_cols)  # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(KEY_FEATURES):
        ax = axes[i]
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue", bins=15)
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Count", fontsize=9)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions — 2026 March Madness Teams", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "1_feature_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# 2. Correlation Heatmap 
"""
Heatmap of Pearson correlations between all numeric features.
"""
def plot_correlation_heatmap():
    """Heatmap of Pearson correlations between all numeric features."""
    corr = df[KEY_FEATURES].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
    # Only mask upper triangle (keep diagonal)
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            mask.iloc[i, j] = True

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.4,
        annot_kws={"size": 7},
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "2_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# 3. Feature Importance Proxy (correlation with ranker) 
"""
Bar chart of |correlation| with 'ranker' (KenPom rank).
Lower ranker = better team, so this acts as a seed-quality proxy
before a model is trained.
"""
def plot_feature_importance_proxy():
    target = "ranker"
    features = [c for c in KEY_FEATURES if c != target]

    corr_with_target = (
        df[features + [target]]
        .corr()[target]
        .drop(target)
        .sort_values(key=abs, ascending=False)
    )

    colors = ["#d62728" if v < 0 else "#1f77b4" for v in corr_with_target]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(corr_with_target.index[::-1], corr_with_target.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation with KenPom Rank", fontsize=11)
    ax.set_title(
        "Feature Importance Proxy\n(Correlation with KenPom Rank — lower rank = better team)",
        fontsize=12, fontweight="bold"
    )
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # Annotate values
    for bar, val in zip(bars[::-1], corr_with_target.values):
        ax.text(
            val + (0.01 if val >= 0 else -0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8,
        )

    # Legend
    import matplotlib.patches as mpatches
    ax.legend(
        handles=[
            mpatches.Patch(color="#1f77b4", label="Higher value → higher rank number (worse)"),
            mpatches.Patch(color="#d62728", label="Higher value → lower rank number (better)"),
        ],
        fontsize=8,
        loc="lower right",
    )

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "3_feature_importance_proxy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# 4. Top features vs KenPom rank (scatter) 
"""
Scatter plots of the 6 features most correlated with ranker.
"""
def plot_top_feature_scatters():
    """Scatter plots of the 6 features most correlated with ranker."""
    target = "ranker"
    features = [c for c in KEY_FEATURES if c != target]
    top6 = (
        df[features + [target]]
        .corr()[target]
        .drop(target)
        .abs()
        .sort_values(ascending=False)
        .head(6)
        .index.tolist()
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, col in zip(axes, top6):
        ax.scatter(df[col], df[target], alpha=0.7, edgecolors="white", linewidth=0.4, color="steelblue", s=60)
        # Trend line
        valid = df[[col, target]].dropna()
        m, b = pd.Series(valid[col]).pipe(lambda x: (
            (valid[target].cov(x) / x.var(), valid[target].mean() - (valid[target].cov(x) / x.var()) * x.mean())
        ))
        xs = [valid[col].min(), valid[col].max()]
        ax.plot(xs, [m * x + b for x in xs], color="crimson", linewidth=1.5, linestyle="--")

        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("KenPom Rank", fontsize=10)
        ax.set_title(f"{col} vs KenPom Rank", fontsize=11, fontweight="bold")

        # Annotate a few notable teams
        for _, row in df.nsmallest(3, target).iterrows():
            if pd.notna(row[col]):
                ax.annotate(row["team"], (row[col], row[target]), fontsize=7, alpha=0.8,
                            xytext=(4, 0), textcoords="offset points")

    fig.suptitle("Top Features vs KenPom Rank", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "4_top_features_vs_rank.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# 5. Win-loss record overview 
"""
Horizontal stacked bar of wins/losses per team, sorted by wins.
"""
def plot_wins_losses():
    sorted_df = df.sort_values("wins", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 16))
    y = range(len(sorted_df))
    ax.barh(list(y), sorted_df["wins"], color="#2ca02c", label="Wins")
    ax.barh(list(y), sorted_df["losses"], left=sorted_df["wins"], color="#d62728", label="Losses")
    ax.set_yticks(list(y))
    ax.set_yticklabels(sorted_df["team"], fontsize=8)
    ax.set_xlabel("Games", fontsize=11)
    ax.set_title("2026 March Madness Teams — Win/Loss Records", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "5_win_loss_records.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# Main
if __name__ == "__main__":
    print(f"Loaded {len(df)} teams, {len(FEATURE_COLS)} features.\n")
    plot_distributions()
    plot_correlation_heatmap()
    plot_feature_importance_proxy()
    plot_top_feature_scatters()
    plot_wins_losses()
    print(f"\nAll visualizations saved to: {os.path.abspath(OUT_DIR)}")
