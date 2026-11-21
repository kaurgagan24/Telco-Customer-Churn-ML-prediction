"""Optional EDA helpers carried over from the notebook (correlation + VIF).

Run on the *processed* dataframe. Needs seaborn / statsmodels / matplotlib.
"""
from __future__ import annotations

import pandas as pd


def churn_correlation(df: pd.DataFrame, target: str = "Churn") -> pd.DataFrame:
    """Correlation of every numeric feature with the target, sorted descending."""
    corr = df.corr(numeric_only=True)[[target]]
    return corr.drop(index=target).sort_values(by=target, ascending=False)


def plot_churn_correlation(df: pd.DataFrame, target: str = "Churn", save_path: str | None = None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    corr = churn_correlation(df, target)
    plt.figure(figsize=(4, max(4, 0.3 * len(corr))))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"Correlation with {target}", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return plt.gcf()


def vif_table(
    df: pd.DataFrame,
    columns: tuple[str, ...] = ("tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"),
) -> pd.DataFrame:
    """Variance Inflation Factor for the numeric features (NaNs median-filled)."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = df[list(columns)].copy()
    X = X.fillna(X.median())
    vif = pd.DataFrame(
        {
            "Feature": X.columns,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        }
    )
    return vif.sort_values("VIF", ascending=False).reset_index(drop=True)
