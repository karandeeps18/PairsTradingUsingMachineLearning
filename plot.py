import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from scipy.stats import linregress

PLOTS_FOLDER = "plots"
os.makedirs(PLOTS_FOLDER, exist_ok=True)

def plot_correlation_heatmap(preprocessed_etfs):
    """Plot the correlation heatmap for the entire dataset."""
    plt.figure(figsize=(10, 8))
    correlation_matrix = preprocessed_etfs.corr()
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, cbar=True)
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(PLOTS_FOLDER, "correlation_heatmap.png"))
    plt.close()

def plot_spread_distribution(selected_pairs, preprocessed_etfs):
    """Plot the distribution of spreads for all selected pairs."""
    for _, pair_row in selected_pairs.iterrows():
        pair_str = pair_row['Pair'].strip("()").replace("'", "").split(", ")
        etf1, etf2 = pair_str[0].strip(), pair_str[1].strip()
        spread = preprocessed_etfs[f"{etf1}_adj_close"] - preprocessed_etfs[f"{etf2}_adj_close"]

        plt.figure(figsize=(8, 6))
        sns.histplot(spread, kde=True, bins=30, label='Spread', color='blue')
        plt.title(f"Spread Distribution for {etf1} - {etf2}")
        plt.xlabel("Spread")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(os.path.join(PLOTS_FOLDER, f"spread_distribution_{etf1}_{etf2}.png"))
        plt.close()

def plot_hurst_boxplot(selected_pairs):
    """Plot a boxplot for the Hurst exponent of selected pairs."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=selected_pairs['Average_Hurst'], color='lightblue')
    plt.title("Hurst Exponent Distribution")
    plt.ylabel("Hurst Exponent")
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_FOLDER, "hurst_boxplot.png"))
    plt.close()

def plot_half_life_distribution(selected_pairs):
    """Plot the distribution of half-lives for selected pairs."""
    plt.figure(figsize=(8, 6))
    sns.histplot(selected_pairs['Half_Life'], bins=20, kde=False, color='green')
    plt.title("Distribution of Half-Life for Selected Pairs")
    plt.xlabel("Half-Life (Days)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_FOLDER, "half_life_distribution.png"))
    plt.close()

def plot_cluster_composition(selected_pairs):
    """Plot the composition of OPTICS clusters."""
    plt.figure(figsize=(8, 6))
    cluster_counts = selected_pairs['Cluster'].value_counts()
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis")
    plt.title("Cluster Composition of Selected Pairs")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Pairs")
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_FOLDER, "cluster_composition.png"))
    plt.close()

def plot_cointegration_pvalues(selected_pairs):
    """Plot the distribution of cointegration test p-values."""
    plt.figure(figsize=(8, 6))
    sns.histplot(selected_pairs['Cointegration_PValue'], bins=30, kde=False, color='red')
    plt.title("Cointegration Test P-Values")
    plt.xlabel("P-Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_FOLDER, "cointegration_pvalues.png"))
    plt.close()

def plot_normalized_prices(selected_pairs, preprocessed_etfs):
    """Plot normalized prices of each pair over time."""
    for _, pair_row in selected_pairs.iterrows():
        pair_str = pair_row['Pair'].strip("()").replace("'", "").split(", ")
        etf1, etf2 = pair_str[0].strip(), pair_str[1].strip()

        ts1 = preprocessed_etfs[f"{etf1}_adj_close"] / preprocessed_etfs[f"{etf1}_adj_close"].iloc[0]
        ts2 = preprocessed_etfs[f"{etf2}_adj_close"] / preprocessed_etfs[f"{etf2}_adj_close"].iloc[0]

        plt.figure(figsize=(12, 6))
        plt.plot(ts1, label=f"{etf1} Normalized", alpha=0.7)
        plt.plot(ts2, label=f"{etf2} Normalized", alpha=0.7)
        plt.title(f"Normalized Prices for {etf1} and {etf2}")
        plt.xlabel("Date")
        plt.ylabel("Normalized Price")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(PLOTS_FOLDER, f"normalized_prices_{etf1}_{etf2}.png"))
        plt.close()

def plot_adf_test_spread_returns(selected_pairs, preprocessed_etfs):
    """Plot ADF test results for the returns of the spread for each pair."""
    for _, pair_row in selected_pairs.iterrows():
        pair_str = pair_row['Pair'].strip("()").replace("'", "").split(", ")
        etf1, etf2 = pair_str[0].strip(), pair_str[1].strip()

        # Calculate spread
        spread = preprocessed_etfs[f"{etf1}_adj_close"] - preprocessed_etfs[f"{etf2}_adj_close"]
        
        # Calculate spread returns
        spread_returns = spread.pct_change().dropna()

        # ADF test
        adf_result = adfuller(spread_returns)
        
        # Plot spread returns
        plt.figure(figsize=(12, 6))
        plt.plot(spread_returns, label='Spread Returns')
        plt.title(f"ADF Test for Spread Returns ({etf1} - {etf2})")
        plt.xlabel("Date")
        plt.ylabel("Spread Returns")
        plt.grid(True)

        # Add ADF statistic and p-value to plot
        plt.text(len(spread_returns) // 2, spread_returns.mean(), 
                 f"ADF Statistic: {adf_result[0]:.2f}\nP-Value: {adf_result[1]:.2g}",
                 bbox=dict(facecolor='white', alpha=0.5), fontsize=10)
        plt.legend()
        plt.savefig(os.path.join(PLOTS_FOLDER, f"adf_test_spread_returns_{etf1}_{etf2}.png"))
        plt.close()

if __name__ == "__main__":
    # File paths
    selected_pairs_path = "all_selected_pairs.csv"
    time_series_path = "preprocessed_etfs.csv"
    selected_pairs = pd.read_csv(selected_pairs_path)
    preprocessed_etfs = pd.read_csv(time_series_path, index_col=0, parse_dates=True)

    # plots
    plot_correlation_heatmap(preprocessed_etfs)
    plot_spread_distribution(selected_pairs, preprocessed_etfs)
    plot_hurst_boxplot(selected_pairs)
    plot_half_life_distribution(selected_pairs)
    plot_cluster_composition(selected_pairs)
    plot_cointegration_pvalues(selected_pairs)
    plot_normalized_prices(selected_pairs, preprocessed_etfs)
    plot_adf_test_spread_returns(selected_pairs, preprocessed_etfs)
