import os
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the plots folder exists
PLOTS_FOLDER = "plots"
os.makedirs(PLOTS_FOLDER, exist_ok=True)

def plot_time_series_by_cluster(selected_pairs, preprocessed_etfs):
    """Plot time series of all selected pairs grouped by OPTICS cluster."""
    clusters = selected_pairs['Cluster'].unique()

    for cluster in clusters:
        cluster_pairs = selected_pairs[selected_pairs['Cluster'] == cluster]

        plt.figure(figsize=(12, 8))
        for _, pair_row in cluster_pairs.iterrows():
            pair_str = pair_row['Pair'].strip("()").replace("'", "").split(", ")
            etf1, etf2 = pair_str[0].strip(), pair_str[1].strip()

            ts1 = preprocessed_etfs[f"{etf1}_adj_close"] / preprocessed_etfs[f"{etf1}_adj_close"].iloc[0]
            ts2 = preprocessed_etfs[f"{etf2}_adj_close"] / preprocessed_etfs[f"{etf2}_adj_close"].iloc[0]

            plt.plot(ts1, label=f"{etf1} (Cluster {cluster})", alpha=0.7)
            plt.plot(ts2, label=f"{etf2} (Cluster {cluster})", alpha=0.7)

        plt.title(f"Normalized Time Series for Cluster {cluster}")
        plt.xlabel("Date")
        plt.ylabel("Normalized Price")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(PLOTS_FOLDER, f"time_series_cluster_{cluster}.png"))
        plt.close()

if __name__ == "__main__":
    # File paths
    selected_pairs_path = "all_selected_pairs.csv"
    time_series_path = "preprocessed_etfs.csv"

    # Load data
    selected_pairs = pd.read_csv(selected_pairs_path)
    preprocessed_etfs = pd.read_csv(time_series_path, index_col=0, parse_dates=True)

    # Generate plots
    plot_time_series_by_cluster(selected_pairs, preprocessed_etfs)
