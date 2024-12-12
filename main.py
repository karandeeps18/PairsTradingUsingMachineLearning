import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from utils import load_data, calculate_pair_statistics
from NoClusterPairSelection import select_pairs_no_clustering
from ThemeClusterPairSelection import select_pairs_theme_clustering
from OpticsPairSelection import select_pairs_optics_clustering

# Function to implement embargo on training observations
def getEmbargoTimes(times, pctEmbargo):
    # Get embargo time for each bar
    step = int(times.shape[0] * pctEmbargo)
    if step == 0:
        mbrg = pd.Series(times, index=times)
    else:
        mbrg = pd.Series(times[step:], index=times[:-step])
        mbrg = pd.concat([mbrg, pd.Series(times[-1], index=times[-step:])])
    return mbrg

# Function to plot Hurst Exponent, Cointegration, Correlation, PCA, and OPTICS
def plot_results(results_df, pca_data, optics_model, pca_components):
    # Create directory for saving plots
    os.makedirs("SelectedPairsRelationalPlots", exist_ok=True)
    os.makedirs("Optics_Plots", exist_ok=True)

    # Plot Hurst Exponent
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['Hurst_Exponent'].dropna(), bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Hurst Exponent')
    plt.ylabel('Frequency')
    plt.title('Distribution of Hurst Exponent')
    plt.grid(True)
    plt.savefig("SelectedPairsRelationalPlots/hurst_exponent_distribution.png")
    plt.close()

    # Plot Cointegration P-Values
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['Cointegration_PValue'].dropna(), bins=20, color='salmon', edgecolor='black')
    plt.xlabel('Cointegration P-Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Cointegration P-Values')
    plt.grid(True)
    plt.savefig("SelectedPairsRelationalPlots/cointegration_pvalue_distribution.png")
    plt.close()

    # Plot Correlation
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['Correlation'].dropna(), bins=20, color='lightgreen', edgecolor='black')
    plt.xlabel('Correlation')
    plt.ylabel('Frequency')
    plt.title('Distribution of Correlation Coefficients')
    plt.grid(True)
    plt.savefig("SelectedPairsRelationalPlots/correlation_distribution.png")
    plt.close()

    # PCA Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.6, color='dodgerblue')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Component Plot')
    plt.grid(True)
    plt.savefig("SelectedPairsRelationalPlots/pca_component_plot.png")
    plt.close()

    # OPTICS Clustering Plot
    plt.figure(figsize=(12, 6))
    reachability = optics_model.reachability_[optics_model.ordering_]
    space = range(len(reachability))
    plt.plot(space, reachability, 'b-', label="Reachability Distance")
    plt.xlabel('Points Ordered by OPTICS')
    plt.ylabel('Reachability Distance')
    plt.axhline(y=0.05, color='r', linestyle='--', label="Cluster Threshold")
    plt.title('OPTICS Reachability Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig("SelectedPairsRelationalPlots/optics_reachability_plot.png")
    plt.close()

    # t-SNE Plot
    tsne = TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(pca_data)
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], alpha=0.6, c=optics_model.labels_, cmap='tab10')
    plt.xlabel('t-SNE Dim 1')
    plt.ylabel('t-SNE Dim 2')
    plt.title('t-SNE Visualization of OPTICS Clusters')
    plt.grid(True)
    plt.savefig("Optics_Plots/tsne_optics_clusters.png")
    plt.close()


def main():
    preprocessed_data_file = 'preprocessed_etfs.csv'
    df = load_data(preprocessed_data_file)

    # Data Partitioning Parameters
    formation_period_days = 730      # 2 years
    validation_period_days = 90      # 3 months
    trading_period_days = 182        # 6 months
    step_size_days = 182             # 6 months
    pct_embargo = 0.01               # 1% embargo
    
    formation_period = timedelta(days=formation_period_days)
    validation_period = timedelta(days=validation_period_days)
    trading_period = timedelta(days=trading_period_days)
    step_size = timedelta(days=step_size_days)

    data_start_date = df.index.min()
    data_end_date = df.index.max()

    formation_start_date = data_start_date
    formation_end_date = formation_start_date + formation_period
    trading_start_date = formation_end_date
    trading_end_date = trading_start_date + trading_period

    # Initialize DataFrame
    all_selected_pairs = pd.DataFrame()

    while trading_end_date <= data_end_date:
        # Extract data for the periods
        formation_data = df[formation_start_date:formation_end_date]
        validation_start_date = formation_end_date - validation_period
        validation_data = df[validation_start_date:formation_end_date]
        trading_data = df[trading_start_date:trading_end_date]

        # Calculate trading_period_days (update in case of leap years)
        actual_trading_period_days = (trading_end_date - trading_start_date).days
        print(f"Processing period: {formation_start_date.date()} to {trading_end_date.date()}")

        # Embargo times
        embargo_times = getEmbargoTimes(df.index, pct_embargo)

        # No Clustering
        selected_pairs_no_cluster = select_pairs_no_clustering(
            formation_data, formation_start_date, formation_end_date,
            trading_start_date, trading_end_date, actual_trading_period_days)
        all_selected_pairs = pd.concat([all_selected_pairs, selected_pairs_no_cluster], ignore_index=True)

        # Theme Clustering
        selected_pairs_theme_cluster = select_pairs_theme_clustering(
            formation_data, formation_start_date, formation_end_date,
            trading_start_date, trading_end_date, actual_trading_period_days
        )
        all_selected_pairs = pd.concat([all_selected_pairs, selected_pairs_theme_cluster], ignore_index=True)

        # OPTICS Clustering
        selected_pairs_optics_cluster = select_pairs_optics_clustering(
            formation_data, formation_start_date, formation_end_date,
            trading_start_date, trading_end_date, actual_trading_period_days
        )
        if isinstance(selected_pairs_optics_cluster, tuple):
            selected_pairs_optics_cluster, pca_data, optics_model = selected_pairs_optics_cluster
            # Plotting results for the current iteration
            plot_results(all_selected_pairs, pca_data, optics_model, pca_components=3)
        all_selected_pairs = pd.concat([all_selected_pairs, selected_pairs_optics_cluster], ignore_index=True)

        # Advance the windows
        formation_start_date += step_size
        formation_end_date += step_size
        trading_start_date += step_size
        trading_end_date += step_size

    # Save 
    all_selected_pairs.to_csv('all_selected_pairs.csv', index=False)

if __name__ == "__main__":
    main()
