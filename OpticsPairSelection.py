import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
from utils import calculate_pair_statistics
from utils import calculate_hurst_exponent 

def select_pairs_optics_clustering(formation_data, formation_start_date, formation_end_date,
                                   trading_start_date, trading_end_date, trading_period_days):
    adj_close_cols = [col for col in formation_data.columns if 'adj_close' in col]
    etf_names = sorted(list(set([col.split('_')[0] for col in adj_close_cols])))
    if not adj_close_cols:
        print("No 'Adj_Close' columns found in the formation data.")
        return pd.DataFrame()

    adj_close_df = formation_data[adj_close_cols].copy()
    adj_close_df.columns = etf_names
    returns_df = adj_close_df.pct_change().dropna()
    if returns_df.empty:
        return pd.DataFrame()

    returns_mean = returns_df.mean()
    returns_std = returns_df.std().replace(0, 1)
    returns_df_standardized = (returns_df - returns_mean) / returns_std

    pca = PCA(n_components=10, random_state=42)
    try:
        principal_components = pca.fit_transform(returns_df_standardized.T)
    except ValueError as e:
        print(f"PCA failed: {e}")
        return pd.DataFrame()

    principal_df = pd.DataFrame(data=principal_components, index=etf_names,
                                columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])

    optics_model = OPTICS(min_samples=2, xi=0.05, min_cluster_size=0.1)
    optics_model.fit(principal_df)
    labels = optics_model.labels_

    features_df = pd.DataFrame(index=etf_names)
    features_df['Cluster'] = labels
    unique_labels = set(labels)
    unique_labels.discard(-1)
    clusters = sorted(unique_labels)
    if not clusters:
        print("No clusters found using OPTICS.")
        return pd.DataFrame()

    pair_results = []
    discrete_lags = [20, 50, 100, 200]  # Define lags for Hurst computation

    for cluster_label in clusters:
        etfs_in_cluster = features_df[features_df['Cluster'] == cluster_label].index.tolist()
        if len(etfs_in_cluster) < 2:
            continue

        pairs = list(combinations(etfs_in_cluster, 2))

        for pair in pairs:
            etf1, etf2 = pair
            series1 = adj_close_df[etf1]
            series2 = adj_close_df[etf2]

            stats = calculate_pair_statistics(series1, series2)
            if stats:
                spread = series1 - series2
                hurst_results = calculate_hurst_exponent(spread.values, discrete_lags)

                if isinstance(hurst_results, dict):
                    stats['Hurst_Exponent'] = hurst_results
                    stats['Average_Hurst'] = np.mean(list(hurst_results.values()))
                else:
                    stats['Hurst_Exponent'] = np.nan
                    stats['Average_Hurst'] = np.nan

                stats['Cluster'] = cluster_label  # Add Cluster information
                pair_results.append(stats)

    if not pair_results:
        print("No pair statistics were calculated.")
        return pd.DataFrame()

    results_df = pd.DataFrame(pair_results)

    if 'Correlation' in results_df.columns:
        results_df = results_df.dropna(subset=['Correlation'])
    else:
        print("No valid pairs found with calculated Correlation.")
        return pd.DataFrame()

    selected_pairs = pd.DataFrame()
    for cluster_label in clusters:
        cluster_pairs = results_df[results_df['Cluster'] == cluster_label]
        if cluster_pairs.empty:
            continue
        spread_std_threshold = cluster_pairs['Spread_STD'].median()

        filtered_pairs = cluster_pairs[
            (cluster_pairs['Correlation'] >= 0.8) &
            (cluster_pairs['Cointegration_PValue'] <= 0.05) &
            (cluster_pairs['Average_Hurst'] < 0.5) &
            (cluster_pairs['Half_Life'] >= 5) &
            (cluster_pairs['Half_Life'] <= trading_period_days) &
            (cluster_pairs['Spread_STD'] <= spread_std_threshold)
        ]

        if not filtered_pairs.empty:
            selected_pairs = pd.concat([selected_pairs, filtered_pairs], ignore_index=True)

    if selected_pairs.empty:
        print("No pairs met the selection criteria.")
        return pd.DataFrame()

    selected_pairs['Method'] = 'Clustering using OPTICS'
    selected_pairs['Formation_Start'] = formation_start_date
    selected_pairs['Formation_End'] = formation_end_date
    selected_pairs['Trading_Start'] = trading_start_date
    selected_pairs['Trading_End'] = trading_end_date

    columns_order = ['Pair', 'Cluster', 'Method', 'Correlation', 'Cointegration_TStats', 'Cointegration_PValue',
                     'Hurst_Exponent', 'Average_Hurst', 'Half_Life', 'Spread_STD',
                     'Formation_Start', 'Formation_End', 'Trading_Start', 'Trading_End']
    selected_pairs = selected_pairs[columns_order]

    return selected_pairs
