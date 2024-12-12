import pandas as pd
from itertools import combinations
from utils import calculate_pair_statistics
from utils import calculate_hurst_exponent
import numpy as np

# """
# Pair Selection using Theme Clustering.

# This module implements the pair selection process by grouping ETFs into predefined
# categories (themes) and selecting pairs within each category.
# """

# def select_pairs_theme_clustering(formation_data, formation_start_date, formation_end_date,
#                                   trading_start_date, trading_end_date, trading_period_days):
#     # List adjusted close columns and ETF names
#     adj_close_cols = [col for col in formation_data.columns if 'adj_close' in col]
#     etf_names = list(set([col.split('_')[0] for col in adj_close_cols]))
#     adj_close_df = formation_data[adj_close_cols]
#     adj_close_df.columns = etf_names  

#     # Load ETF categories
#     etf_info = pd.read_csv('energy_etf_descriptions.csv')  # Ensure the CSV has 'ETF' and 'Segment' columns
#     categories = etf_info['Segment'].unique()
#     pair_results = []

#     # Calculate statistics for each pair within each category
#     for category in categories:
#         etfs_in_category = etf_info[etf_info['Segment'] == category]['Ticker'].tolist()
#         pairs = list(itertools.combinations(etfs_in_category, 2))
#         for pair in pairs:
#             etf1, etf2 = pair

#             # Skip pairs if one or both ETFs are missing in the data
#             if etf1 not in adj_close_df.columns or etf2 not in adj_close_df.columns:
#                 print(f"Skipping pair ({etf1}, {etf2}) as one or both are missing in the data.")
#                 continue

#             stats = calculate_pair_statistics(adj_close_df[etf1], adj_close_df[etf2])
#             if stats is not None:
#                 stats['Segment'] = category  # Segment based
#                 pair_results.append(stats)

#     # Convert the results to a DataFrame
#     results_df = pd.DataFrame(pair_results)

#     # Only proceed if the 'Correlation' column is present in the DataFrame
#     if 'Correlation' in results_df.columns:
#         # Drop rows with NaN values in the 'Correlation' column
#         results_df = results_df.dropna(subset=['Correlation'])
#     else:
#         print("No valid pairs found with calculated Correlation.")
#         return pd.DataFrame()  # Return empty DataFrame if Correlation column is missing

#     # Selection criteria
#     correlation_threshold = 0.8
#     cointegration_pvalue_threshold = 0.05
#     hurst_exponent_threshold = 0.5  # Hurst exponent should be less than 0.5
#     min_half_life = 5  # Minimum half-life in days
#     max_half_life = trading_period_days  # Should not exceed the trading period
#     spread_std_threshold = results_df['Spread_STD'].median() if not results_df.empty else None

#     # Select pairs based on the criteria
#     selected_pairs = pd.DataFrame()
#     for category in categories:
#         category_pairs = results_df[results_df['Segment'] == category]
#         if category_pairs.empty:
#             continue
#         spread_std_threshold = category_pairs['Spread_STD'].median()
#         filtered_pairs = category_pairs[
#             (category_pairs['Correlation'] >= correlation_threshold) &
#             (category_pairs['Cointegration_PValue'] <= cointegration_pvalue_threshold) &
#             (category_pairs['Spread_STD'] <= spread_std_threshold)
#         ]
#         selected_pairs = pd.concat([selected_pairs, filtered_pairs], ignore_index=True)

#     if selected_pairs.empty:
#         print("No pairs met the selection criteria.")
#         return pd.DataFrame()

#     # Add metadata
#     selected_pairs['Method'] = 'Clustering by Segment'
#     selected_pairs['Formation_Start'] = formation_start_date
#     selected_pairs['Formation_End'] = formation_end_date
#     selected_pairs['Trading_Start'] = trading_start_date
#     selected_pairs['Trading_End'] = trading_end_date

#     # Reorder columns
#     columns_order = [
#         'Pair', 'Method', 'Correlation', 'Cointegration_TStats', 'Cointegration_PValue',
#         'Hurst_Exponent', 'Half_Life', 'Spread_STD', 'Formation_Start', 'Formation_End',
#         'Trading_Start', 'Trading_End'
#     ]
#     selected_pairs = selected_pairs[columns_order]

#     return selected_pairs
def select_pairs_theme_clustering(formation_data, formation_start_date, formation_end_date,
                                  trading_start_date, trading_end_date, trading_period_days):
    adj_close_cols = [col for col in formation_data.columns if 'adj_close' in col]
    etf_names = list(set([col.split('_')[0] for col in adj_close_cols]))
    adj_close_df = formation_data[adj_close_cols]
    adj_close_df.columns = etf_names

    etf_info = pd.read_csv('energy_etf_descriptions.csv')
    categories = etf_info['Segment'].unique()
    pair_results = []
    discrete_lags = [20, 50, 100, 200]

    for category in categories:
        etfs_in_category = etf_info[etf_info['Segment'] == category]['Ticker'].tolist()
        pairs = list(combinations(etfs_in_category, 2))
        for pair in pairs:
            etf1, etf2 = pair
            if etf1 not in adj_close_df.columns or etf2 not in adj_close_df.columns:
                continue

            stats = calculate_pair_statistics(adj_close_df[etf1], adj_close_df[etf2])
            if stats:
                spread = adj_close_df[etf1] - adj_close_df[etf2]
                hurst_results = calculate_hurst_exponent(spread.values, discrete_lags)

                if isinstance(hurst_results, dict):
                    stats['Hurst_Exponent'] = hurst_results
                    stats['Average_Hurst'] = np.mean(list(hurst_results.values()))
                else:
                    stats['Hurst_Exponent'] = np.nan
                    stats['Average_Hurst'] = np.nan

                stats['Segment'] = category
                pair_results.append(stats)

    results_df = pd.DataFrame(pair_results)

    if 'Correlation' in results_df.columns:
        results_df = results_df.dropna(subset=['Correlation'])
    else:
        print("No valid pairs found with calculated Correlation.")
        return pd.DataFrame()

    selected_pairs = pd.DataFrame()
    for category in categories:
        category_pairs = results_df[results_df['Segment'] == category]
        if category_pairs.empty:
            continue
        spread_std_threshold = category_pairs['Spread_STD'].median()

        filtered_pairs = category_pairs[
            (category_pairs['Correlation'] >= 0.8) &
            (category_pairs['Cointegration_PValue'] <= 0.05) &
            (category_pairs['Average_Hurst'] < 0.5) &  # Filtering by Average_Hurst
            (category_pairs['Half_Life'] >= 5) &
            (category_pairs['Half_Life'] <= trading_period_days) &
            (category_pairs['Spread_STD'] <= spread_std_threshold)
        ]

        selected_pairs = pd.concat([selected_pairs, filtered_pairs], ignore_index=True)

    if selected_pairs.empty:
        print("No pairs met the selection criteria.")
        return pd.DataFrame()

    selected_pairs['Method'] = 'Clustering by Segment'
    selected_pairs['Formation_Start'] = formation_start_date
    selected_pairs['Formation_End'] = formation_end_date
    selected_pairs['Trading_Start'] = trading_start_date
    selected_pairs['Trading_End'] = trading_end_date

    columns_order = ['Pair', 'Method', 'Correlation', 'Cointegration_TStats', 'Cointegration_PValue', 'Hurst_Exponent',
                     'Average_Hurst', 'Half_Life', 'Spread_STD', 'Formation_Start', 'Formation_End', 'Trading_Start', 'Trading_End']
    selected_pairs = selected_pairs[columns_order]

    return selected_pairs