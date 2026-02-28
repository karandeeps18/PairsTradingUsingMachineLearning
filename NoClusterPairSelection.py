import pandas as pd
import numpy as np
from itertools import combinations
from utils import calculate_pair_statistics, calculate_hurst_exponent

def select_pairs_no_clustering(formation_data, formation_start_date, formation_end_date,
                               trading_start_date, trading_end_date, trading_period_days):
    adj_close_cols = [col for col in formation_data.columns if 'adj_close' in col]
    etf_names = sorted(list(set([col.split('_')[0] for col in adj_close_cols])))
    
    if not adj_close_cols:
        print("No 'Adj_Close' columns found in the formation data.")
        return pd.DataFrame()

    adj_close_df = formation_data[adj_close_cols]
    adj_close_df.columns = etf_names

    # Generate all possible pairs n(n-1)
    pairs = list(combinations(etf_names, 2))
    pair_results = []
    discrete_lags = [20, 50, 100, 200]  #  lags for Hurst computation

    # Calculate statistics for each pair
    for pair in pairs:
        etf1, etf2 = pair
        series1 = adj_close_df[etf1]
        series2 = adj_close_df[etf2]
        
        stats = calculate_pair_statistics(series1, series2)
        if stats:
            # Compute and integrate the Hurst Exponent for the spread
            spread = series1 - series2
            hurst_results = calculate_hurst_exponent(spread.values, discrete_lags)
            
            # Store results
            if isinstance(hurst_results, dict):
                stats['Hurst_Exponent'] = hurst_results  # Store Hurst results for all lags
            else:
                stats['Hurst_Exponent'] = np.nan  # Handle errors gracefully
            
            pair_results.append(stats)

    results_df = pd.DataFrame(pair_results)

    if 'Correlation' in results_df.columns:
        results_df = results_df.dropna(subset=['Correlation'])
    else:
        print("No valid pairs found with calculated Correlation.")
        return pd.DataFrame()  # Return empty DataFrame if Correlation column is missing

    # Selection criteria
    correlation_threshold = 0.8
    cointegration_pvalue_threshold = 0.05
    spread_std_threshold = results_df['Spread_STD'].median() if not results_df.empty else None
    min_half_life = 5
    max_half_life = trading_period_days
    # average lag
    results_df['Average_Hurst'] = results_df['Hurst_Exponent'].apply(
        lambda x: np.mean(list(x.values())) if isinstance(x, dict) else np.nan
    )

    # Select pairs based on selection criteria
    selected_pairs = results_df[
        (results_df['Correlation'] >= correlation_threshold) &
        (results_df['Cointegration_PValue'] <= cointegration_pvalue_threshold) &
        (results_df['Spread_STD'] <= spread_std_threshold) &
        (results_df['Average_Hurst'] < 0.5) &                       #  criterion for Hurst Exponent 0.5 for half life value 
        (results_df['Half_Life'] >= min_half_life) &
        (results_df['Half_Life'] <= max_half_life)
    ]

    if selected_pairs.empty:
        print("No pairs met the selection criteria.")
        return pd.DataFrame()

    # Add metadata
    selected_pairs['Method'] = 'No Clustering'
    selected_pairs['Formation_Start'] = formation_start_date
    selected_pairs['Formation_End'] = formation_end_date
    selected_pairs['Trading_Start'] = trading_start_date
    selected_pairs['Trading_End'] = trading_end_date

    # Reorder columns
    columns_order = [
        'Pair', 'Method', 'Correlation', 'Cointegration_TStats', 'Cointegration_PValue',
        'Hurst_Exponent', 'Average_Hurst', 'Half_Life', 'Spread_STD',
        'Formation_Start', 'Formation_End', 'Trading_Start', 'Trading_End'
    ]
    selected_pairs = selected_pairs[columns_order]

    return selected_pairs
