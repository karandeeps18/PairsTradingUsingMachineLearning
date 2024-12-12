import pandas as pd

def filter_best_pairs(selected_pairs_file):
    # Load the selected pairs DataFrame
    selected_pairs_df = pd.read_csv(selected_pairs_file)

    # Standardize the Pair column to have consistent formatting
    selected_pairs_df['Pair'] = selected_pairs_df['Pair'].apply(lambda x: '-'.join(sorted(x.replace("(", "").replace(")", "").replace("'", "").split(','))))

    # Define selection criteria for the best pairs
    correlation_threshold = 0.9
    cointegration_pvalue_threshold = -3.5
    hurst_exponent_threshold = 0.5
    min_half_life = 30
    max_half_life = 60

    # Keep pairs that are either 'Clustering by Segment' or meet the other criteria for best pairs
    filtered_pairs = selected_pairs_df[
        (
            (selected_pairs_df['Correlation'] >= correlation_threshold) &
            (selected_pairs_df['Cointegration_PValue'] <= cointegration_pvalue_threshold) &
            (selected_pairs_df['Hurst_Exponent'] < hurst_exponent_threshold) &
            (selected_pairs_df['Half_Life'] >= min_half_life) &
            (selected_pairs_df['Half_Life'] <= max_half_life) 
        )
    ]

    # Save the filtered pairs to a new CSV file
    filtered_pairs.to_csv('filtered_best_pairs.csv', index=False)

    return filtered_pairs

# Example usage
if __name__ == "__main__":
    filtered_pairs = filter_best_pairs('all_selected_pairs.csv')
    print("Filtered pairs saved to 'filtered_best_pairs.csv'")
