import pandas as pd
import sys

# Load raw energy data
# Note: Using index_col=0 because the first column is always the date/timestamp
df = pd.read_csv('Energy_ETF_price_data.csv', index_col=0, parse_dates=True)

# Drop exact time duplicates - sometimes happens with vendor data merges
df = df[~df.index.duplicated(keep='first')]

# Grabbing only the adj_close columns for the pairs model
adj_cols = [c for c in df.columns if c.endswith('_adj_close')]
data = df[adj_cols].copy()

print(f"Starting with {len(data.columns)} tickers...")

# Cleaning ---

# Drop tickers with any NaNs. 
# NNs crash on nulls and ffill can introduce lookahead bias if not careful.
bad_tickers = data.columns[data.isna().any()].tolist()
if bad_tickers:
    print(f"Dropping {len(bad_tickers)} tickers with gaps: {bad_tickers}")
    data = data.drop(columns=bad_tickers)

# Remove "Dead" tickers (constant price / zero variance)
stale = data.columns[data.nunique() <= 1].tolist()
if stale:
    print(f"Dropping {len(stale)} stale/dead tickers: {stale}")
    data = data.drop(columns=stale)

# Final check
print(f"Cleaned dataset: {data.shape[1]} tickers remaining.")
data.to_csv('preprocessed_etfs.csv')
print("Done. Saved to preprocessed_etfs.csv")
