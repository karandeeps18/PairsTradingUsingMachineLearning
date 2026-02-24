# preprocess_etf_data.py

"""
Script to preprocess historical OHLCV data for theETFs.

This script performs the following tasks:
1. Loads the raw OHLCV data from  'OHLCV.csv'.
2. Validates data integrity by checking for duplicate dates and missing values.
3. Extracts 'Adj_Close' columns necessary for pair selection.
4. Removes ETFs (columns) that contain any missing 'Adj_Close' data.
5. Removes ETFs with constant 'Adj_Close' data (no variance).
6. Saves the preprocessed 'Adj_Close' data to 'Adj_Close_ETFs.csv' for use in pair selection scripts.
7. Logs the removed ETFs for auditing purposes.
"""

import pandas as pd
import sys
import os

def load_data(file_path):
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df
    except FileNotFoundError:
        print(f" Error: File '{file_path}' not found. Please ensure the file exists in the specified directory.")
        sys.exit(1)
    except Exception as e:
        print(f" Error loading data from '{file_path}': {e}")
        sys.exit(1)

def validate_data(df):

    # Check for duplicate dates
    duplicate_dates = df.index[df.index.duplicated()].unique()
    if len(duplicate_dates) > 0:
        print(f"⚠️ Warning: Found {len(duplicate_dates)} duplicate dates. Removing duplicates...")
        df = df[~df.index.duplicated(keep='first')]
        print(" Duplicate dates removed.")
    else:
        print("No duplicate dates found.")

    # Check for missing values
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        print(f"⚠️ Warning: Found {total_missing} missing values in the dataset.")
        print("🔄 Handling missing data by forward-filling and backward-filling...")
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        remaining_missing = df.isnull().sum().sum()
        if remaining_missing > 0:
            print(f"⚠️ Warning: Still {remaining_missing} missing values after filling.")
        else:
            print(" All missing values have been handled.")
    else:
        print(" No missing values found in the dataset.")

    return df

def extract_adj_close(df, expected_etfs=100):
    print("\n📊 **Extracting 'Adj Close' Columns**")
    # Identify columns that end with '_Adj_Close'
    adj_close_cols = [col for col in df.columns if col.endswith('_adj_close')]
    actual_etfs = len(adj_close_cols)

    if actual_etfs == 0:
        print(" Error: No 'Adj Close' columns found in the dataset.")
        sys.exit(1)
    elif actual_etfs < expected_etfs:
        print(f"⚠️ Warning: Expected {expected_etfs} 'Adj Close' columns, but found {actual_etfs}.")
        # Optionally, list missing ETFs if you have the original list
    else:
        print(f" Found {actual_etfs} 'Adj Close' columns as expected.")

    # Extract 'Adj_Close' columns
    adj_close_df = df[adj_close_cols].copy()

    print(f" 'Adj_Close' columns have been extracted.")
    return adj_close_df

def remove_etfs_with_missing_data(adj_close_df, log_file='removed_etfs.log'):
    print("\n🗑️ **Removing ETFs with Missing 'Adj_Close' Data**")
    # Identify ETFs with any missing values
    etfs_with_missing = adj_close_df.columns[adj_close_df.isnull().any()].tolist()
    num_etfs_with_missing = len(etfs_with_missing)

    if num_etfs_with_missing > 0:
        print(f"⚠️ Warning: Found {num_etfs_with_missing} ETFs with missing 'Adj_Close' data. Removing these ETFs...")
        adj_close_df = adj_close_df.drop(columns=etfs_with_missing)
        print(f"✅ Removed {num_etfs_with_missing} ETFs: {etfs_with_missing}")

        # Log the removed ETFs
        with open(log_file, 'a') as f:
            f.write(f"Removed ETFs due to missing 'Adj_Close' data:\n")
            for etf in etfs_with_missing:
                f.write(f"{etf}\n")
    else:
        print("✅ No ETFs with missing 'Adj_Close' data found.")

    return adj_close_df

def remove_etfs_with_constant_data(adj_close_df, log_file='removed_etfs.log'):
    print("\n🗑️ **Removing ETFs with Constant 'Adj_Close' Data**")
    # Identify ETFs with zero variance (constant values)
    etfs_with_zero_variance = adj_close_df.columns[adj_close_df.nunique() <= 1].tolist()
    num_etfs_with_zero_variance = len(etfs_with_zero_variance)
    if num_etfs_with_zero_variance > 0:
        print(f"⚠️ Warning: Found {num_etfs_with_zero_variance} ETFs with constant 'Adj_Close' data. Removing these ETFs...")
        adj_close_df = adj_close_df.drop(columns=etfs_with_zero_variance)
        print(f" Removed {num_etfs_with_zero_variance} ETFs: {etfs_with_zero_variance}")

        # Log the removed ETFs
        with open(log_file, 'a') as f:
            f.write(f"Removed ETFs due to constant 'Adj_Close' data:\n")
            for etf in etfs_with_zero_variance:
                f.write(f"{etf}\n")
    else:
        print("  No ETFs with constant 'Adj_Close' data found.")

    return adj_close_df

def save_preprocessed_data(adj_close_df, output_file='Adj_Close_ETFs.csv'):
    print(f"\n **Saving Preprocessed Data to '{output_file}'**")
    try:
        adj_close_df.to_csv(output_file)
        print(f" Preprocessed data has been saved to '{output_file}'.")
    except Exception as e:
        print(f" Error saving preprocessed data to '{output_file}': {e}")
        sys.exit(1)

def main():
    raw_data_file = 'Energy_ETF_price_data.csv'
    preprocessed_data_file = 'preprocessed_etfs.csv'
    log_file = 'removed_etfs.log'
    if os.path.exists(log_file):
        os.remove(log_file)
    if not os.path.exists(raw_data_file):
        print(f"❌ Error: Raw data file '{raw_data_file}' does not exist. Please run the data download script first.")
        sys.exit(1)

    df = load_data(raw_data_file)
    df = validate_data(df)
    adj_close_df = extract_adj_close(df, expected_etfs=100)
    
    # Remove ETFs with any missing 'Adj_Close' data
    adj_close_df = remove_etfs_with_missing_data(adj_close_df, log_file=log_file)

     # Remove ETFs with constant 'Adj_Close' data
    adj_close_df = remove_etfs_with_constant_data(adj_close_df, log_file=log_file)

    # Final check for any remaining missing values
    final_missing = adj_close_df.isnull().sum().sum()
    if final_missing > 0:
        print(f"\n Error: There are still {final_missing} missing values in '{preprocessed_data_file}'. Please address them manually.")
        sys.exit(1)
    else:
        print("\n No missing values remain in the preprocessed 'Adj_Close' data.")


    constant_after_cleaning = adj_close_df.columns[adj_close_df.nunique() <= 1].tolist()
    if constant_after_cleaning:
        print(f"\n Error: ETFs with constant 'Adj_Close' data still exist: {constant_after_cleaning}. Please address them manually.")
        sys.exit(1)
    else:
        print("No ETFs with constant 'Adj_Close' data remain.")

    # Save the preprocessed data
    save_preprocessed_data(adj_close_df, preprocessed_data_file)

    print("\n🎉 Data preprocessing completed successfully.")
    print(f"📄 Details of removed ETFs can be found in '{log_file}'.")

if __name__ == "__main__":
    main()
