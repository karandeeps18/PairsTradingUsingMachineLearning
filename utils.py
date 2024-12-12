import pandas as pd
import numpy as np
from scipy.stats import pearsonr, skew, kurtosis
from statsmodels.tsa.stattools import coint, adfuller
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import warnings
from statsmodels.tsa.ar_model import AutoReg

warnings.filterwarnings("ignore")

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return df

# Stationarity check
def is_not_stationary(series, significance_level=0.05):
    """
    Check if a time series is stationary using the Augmented Dickey-Fuller test
    
    Input:
        series (pd.Series): The time series to check for stationarity.
        significance_level (float): The significance level for the test. Default is 0.05.  
    
    Returns:
        bool: True if the series is non-stationary (p-value > significance_level)
    """
    if series.nunique() <= 1:
        print(f"⚠️ Warning: Series '{series.name}' is constant. Automatically considered non-stationary.")
        return True  # Consider constant series as non-stationary

    try:
        adf_result = adfuller(series, regression="ct", autolag="AIC")
        p_value = adf_result[1]
        is_non_stationary = p_value > significance_level
        return is_non_stationary
    except ValueError as e:
        print(f"Error performing ADF test on '{series.name}': {e}")
        return True  # Treat errors as non-stationary
    except Exception as e:
        print(f"Unexpected error during ADF test on '{series.name}': {e}")
        return True  # Treat unexpected errors as non-stationary

# Updated Engle-Granger test to handle constant series
def egle_granger_test_bidirectional(series1, series2):
    """
    Perform Engle-Granger test in both directions to check for cointegration.
    
    Input:
        series1 (pd.Series): First time series.
        series2 (pd.Series): Second time series.
    
    Returns:
        dict: Contains t-statistics and p-value of the better cointegration result.
    """
    # Skip testing if either series is constant
    if series1.nunique() <= 1 or series2.nunique() <= 1:
        print(f"⚠️ Warning: One or both series are constant. Skipping Engle-Granger test.")
        return {'tstat': np.nan, 'pvalue': np.nan}
    
    # Test 1: Series1 ~ Series2
    try:
        test1 = coint(series1, series2)
        tstat1, pvalue1 = test1[0], test1[1]
    except ValueError as e:
        print(f" Error performing cointegration test on '{series1.name}' and '{series2.name}': {e}")
        tstat1, pvalue1 = np.nan, np.nan
    
    # Test 2: Series2 ~ Series1
    try:
        test2 = coint(series2, series1)
        tstat2, pvalue2 = test2[0], test2[1]
    except ValueError as e:
        print(f" Error performing cointegration test on '{series2.name}' and '{series1.name}': {e}")
        tstat2, pvalue2 = np.nan, np.nan
    
    # Select the test with the lowest t-statistics 
    if np.isnan(tstat1) or (not np.isnan(tstat2) and tstat2 < tstat1):
        return {'tstat': tstat2, 'pvalue': pvalue2}
    else:
        return {'tstat': tstat1, 'pvalue': pvalue1}

# Hurst Exponent calculation
def calculate_hurst_exponent(p, lags):
    """
    Calculate the Hurst Exponent of a time series."""
    hurst_results = {}
    for lag in lags: 
        if lag >= len(p):
            continue
        
        pp = np.subtract(p[lag:], p[:-lag]) # compute price diff
        variancetau = np.var(pp) # compute variance of the diff
        log_lag = np.log10(lag)  # logs for regression 
        log_variance = np.log10(variancetau)
        hurst = log_variance / log_lag # hurst exponent can be directly calcualted from the log-log slope
        hurst_results[lag] = hurst/2 # divide by 2
    return hurst_results 
discrete_lags = [20, 100, 250, 500, 1000]

        


# Half-life calculation
def calculate_half_life(spread):
    """
    Calculate half-life of the mean reversion speed.
    
    Input:
        spread (pd.Series): Spread series.
    
    Returns:
        float: Half-life in days.
    """
    spread = spread.dropna()
    if len(spread) < 2:
        return np.nan 
    
    model = AutoReg(spread, lags=1, old_names=False)
    res = model.fit()
    phi = res.params[1]
    
    if phi >= 1 or phi <= -1:
        return np.nan  # Non-Stationary process
    
    lambda_param = -np.log(phi)
    if lambda_param <= 0:
        return np.nan  # Invalid
    
    half_life = np.log(2) / lambda_param
    return half_life

# Pair statistics calculation
def calculate_pair_statistics(etf1_adj_close, etf2_adj_close, significance_level=0.05, hurst_max_lag=100):
    """
    Calculate statistical measures for a pair of ETFs.

    Input: Etf1 and ETF2 adj_close data

    Parameters:
        etf1_adj_close (pd.Series): Adjusted close prices for ETF1.
        etf2_adj_close (pd.Series): Adjusted close prices for ETF2. 

    Returns:
        dict: A dictionary containing correlation, cointegration p-value, and spread standard deviation.
    """
    prices = pd.concat([etf1_adj_close, etf2_adj_close], axis=1).dropna()
    if prices.empty: 
        return None 
    
    etf1, etf2 = prices.columns
    etf1_series = prices[etf1]
    etf2_series = prices[etf2]
    
    # Test for non-stationarity I[1] process
    non_stationary1 = is_not_stationary(etf1_series, significance_level)
    non_stationary2 = is_not_stationary(etf2_series, significance_level)
     
    if not (non_stationary1 and non_stationary2):
        return None  # One or both series are stationary, skip this pair
    
    # Cointegration Test
    coint_test = egle_granger_test_bidirectional(etf1_series, etf2_series)
    coint_t_stat = coint_test['tstat']
    Cointegration_PValue = coint_test['pvalue']

    # Calculate spread
    spread = etf1_series - etf2_series 
    
    # Calculate Hurst exponent
    hurst_results = calculate_hurst_exponent(spread.values, discrete_lags)
    
    # Half-life calculation
    half_life = calculate_half_life(spread)
    
    # Correlation calculation
    etf1_returns = etf1_series.pct_change().dropna()
    etf2_returns = etf2_series.pct_change().dropna()
    corr_result = pearsonr(etf1_returns, etf2_returns)
    corr_coef = corr_result[0]

    # Spread STD Dev
    spread_std = spread.std()
    
    return {
        'Pair': (etf1, etf2),
        'Correlation': corr_coef,
        'Cointegration_TStats': coint_t_stat,
        'Cointegration_PValue': Cointegration_PValue,
        'Hurst_Exponent': hurst_results, 
        'Half_Life': half_life, 
        'Spread_STD': spread_std
    }

# Feature extraction
def extract_features(etf_prices):
    """
    Extract features from ETF price data.
    
    Input:
        etf_prices (pd.Series): Adjusted close prices for an ETF.
    
    Returns:
        dict: A dictionary containing feature metrics like mean return, standard deviation, skewness, etc.
    """
    etf_returns = etf_prices.pct_change().dropna()
    features = {
        'Mean_Return': etf_returns.mean(),
        'Std_Return': etf_returns.std(),
        'Skewness': skew(etf_returns),
        'Kurtosis': kurtosis(etf_returns),
        'Avg_RSI': RSIIndicator(close=etf_prices, window=14).rsi().mean(),
        'Avg_SMA': SMAIndicator(close=etf_prices, window=14).sma_indicator().mean()
    }
    return features
