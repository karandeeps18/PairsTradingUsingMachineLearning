import yfinance as yf
import pandas as pd

tickers_df = pd.read_csv('energy_etf_descriptions.csv')
tickers = tickers_df['Ticker'].tolist()

# start and end dates 
start_date = '2015-01-01'
end_date = '2024-11-30'
combined_data = pd.DataFrame()


for ticker in tickers:
    ohlcv_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    ohlcv_data = ohlcv_data.rename(columns={
        'Open': f'{ticker}_open',
        'High': f'{ticker}_high',
        'Low': f'{ticker}_low',
        'Adj Close': f'{ticker}_adj_close',
        'Volume': f'{ticker}_volume'
    })
    ohlcv_data = ohlcv_data.drop(columns=['Close'])
    if combined_data.empty:
        combined_data = ohlcv_data
    else:
        combined_data = combined_data.join(ohlcv_data, how='outer')
combined_data.to_csv('Energy_ETF_price_data.csv')