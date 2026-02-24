import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

# Set up output
out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)

# Data loading
pairs = pd.read_csv("all_selected_pairs.csv")
prices = pd.read_csv("preprocessed_etfs.csv", index_col=0, parse_dates=True)

# Global Market Context
plt.figure(figsize=(10, 8))
sns.heatmap(prices.corr(), cmap="coolwarm", xticklabels=False, yticklabels=False)
plt.title("Asset Correlation Matrix")
plt.savefig(f"{out_dir}/market_corr.png")

# Pair Selection Stats (Histograms in one figure for quick review)
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(pairs['Average_Hurst'], kde=True, ax=ax[0], color='blue').set_title('Hurst Distribution')
sns.histplot(pairs['Half_Life'], bins=20, ax=ax[1], color='green').set_title('Half-Life (Days)')
sns.histplot(pairs['Cointegration_PValue'], bins=30, ax=ax[2], color='red').set_title('Coint P-Values')
plt.tight_layout()
plt.savefig(f"{out_dir}/selection_metrics.png")

# Individual Pair Analysis (Focus on Spread Dynamics)
# We only plot the top 5 pairs
for _, row in pairs.head(5).iterrows():
    p_text = row['Pair'].replace("(", "").replace(")", "").replace("'", "")
    t1, t2 = [t.strip() for t in p_text.split(',')]
    
    # Calculate Spread
    s1, s2 = prices[f"{t1}_adj_close"], prices[f"{t2}_adj_close"]
    spread = s1 - s2
    
    # Normalized Plot (Visual Cointegration Check)
    plt.figure(figsize=(12, 5))
    plt.plot(s1/s1[0], label=t1, alpha=0.8)
    plt.plot(s2/s2[0], label=t2, alpha=0.8)
    plt.title(f"Price Convergence: {t1} vs {t2}")
    plt.legend()
    plt.savefig(f"{out_dir}/price_{t1}_{t2}.png")
    
    # Spread Stationarity (ADF Check)
    # p-value
    adf_p = adfuller(spread.diff().dropna())[1]
    plt.figure(figsize=(10, 4))
    spread.plot(title=f"Spread {t1}-{t2} (ADF p={adf_p:.4f})")
    plt.axhline(spread.mean(), color='red', linestyle='--')
    plt.savefig(f"{out_dir}/spread_{t1}_{t2}.png")
    plt.close('all')

print(f"Visualizations dumped to {out_dir}/")
