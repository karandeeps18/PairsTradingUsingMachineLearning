#  Pairs Trading Using Machine Learning

This repository implements a robust framework for market-neutral pairs trading using a combination of **statistical techniques**, **theme-based clustering**, and **unsupervised learning (OPTICS)** for pair selection, followed by **LSTM modeling** for spread prediction and trading signal generation.
This project is the part of Fall 2024 Advanced Machine Learning (QFGB-890H-001) 

##  Overview

Pairs trading is a classic quantitative strategy that exploits mean-reverting relationships between two assets. This project extends traditional statistical arbitrage by integrating:

- Statistical Filters (Correlation, Cointegration, Hurst Exponent, Half-Life)
- Theme-Based Clustering (GICS segments of Energy ETFs)
- OPTICS Clustering with PCA for unsupervised pair discovery
- LSTM neural networks for predicting future spread movements
- Full backtesting engine with performance metrics (Sharpe, Sortino, MDD, etc.)

our approach enhances traditional methods through unsupervised learning and temporal modeling using LSTMs.

---

##  Methodology

### 1. Pair Selection Framework

We implement 3 strategies to select statistically valid pairs:

- **No Clustering (Statistical Criteria)**  
  - Correlation ≥ 0.8  
  - Cointegration p-value ≤ 0.05  
  - Hurst Exponent < 0.5  
  - Half-Life ∈ [5, 90]

- **Theme-Based Clustering**  
  - Groups ETFs by economic segment using GICS (e.g., MLPs, Oil & Gas, etc.)
  - Applies same filters within each cluster

- **OPTICS Clustering + PCA**  
  - Reduces high-dimensional ETF return data
  - Identifies clusters with no assumption of shape/density
  - Applies filtering within discovered clusters

### 2. LSTM-Based Spread Prediction

- Predicts future spread (St+1) using historical spread and engineered features
- Generates signals: BUY / SELL / EXIT / FLAT based on predicted change
- Trained using early stopping and hyperparameter tuning

---

##  Performance Evaluation

Metrics tracked during backtesting:
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Profit & Loss
- Trade Count

### LSTM vs Statistical Model  
- **LSTM Strategy** outperforms traditional momentum-based pairs trading with:
  - Higher Sharpe (2.05 vs 0.93)
  - Lower Drawdown (−10.47% vs −13.53%)

---

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, statsmodels, ta
- tensorflow/keras for LSTM

-- 

## License
This repository is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0).






