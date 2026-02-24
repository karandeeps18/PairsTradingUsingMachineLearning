import pandas as pd
import matplotlib.pyplot as plt
import os
os.makedirs("plots", exist_ok=True)

# Load data
pairs = pd.read_csv("all_selected_pairs.csv")
prices = pd.read_csv("preprocessed_etfs.csv", index_col=0, parse_dates=True)

# Group by cluster to see how the OPTICS algorithm actually behaved
for cluster, group in pairs.groupby('Cluster'):
    plt.figure(figsize=(12, 7))
    
    for _, row in group.iterrows():
        # parse string
        p_text = row['Pair'].replace("(", "").replace(")", "").replace("'", "")
        t1, t2 = [t.strip() for t in p_text.split(',')]
        
        # Normalize to 1.0 for visual comparison
        p1 = prices[f"{t1}_adj_close"] / prices[f"{t1}_adj_close"].iloc[0]
        p2 = prices[f"{t2}_adj_close"] / prices[f"{t2}_adj_close"].iloc[0]
        
        plt.plot(p1, alpha=0.6, label=t1)
        plt.plot(p2, alpha=0.6, label=t2)
    
    plt.title(f"Cluster {cluster}: Convergence Check")
    plt.legend(ncol=2, fontsize='small') # note: clusters can have many lines
    plt.grid(alpha=0.3)
    plt.savefig(f"plots/cluster_{cluster}_check.png")
    plt.close()

print("Cluster plots generated.")
