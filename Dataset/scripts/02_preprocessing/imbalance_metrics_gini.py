import numpy as np
import pandas as pd

def gini(counts):
    """Calculate the Gini coefficient of a distribution of counts."""
    counts = np.array(counts, dtype=float)
    counts = np.sort(counts)
    n = len(counts)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * counts)) / (n * np.sum(counts)) - (n + 1) / n

# Paths to datasets
raw_path = r"C:\Users\goura\Documents\Master Thesis\Dataset\data\icecat_train_en_filtered.csv"
balanced_path = r"C:\Users\goura\Documents\Master Thesis\Dataset\data\icecat_hierarchically_balanced.csv"

# 1. Analyze Raw Corpus
print("Analyzing Raw Corpus Imbalance...")
try:
    df_raw = pd.read_csv(raw_path, low_memory=False)
    counts_raw = df_raw["leaf_category"].value_counts().values
    gini_raw = gini(counts_raw)

    print(f"Total products      : {len(df_raw):,}")
    print(f"Level 3 categories  : {len(counts_raw)}")
    print(f"Max category size   : {counts_raw.max():,}")
    print(f"Min category size   : {counts_raw.min():,}")
    print(f"Imbalance ratio     : {counts_raw.max() / counts_raw.min():.1f}:1")
    print(f"Gini coefficient    : {gini_raw:.4f}")
except FileNotFoundError:
    print(f"Raw data file not found at: {raw_path}")
    gini_raw = None

# 2. Analyze Balanced Corpus
print("\nAnalyzing Balanced Corpus Imbalance...")
try:
    df_bal = pd.read_csv(balanced_path)
    counts_bal = df_bal["leaf_category"].value_counts().values
    gini_bal = gini(counts_bal)

    print(f"Total products      : {len(df_bal):,}")
    print(f"Level 3 categories  : {len(counts_bal)}")
    print(f"Max category size   : {counts_bal.max():,}")
    print(f"Min category size   : {counts_bal.min():,}")
    print(f"Imbalance ratio     : {counts_bal.max() / counts_bal.min():.1f}:1")
    print(f"Gini coefficient    : {gini_bal:.4f}")
except FileNotFoundError:
    print(f"Balanced data file not found at: {balanced_path}")
    gini_bal = None

# 3. Summary
if gini_raw is not None and gini_bal is not None:
    reduction = (gini_raw - gini_bal) / gini_raw * 100
    print("\nSummary of Gini Reduction")
    print(f"Before balancing    : {gini_raw:.4f}")
    print(f"After balancing     : {gini_bal:.4f}")
    print(f"Reduction           : {reduction:.1f}%")
