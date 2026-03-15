import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# --- CONFIG ---
INPUT_CSV = r'C:\Users\goura\Documents\Master Thesis\Dataset\data\icecat_hierarchically_balanced.csv'
INPUT_EMBEDDINGS = r'C:\Users\goura\Documents\Master Thesis\Dataset\data\umap_5d.npy'

# 1. Load Data
print("Loading Data...")
df = pd.read_csv(INPUT_CSV)
X = np.load(INPUT_EMBEDDINGS)
if len(df) != len(X): df = df.iloc[:len(X)]

# 2. Extract True Category
def get_true_cat(path):
    if pd.isna(path): return "Unknown"
    return str(path).split('>')[-1].strip()

df['True_Category'] = df['pathlist_names'].apply(get_true_cat)
top_cats = df['True_Category'].value_counts().head(20).index
subset = df[df['True_Category'].isin(top_cats)].copy()
X_subset = X[subset.index] # Get subset embeddings to speed up if needed, but we used full X for training

# 3. Run Models (Ward vs Single)
# We use K=411 (Baseline)
k = 411
print(f"Running Ward Linkage (K={k})...")
ward = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(X)

print(f"Running Single Linkage (K={k})...")
single = AgglomerativeClustering(n_clusters=k, linkage='single').fit(X)

# Assign labels to the subset dataframe
subset['Ward_ID'] = ward.labels_[subset.index]
subset['Single_ID'] = single.labels_[subset.index]

# 4. Generate Heatmaps
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

def plot_heatmap(ax, col, title):
    # Crosstab
    matrix = pd.crosstab(subset['True_Category'], subset[col])
    # Normalize
    matrix_norm = matrix.div(matrix.sum(axis=1), axis=0)
    # Sort
    idx = matrix_norm.idxmax(axis=1).sort_values().index
    matrix_sorted = matrix_norm.loc[idx]
    
    sns.heatmap(matrix_sorted, ax=ax, cmap="Blues", cbar=False, xticklabels=False)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("True Category")
    ax.set_xlabel("Predicted Clusters")

plot_heatmap(axes[0], 'Ward_ID', "Ward Linkage\n(Balanced Variance)")
plot_heatmap(axes[1], 'Single_ID', "Single Linkage\n(Nearest Point)")

plt.tight_layout()
plt.savefig('linkage_confusion_comparison.png', dpi=300)
print("Saved 'linkage_confusion_comparison.png'. Check for the 'Giant Line' in Single Linkage.")
plt.show()
