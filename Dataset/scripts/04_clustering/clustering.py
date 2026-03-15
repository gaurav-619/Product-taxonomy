import pandas as pd
import numpy as np
import hdbscan
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# Configuration
INPUT_CSV = r'C:\Users\goura\Documents\Master Thesis\Dataset\data\icecat_hierarchically_balanced.csv'
INPUT_EMBEDDINGS = r'C:\Users\goura\Documents\Master Thesis\Dataset\data\umap_5d.npy'

# 1. Load Data
print("Loading Data...")
df = pd.read_csv(INPUT_CSV)
X = np.load(INPUT_EMBEDDINGS)

# Ensure alignment
if len(df) != len(X):
    df = df.iloc[:len(X)]

# Run Models to Generate Comparative Labels (K=411 or Natural K)

# HDBSCAN
hdb = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=5).fit(X)
k_target = len(set(hdb.labels_)) - (1 if -1 in hdb.labels_ else 0)

# Others (Forced K)
km = KMeans(n_clusters=k_target, random_state=42).fit(X)
agg = AgglomerativeClustering(n_clusters=k_target, linkage='ward').fit(X)
gmm = GaussianMixture(n_components=k_target, random_state=42).fit(X)
gmm_labels = gmm.predict(X)

# Compile Master Clustering Explorer Dataframe
master_df = df.copy()
master_df['HDB_ID'] = hdb.labels_
master_df['KM_ID'] = km.labels_
master_df['AGG_ID'] = agg.labels_
master_df['GMM_ID'] = gmm_labels

# Select relevant columns for inspection
cols = ['IcecatId', 'Title', 'Brand', 'pathlist_names', 'HDB_ID', 'KM_ID', 'AGG_ID', 'GMM_ID']
# Add description if it exists
if 'LongDesc' in df.columns: cols.insert(3, 'LongDesc')

# Save
output_file = 'Master_Clustering_Explorer.csv'
master_df[cols].to_csv(output_file, index=False)
print(f"DONE. Saved '{output_file}'. Open this in Excel to inspect rows.")
