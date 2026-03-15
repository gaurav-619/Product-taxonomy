import pandas as pd
import numpy as np
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)

# Configuration

BALANCED_CSV = r'C:\Users\goura\Documents\Master Thesis\Dataset\data\icecat_hierarchically_balanced.csv'
EMBEDDINGS   = r'C:\Users\goura\Documents\Master Thesis\Dataset\data\umap_5d.npy'
NAMING_CSV   = r'C:\Users\goura\Documents\Master Thesis\Naming_Robust_Final_clean.csv'
OUTPUT_CSV   = r'C:\Users\goura\Documents\Master Thesis\Final_Evaluation_Results.csv'
K_TARGET     = 411

# Load Data

print("STEP 1: Loading data...")
try:
    df = pd.read_csv(BALANCED_CSV)
    X  = np.load(EMBEDDINGS)
except FileNotFoundError as e:
    print(f"ERROR: Could not find a required file. {e}")
    exit()

print(f"  CSV shape        : {df.shape}")
print(f"  Embeddings shape : {X.shape}")
print(f"  Match            : {len(df) == len(X)}")

# Align if mismatch
if len(df) != len(X):
    min_len = min(len(df), len(X))
    df = df.iloc[:min_len]
    X  = X[:min_len]
    print(f"  Aligned to       : {min_len} rows")

# Run Agglomerative Clustering (K=411, Ward)

print(f"\nSTEP 2: Running Agglomerative Clustering (K={K_TARGET}, Ward)...")
agg = AgglomerativeClustering(n_clusters=K_TARGET, linkage='ward')
df['Cluster_ID'] = agg.fit_predict(X)
print(f"  Unique clusters  : {df['Cluster_ID'].nunique()}")
print(f"  Coverage         : 100% (Agglomerative assigns all points)")

# Internal Metrics

print("\nSTEP 3: Computing Internal Metrics...")

# Sample 10k for speed on Silhouette
if len(X) > 10000:
    idx      = np.random.RandomState(42).choice(len(X), 10000, replace=False)
    X_sample = X[idx]
    l_sample = df['Cluster_ID'].values[idx]
else:
    X_sample = X
    l_sample = df['Cluster_ID'].values

sil = silhouette_score(X_sample, l_sample)
dbi = davies_bouldin_score(X_sample, l_sample)
chi = calinski_harabasz_score(X_sample, l_sample)

print(f"  Silhouette Score        : {sil:.4f}  (higher better, max 1.0)")
print(f"  Davies-Bouldin Index    : {dbi:.4f}  (lower better)")
print(f"  Calinski-Harabasz Index : {chi:.2f}  (higher better)")

# Cluster Purity
print("\n  Computing Cluster Purity...")
purities = []
for cid in df['Cluster_ID'].unique():
    subset      = df[df['Cluster_ID'] == cid]['pathlist_names']
    dominant    = subset.value_counts().iloc[0]
    purity      = dominant / len(subset)
    purities.append(purity)

mean_purity = np.mean(purities)
print(f"  Mean Cluster Purity     : {mean_purity:.4f}  (higher better, max 1.0)")

# External Metrics (vs Icecat ground truth)

print("\nSTEP 4: Computing External Metrics vs Icecat (pathlist_names)...")

y_true = df['pathlist_names'].fillna('Unknown').astype(str).values
y_pred = df['Cluster_ID'].values

ari = adjusted_rand_score(y_true, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred)
hom = homogeneity_score(y_true, y_pred)
com = completeness_score(y_true, y_pred)
vms = v_measure_score(y_true, y_pred)

print(f"  ARI (Adjusted Rand Index)    : {ari:.4f}  (0=random, 1=perfect)")
print(f"  NMI (Normalized Mutual Info) : {nmi:.4f}  (0=none,   1=perfect)")
print(f"  Homogeneity                  : {hom:.4f}  (clusters contain 1 class?)")
print(f"  Completeness                 : {com:.4f}  (class in 1 cluster?)")
print(f"  V-Measure                    : {vms:.4f}  (harmonic mean of H and C)")

# Coverage Logic

print("\nSTEP 5: Coverage Check...")
unassigned  = (df['Cluster_ID'] == -1).sum()
coverage    = ((len(df) - unassigned) / len(df)) * 100
print(f"  Total products   : {len(df)}")
print(f"  Unassigned (-1)  : {unassigned}")
print(f"  Coverage         : {coverage:.2f}%")

# Naming Quality Assessment

print("\nSTEP 6: Loading Naming Evaluation Scores...")
try:
    naming_df      = pd.read_csv(NAMING_CSV)
    mean_coherence = naming_df['Coherence_Score'].mean()

    exc_pct  = (naming_df['Quality'] == 'Excellent (≥0.60)').mean() * 100
    good_pct = (naming_df['Quality'] == 'Good (0.45-0.60)').mean()  * 100
    acc_pct  = (naming_df['Quality'] == 'Acceptable (0.30-0.45)').mean() * 100
    low_pct  = (naming_df['Quality'] == 'Low (<0.30)').mean()     * 100

    print(f"  Mean Coherence Score : {mean_coherence:.4f}")
    print(f"  Excellent Quality    : {exc_pct:.1f}%")
    print(f"  Good Quality         : {good_pct:.1f}%")
    print(f"  Acceptable Quality   : {acc_pct:.1f}%")
    print(f"  Low Quality          : {low_pct:.1f}%")
except FileNotFoundError:
    print(f"  WARNING: Could not find {NAMING_CSV}. Skipping naming evaluation.")
    mean_coherence = "N/A"

# Save Result

results_df = pd.DataFrame({
    'Metric': [
        'Silhouette Score',
        'Davies-Bouldin Index',
        'Calinski-Harabasz Index',
        'Mean Cluster Purity',
        'Adjusted Rand Index (ARI)',
        'Normalized Mutual Information (NMI)',
        'Homogeneity',
        'Completeness',
        'V-Measure',
        'Coverage (%)',
        'Mean Naming Coherence'
    ],
    'Value': [
        sil, dbi, chi, mean_purity,
        ari, nmi, hom, com, vms,
        coverage, mean_coherence
    ],
    'Type': [
        'Internal', 'Internal', 'Internal', 'Internal',
        'External', 'External', 'External', 'External', 'External',
        'Coverage',
        'Naming'
    ],
    'Interpretation': [
        'Higher is better (max 1.0)',
        'Lower is better (min 0.0)',
        'Higher is better',
        'Higher is better (max 1.0)',
        'Higher is better (max 1.0)',
        'Higher is better (max 1.0)',
        'Higher is better (max 1.0)',
        'Higher is better (max 1.0)',
        'Higher is better (max 1.0)',
        '100% = full product coverage',
        'Higher is better (max 1.0)'
    ]
})

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
results_df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "═" * 60)
print("  FINAL EVALUATION RESULTS")
print("═" * 60)
print(results_df.to_string(index=False))
print("═" * 60)
print(f"\nSaved to: {OUTPUT_CSV}")  