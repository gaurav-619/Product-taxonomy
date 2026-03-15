import hdbscan
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt

# HDBSCAN Method Comparison (EOM vs Leaf) and DBCV Metric Assessment
def compare_hdbscan_methods(embeddings, min_cluster_size=50):
    methods = ['eom', 'leaf']
    results = {}
    
    print(f"--- Running Comparison with min_cluster_size={min_cluster_size} ---")
    
    for method in methods:
        # Initialize and fit
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=None, # Defaults to min_cluster_size if None
            cluster_selection_method=method,
            gen_min_span_tree=True # Required for validity scores
        )
        labels = clusterer.fit_predict(embeddings)
        
        # Get Validation Score (DBCV)
        # Note: relative_validity_ ranges from -1 to 1 (higher is better)
        dbcv_score = clusterer.relative_validity_
        
        # Basic Stats
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        percent_noise = (n_noise / len(labels)) * 100
        
        results[method] = {
            'model': clusterer,
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'percent_noise': percent_noise,
            'DBCV_score': dbcv_score
        }
        
        print(f"Method: {method.upper()}")
        print(f"  Clusters: {n_clusters}")
        print(f"  Noise: {n_noise} ({percent_noise:.1f}%)")
        print(f"  DBCV Score: {dbcv_score:.4f}")
        print("-" * 30)
        
    return results

# Usage:
# comparison_results = compare_hdbscan_methods(embeddings_reduced)

# Probabilistic Analysis (Soft Clustering)
def analyze_soft_clusters(model, embeddings):
    # Generate soft clusters (matrix of shape [n_samples, n_clusters])
    soft_clusters = hdbscan.all_points_membership_vectors(model)
    
    # Get the highest probability for each point
    max_probs = np.max(soft_clusters, axis=1)
    
    # Identify "Strong Noise" vs "Weak Noise"
    # Points labeled -1 but with high membership probability to a cluster
    hard_labels = model.labels_
    noise_indices = np.where(hard_labels == -1)[0]
    
    # Check if any noise points actually have >50% probability of belonging somewhere
    hidden_gems = soft_clusters[noise_indices].max(axis=1) > 0.5
    count_hidden = np.sum(hidden_gems)
    
    print(f"Soft Clustering Analysis:")
    print(f"  Noise points that strongly belong to a cluster (>50% prob): {count_hidden}")
    
    return soft_clusters

# Usage:
# soft_matrix = analyze_soft_clusters(comparison_results['eom']['model'], embeddings_reduced)

# Visual Stability Check with UMAP 2D Projection
def visualize_stability(embeddings, labels, title="Cluster Stability"):
    # Reduce to 2D for visualization (if not already 2D)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    # Plot noise as gray/black, clusters as colors
    clustered = (labels >= 0)
    plt.scatter(embedding_2d[~clustered, 0], embedding_2d[~clustered, 1], 
                c=(0.5, 0.5, 0.5), s=0.5, alpha=0.3, label='Noise')
    plt.scatter(embedding_2d[clustered, 0], embedding_2d[clustered, 1], 
                c=labels[clustered], s=0.5, cmap='Spectral', alpha=0.6)
    plt.title(title)
    plt.show()

# Usage:
# visualize_stability(embeddings_reduced, comparison_results['eom']['labels'])
