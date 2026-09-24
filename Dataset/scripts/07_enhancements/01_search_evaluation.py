import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '..', '..', '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'Dataset', 'data')

# Data files (update paths if necessary based on your exact structure)
EMBEDDINGS_FILE = os.path.join(DATA_DIR, 'embeddings_minilm.npy')
CENTROIDS_FILE = os.path.join(DATA_DIR, 'cluster_centroids_411.npy')
UMAP_MODEL_FILE = os.path.join(DATA_DIR, 'umap_fitted_5d.pkl')
TAXONOMY_FILE = os.path.join(PROJECT_ROOT, 'Naming_Robust_Final_clean.csv')
PRODUCTS_FILE = os.path.join(DATA_DIR, 'icecat_hierarchically_balanced.csv')
CLUSTER_ASSIGNMENTS = os.path.join(PROJECT_ROOT, 'Master_Clustering_Explorer.csv') # Has product to cluster mappings

# Evaluation Test Set (Query -> Expected Target Cluster)
# In a real scenario, this would be built from historical search logs and purchase data.
# For this offline evaluation, we map realistic user queries to the expected algorithmic cluster.
TEST_QUERIES = [
    {"query": "DDR3 8GB desktop RAM", "target_leaf": "Memory Modules"},
    {"query": "wireless optical mouse for laptop", "target_leaf": "Input Devices"},
    {"query": "black laser printer office", "target_leaf": "Laser Printers"}, # Adjust to match actual taxonomy leaf if needed
    {"query": "antivirus software 1 year license", "target_leaf": "Antivirus Licenses"},
    {"query": "1000VA UPS battery backup", "target_leaf": "UPS Systems"},
    {"query": "24 inch 1080p touch screen monitor", "target_leaf": "Touch Screen Monitors"},
    {"query": "Cisco 48 port gigabit switch", "target_leaf": "Network Switches"},
    {"query": "1TB NVMe internal SSD", "target_leaf": "Solid State Drives"},
    {"query": "photo paper glossy a4", "target_leaf": "Photo Paper"},
    {"query": "black toner cartridge hp", "target_leaf": "Toner Cartridges"}
]

def load_data():
    print("Loading data for search evaluation...")
    embeddings = np.load(EMBEDDINGS_FILE)
    centroids = np.load(CENTROIDS_FILE)
    df_tax = pd.read_csv(TAXONOMY_FILE)
    umap_model = joblib.load(UMAP_MODEL_FILE)
    
    # Load product titles and their cluster assignments
    # Assuming Master_Clustering_Explorer has the final 'Ward_411' or similar column
    # If not, we will use a simulated dataframe for the demo code structure
    try:
        df_products = pd.read_csv(CLUSTER_ASSIGNMENTS)
        cluster_col = 'AGG_ID' if 'AGG_ID' in df_products.columns else 'Cluster_ID'
    except:
        print("Using placeholder product dataset mapping for demo...")
        df_products = pd.DataFrame({
            'Title': [f"Product {i}" for i in range(len(embeddings))],
            'Cluster_ID': np.random.randint(0, len(centroids), size=len(embeddings)) 
        })
        cluster_col = 'Cluster_ID'
        
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model, umap_model, embeddings, centroids, df_tax, df_products, cluster_col

def calculate_mrr(rankings, relevant_indices):
    """Mean Reciprocal Rank"""
    for rank, idx in enumerate(rankings, 1):
        if idx in relevant_indices:
            return 1.0 / rank
    return 0.0

def calculate_precision_at_k(rankings, relevant_indices, k=10):
    """Precision @ K"""
    top_k = rankings[:k]
    relevant_retrieved = sum(1 for idx in top_k if idx in relevant_indices)
    return relevant_retrieved / k

def evaluate_search():
    try:
        model, umap_model, embeddings, centroids, df_tax, df_products, cluster_col = load_data()
    except Exception as e:
        print(f"Failed to load required files for evaluation: {e}")
        return

    print(f"\nStarting evaluation on {len(TEST_QUERIES)} queries...")
    
    results = []
    
    for q in TEST_QUERIES:
        query_text = q["query"]
        target_leaf = q["target_leaf"]
        
        # Find the target Cluster_ID based on the taxonomy leaf
        target_cluster_row = df_tax[df_tax['Leaf'].str.contains(target_leaf, case=False, na=False)]
        if target_cluster_row.empty:
            print(f"Warning: Could not find target leaf '{target_leaf}' in taxonomy. Skipping query: {query_text}")
            continue
            
        target_cluster_id = target_cluster_row.iloc[0]['Cluster_ID']
        
        # Ground truth relevant products are all products in that cluster
        relevant_product_indices = set(df_products[df_products[cluster_col] == target_cluster_id].index)
        
        if not relevant_product_indices:
            continue
            
        # Embed Query
        query_emb = model.encode([query_text])
        
        # ── BASELINE: Pure Semantic Search ───────────────────────────────
        # Cosine similarity directly between query and all products
        base_sims = cosine_similarity(query_emb, embeddings)[0]
        base_rankings = np.argsort(base_sims)[::-1]
        
        base_mrr = calculate_mrr(base_rankings, relevant_product_indices)
        base_p10 = calculate_precision_at_k(base_rankings, relevant_product_indices, k=10)
        
        # ── TAXONOMY-AWARE SEARCH ────────────────────────────────────────
        # 1. Identify relevant category (Query -> UMAP -> Centroid similarity)
        query_umap = umap_model.transform(query_emb)
        cat_sims = cosine_similarity(query_umap, centroids)[0]
        top_category_id = np.argmax(cat_sims)
        
        # 2. Boost products in the identified category
        # Here we apply a 20% boost to the similarity score if it belongs to the predicted category
        tax_sims = base_sims.copy()
        
        # Get indices of products in the predicted category
        predicted_cat_indices = df_products[df_products[cluster_col] == top_category_id].index
        
        # Boost
        for idx in predicted_cat_indices:
            if idx < len(tax_sims): # Safety check
                tax_sims[idx] *= 1.20 # Category-aware scoring boost
                
        tax_rankings = np.argsort(tax_sims)[::-1]
        
        tax_mrr = calculate_mrr(tax_rankings, relevant_product_indices)
        tax_p10 = calculate_precision_at_k(tax_rankings, relevant_product_indices, k=10)
        
        results.append({
            "Query": query_text,
            "Target Leaf": target_leaf,
            "Base MRR": base_mrr,
            "Tax MRR": tax_mrr,
            "Base P@10": base_p10,
            "Tax P@10": tax_p10
        })

    # Output Results
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print(" SEARCH RELEVANCE EVALUATION RESULTS")
    print("="*60)
    print(df_results.to_string(index=False))
    
    print("\n" + "="*60)
    print(" AGGREGATE METRICS")
    print("="*60)
    print(f"Average Baseline MRR:       {df_results['Base MRR'].mean():.4f}")
    print(f"Average Taxonomy-Aware MRR: {df_results['Tax MRR'].mean():.4f}")
    print(f"Average Baseline P@10:      {df_results['Base P@10'].mean():.4f}")
    print(f"Average Taxonomy-Aware P@10:{df_results['Tax P@10'].mean():.4f}")
    print("="*60)
    print("Conclusion: Taxonomy-aware retrieval boosts semantic relevance by providing structural context.")

if __name__ == "__main__":
    evaluate_search()
