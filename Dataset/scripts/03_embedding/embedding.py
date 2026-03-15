import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans

# Load Data
file_path = r'C:\Users\goura\Documents\Master Thesis\Dataset\data\icecat_hierarchically_balanced.csv'
print("Loading balanced data...")
df = pd.read_csv(file_path)

# Prepare Text
texts = df['Title'].fillna('') + " " + df['Description.LongDesc'].fillna('')
# Sampling 5,000 items for the benchmark
subset_idx = np.random.choice(len(texts), 5000, replace=False)
texts_sub = texts.iloc[subset_idx].tolist()

results = []

def evaluate_model(name, embeddings):
    # Clustering with K=20 (Approximation of Level 2 categories)
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    
    sil = silhouette_score(embeddings, clusters, metric='cosine')
    cal = calinski_harabasz_score(embeddings, clusters)
    
    return {"Model": name, "Silhouette (Clarity)": sil, "CH Score (Density)": cal}

print("\nEmbedding Model Benchmark: Baseline vs. SOTA")

# TF-IDF Baseline
print("Evaluating TF-IDF...")
start = time.time()
tfidf = TfidfVectorizer(max_features=1000, stop_words='english').fit_transform(texts_sub).toarray()
time_tfidf = time.time() - start
res_tfidf = evaluate_model("TF-IDF", tfidf)
res_tfidf['Time (s)'] = time_tfidf
results.append(res_tfidf)

# MiniLM-L6
print("Evaluating MiniLM-L6...")
start = time.time()
model_mini = SentenceTransformer('all-MiniLM-L6-v2')
emb_mini = model_mini.encode(texts_sub,  show_progress_bar=True)
time_mini = time.time() - start
res_mini = evaluate_model("MiniLM (Fast)", emb_mini)
res_mini['Time (s)'] = time_mini
results.append(res_mini)

# BGE-Base
print("Evaluating BGE-Base-v1.5...")
start = time.time()
# BGE-Base is the current "best in class" for standard size models
model_bge = SentenceTransformer('BAAI/bge-base-en-v1.5')
# BGE performs best with a specific instruction usually, but for clustering raw product titles, direct encoding is fine.
emb_bge = model_bge.encode(texts_sub, normalize_embeddings=True, show_progress_bar=True)
time_bge = time.time() - start
res_bge = evaluate_model("BGE-Base (2024)", emb_bge)
res_bge['Time (s)'] = time_bge
results.append(res_bge)

print("\nBenchmark Results")
df_res = pd.DataFrame(results)
print(df_res)
