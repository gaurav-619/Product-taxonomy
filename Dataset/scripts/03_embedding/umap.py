import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt

# Vectorization & Dimensionality Reduction

# Load Dataset
file_path = r'C:\Users\goura\Documents\Master Thesis\Dataset\data\icecat_hierarchically_balanced.csv'
print(f"Loading full dataset from: {file_path}")
df = pd.read_csv(file_path)

# Prepare Text
# Combining Title + LongDesc gives the best signal
texts = df['Title'].fillna('') + " " + df['Description.LongDesc'].fillna('')
print(f"Processing {len(texts)} products...")

print("Encoding with MiniLM...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts.tolist(), show_progress_bar=True)

# Apply UMAP (Dimensionality Reduction)
print("Running UMAP...")

print("Reducing to 5D for Clustering...")
umap_model_5d = umap.UMAP(n_neighbors=15, 
                          n_components=5, 
                          min_dist=0.0, 
                          metric='cosine', 
                          random_state=42)
umap_embeddings_5d = umap_model_5d.fit_transform(embeddings)

print("Reducing to 2D for Visualization...")
umap_model_2d = umap.UMAP(n_neighbors=15, 
                          n_components=2, 
                          min_dist=0.0, 
                          metric='cosine', 
                          random_state=42)
umap_embeddings_2d = umap_model_2d.fit_transform(embeddings)

# Save Checkpoints
output_dir = r'C:\Users\goura\Documents\Master Thesis\Dataset\data\\'
print(f"Saving files to {output_dir}...")

np.save(output_dir + 'embeddings_minilm.npy', embeddings)
np.save(output_dir + 'umap_5d.npy', umap_embeddings_5d)
np.save(output_dir + 'umap_2d.npy', umap_embeddings_2d)

# Verify with a Plot
print("Verifying results...")
plt.figure(figsize=(10, 8))
plt.scatter(umap_embeddings_2d[:, 0], umap_embeddings_2d[:, 1], s=0.5, alpha=0.5, c='blue')
plt.title('The Product Galaxy (UMAP Projection)', fontsize=16)
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.savefig(output_dir + 'umap_projection.png')
print(f"Map saved to {output_dir}umap_projection.png")
# plt.show()


