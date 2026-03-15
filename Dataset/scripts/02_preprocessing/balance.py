import pandas as pd
import numpy as np

# Load Dataset
file_path = r'C:\Users\goura\Documents\Master Thesis\Dataset\data\icecat_train_en_filtered.csv'
df = pd.read_csv(file_path, low_memory=False)

# Basic cleaning and selection of relevant columns
keep_cols = ['IcecatId', 'GTIN', 'Brand', 'Title', 'Description.LongDesc', 
             'SummaryDescription.LongSummaryDescription', 'SummaryDescription.ShortSummaryDescription',
             'pathlist_names', 'leaf_category', 'Category.CategoryID']
df_work = df[keep_cols].copy()

print("="*70)
print("Hierarchical Balancing & Normalization")
print("="*70)

# Normalize to Level 3 hierarchy
print("\n--- 1. Normalizing to Level 3 ---")

def get_level_3_path(path):
    if not isinstance(path, str): return ""
    parts = path.split('>')
    # Take first 3 parts (Root > L2 > L3)
    if len(parts) >= 3:
        return '>'.join(parts[:3])
    return path 

df_work['path_L3'] = df_work['pathlist_names'].apply(get_level_3_path)

# Extract new labels
df_work['leaf_L3'] = df_work['path_L3'].apply(lambda x: x.split('>')[-1] if x else "UNKNOWN")
# Extract Level 2 parent for hierarchical sampling
df_work['parent_L2'] = df_work['path_L3'].apply(lambda x: x.split('>')[1] if len(x.split('>')) >= 2 else "UNKNOWN")

print(f"Original unique leaves: {df_work['leaf_category'].nunique()}")
print(f"Normalized Level 3 leaves: {df_work['leaf_L3'].nunique()}")


# Hierarchical Sampling Strategy
print("\n--- 2. Performing Hierarchical Balancing ---")

# Config
TARGET_TOTAL = 50000
NUM_L2_BRANCHES = df_work['parent_L2'].nunique() 
TARGET_PER_L2 = int(TARGET_TOTAL / NUM_L2_BRANCHES)

print(f"Target Total: {TARGET_TOTAL}")
print(f"Level 2 Branches: {NUM_L2_BRANCHES}")
print(f"Target per Branch: ~{TARGET_PER_L2}")

sampled_frames = []

# Iterate over each Level 2 branch (e.g., "Computers", "Software")
for l2_name, l2_group in df_work.groupby('parent_L2'):  
    
    # Get Level 3 children in this branch
    l3_subgroups = l2_group['leaf_L3'].unique()
    num_l3 = len(l3_subgroups)

    if num_l3 == 0: continue
        
    # Calculate target per leaf (split the branch budget evenly)
    target_per_l3 = int(TARGET_PER_L2 / num_l3)
    
    # Enforce reasonable bounds (don't sample 1 item, don't force 1000 if data is rare)
    # We want at least 10 items if available to have a valid cluster
    min_required = 10 
    
    # print(f"  Branch '{l2_name}': {len(l2_group)} rows, {num_l3} leaves. Target ~{target_per_l3}/leaf")

    for l3_name, l3_group in l2_group.groupby('leaf_L3'):
        # Cap at available data
        n_available = len(l3_group)
        
        # We take the target, but if target is too small (e.g. branch has 100 children), 
        # ensure we take at least min_required (if available)
        n_sample = max(target_per_l3, min_required)
        n_sample = min(n_sample, n_available)
        
        # Perform sampling
        if n_sample > 0:
            s = l3_group.sample(n=n_sample, random_state=42)
            sampled_frames.append(s)

# Combine
df_balanced = pd.concat(sampled_frames)

print(f"\n--- Balancing Complete ---")
print(f"Original Rows: {len(df_work)}")
print(f"Balanced Rows: {len(df_balanced)}")

# 4. Final Sanity Check
print("\nTop 5 Level 2 Categories (Should be roughly balanced):")
print(df_balanced['parent_L2'].value_counts().head(5))

print("\nTop 5 Level 3 Categories (Should not be huge spikes):")
print(df_balanced['leaf_L3'].value_counts().head(5))

# Save output
save_path = r'C:\Users\goura\Documents\Master Thesis\Dataset\data\icecat_hierarchically_balanced.csv'
df_balanced.to_csv(save_path, index=False)
print(f"\n✓ Saved balanced dataset to: {save_path}")
