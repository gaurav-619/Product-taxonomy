# Listing 5.7: Terminal-Style Text-Based Taxonomy Generation
import pandas as pd
import os

# --- 1. CONFIGURATION ---
INPUT_FILE = r'C:\Users\goura\Documents\Master Thesis\Naming_Robust_Final_clean1.csv'

if not os.path.exists(INPUT_FILE):
    INPUT_FILE = r'C:\Users\goura\Documents\Master Thesis\Naming_Robust_Final_clean.csv'

df_final = pd.read_csv(INPUT_FILE)

# Standardize columns
df_final = df_final[['Root', 'Parent', 'Leaf']].rename(columns={
    'Root': 'Level_1',
    'Parent': 'Level_2',
    'Leaf': 'Level_3'
})

# --- 2. AUTONOMOUS DOMAIN SELECTION ---
# Find the top 2 domains to keep the printout concise for the thesis page
top_domains = df_final['Level_1'].value_counts().nlargest(2).index.tolist()

# --- 3. GENERATE TERMINAL TREE ---

for i, l1 in enumerate(top_domains):
    l1_subset = df_final[df_final['Level_1'] == l1] 
    # Get unique Level 2 categories, limited to 3 for brevity
    l2_groups = l1_subset['Level_2'].unique()[:3] 
    
    is_last_l1 = (i == len(top_domains) - 1)
    l1_branch = "└── " if is_last_l1 else "├── "
    l1_indent = "    " if is_last_l1 else "│   "
    
    print(f"{l1_branch}{l1}")
    
    for j, l2 in enumerate(l2_groups):
        # Get unique Level 3 leaves for this L2, limited to 3
        l3_leaves = l1_subset[l1_subset['Level_2'] == l2]['Level_3'].unique()[:3]
        
        is_last_l2 = (j == len(l2_groups) - 1)
        l2_branch = "└── " if is_last_l2 else "├── "
        l2_indent = "    " if is_last_l2 else "│   "
        
        print(f"{l1_indent}{l2_branch}{l2}")
        
        for k, l3 in enumerate(l3_leaves):
            is_last_l3 = (k == len(l3_leaves) - 1)
            l3_branch = "└── " if is_last_l3 else "├── "
            print(f"{l1_indent}{l2_indent}{l3_branch}{l3}")