import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
file_path = r'C:\Users\goura\Documents\Master Thesis\Dataset\data\icecat_train_en_filtered.csv'
print(f"Loading data from: {file_path} ...")

# Read CSV file
try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
    print(f"Shape: {df.shape} (Rows, Columns)")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Analysis of column types and missing values
print("\n--- 1. COLUMN SUMMARY ---")
summary_data = []

for col in df.columns:
    # Calculate missing percentage
    missing_count = df[col].isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    
    # Get number of unique values
    n_unique = df[col].nunique()
    
    # Get a sample value (first non-null)
    sample_val = df[col].dropna().iloc[0] if n_unique > 0 else "ALL NULL"
    
    summary_data.append({
        'Column': col,
        'Type': str(df[col].dtype),
        'Missing (%)': round(missing_pct, 1),
        'Unique Values': n_unique,
        'Sample': str(sample_val)[:30]  # Truncate for display
    })

summary_df = pd.DataFrame(summary_data)
# Display columns with low missingness first
print(summary_df.sort_values('Missing (%)').to_string(index=False))

# Text field analysis (word counts)
print("\n--- 2. TEXT FIELD ANALYSIS ---")

# Heuristic: Treat object columns with >50 unique values as potential text fields
text_candidates = [c for c in df.columns if df[c].dtype == 'object' and df[c].nunique() > 50]

if not text_candidates:
    print("No obvious text columns found.")
else:
    print(f"Analyzing potential text columns: {text_candidates}")
    
    text_stats = []
    for col in text_candidates:
        # Convert to string, drop NaNs
        s = df[col].dropna().astype(str)
        
        # Calculate length in words (approximate)
        word_counts = s.apply(lambda x: len(x.split()))
        
        text_stats.append({
            'Column': col,
            'Avg Words': round(word_counts.mean(), 1),
            'Median Words': word_counts.median(),
            'Max Words': word_counts.max()
        })
    
    text_stats_df = pd.DataFrame(text_stats)
    print(text_stats_df.sort_values('Avg Words', ascending=False).to_string(index=False))

# Category distribution analysis
print("\n--- 3. CATEGORY DISTRIBUTION ---")

# Heuristic: Look for columns that might be categories (fewer unique values than rows)

potential_category_cols = [c for c in df.columns if df[c].nunique() < 1000 and df[c].nunique() > 1]

if not potential_category_cols:
    print("No obvious category columns found (based on unique count heuristic).")
else:
    for col in potential_category_cols:
        print(f"\nDistribution for: {col}")
        dist = df[col].value_counts().head(10) # Top 10 categories
        print(dist)
        
        # Check for imbalance
        top_1_pct = (dist.iloc[0] / len(df)) * 100
        print(f"-> Top category '{dist.index[0]}' covers {round(top_1_pct, 1)}% of rows.")

print("\n--- EDA COMPLETE ---")
