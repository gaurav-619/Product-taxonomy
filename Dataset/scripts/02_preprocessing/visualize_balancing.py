import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
RAW_CSV_PATH      = r"C:\Users\goura\Documents\Master Thesis\Dataset\data\icecat_train_en_filtered.csv"
BALANCED_CSV_PATH = r"C:\Users\goura\Documents\Master Thesis\Dataset\data\icecat_hierarchically_balanced.csv"
OUTPUT_PATH       = r"C:\Users\goura\Documents\Master Thesis\Dataset\data\balancing_visualization.png"

# Gini Function
def gini(counts):
    counts = np.sort(np.array(counts, dtype=float))
    n = len(counts)
    if n == 0 or np.sum(counts) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * counts)) / (n * np.sum(counts)) - (n + 1) / n

# Level Extraction Utility
def extract_levels(df, use_parent_L2=False):
    """
    L1 : pathlist_names split[0]
    L2 : parent_L2 column (if use_parent_L2=True) else pathlist_names split[1]
    L3 : leaf_category (authoritative — handles paths deeper than 3 levels)
    """
    split = df['pathlist_names'].astype(str).str.split('>', expand=True)
    df = df.copy()
    df['L1'] = split[0].str.strip() if 0 in split.columns else "UNKNOWN"
    if use_parent_L2 and 'parent_L2' in df.columns:
        df['L2'] = df['parent_L2']
    else:
        df['L2'] = split[1].str.strip() if 1 in split.columns else "UNKNOWN"
    df['L3'] = df['leaf_category'] if 'leaf_category' in df.columns else "UNKNOWN"
    return df

# Loading and Extraction
print("Loading datasets...")
df_raw = extract_levels(pd.read_csv(RAW_CSV_PATH, low_memory=False), use_parent_L2=False)
df_bal = extract_levels(pd.read_csv(BALANCED_CSV_PATH, low_memory=False), use_parent_L2=True)

# Pre-compute distributions used across multiple plots
l2_raw_counts   = df_raw['L2'].value_counts()
l2_bal_counts   = df_bal['L2'].value_counts()
l3_raw_dist     = df_raw['L3'].value_counts().values
l3_bal_dist     = df_bal['L3'].value_counts().values
gini_raw        = gini(l3_raw_dist)
gini_bal        = gini(l3_bal_dist)

print("Creating visualizations...")
sns.set_style("whitegrid")
fig = plt.figure(figsize=(20, 12))

# Level 2 Comparison Plot
ax1 = fig.add_subplot(2, 3, 1)

top_l2 = l2_raw_counts.head(15).index
x      = np.arange(len(top_l2))
width  = 0.35

bars1 = ax1.bar(x - width/2, l2_raw_counts[top_l2].values,
                width, label='Raw', alpha=0.8, color='#e74c3c')
bars2 = ax1.bar(x + width/2,
                [l2_bal_counts.get(c, 0) for c in top_l2],
                width, label='Balanced', alpha=0.8, color='#27ae60')

ax1.set_xlabel('Level 2 Categories', fontweight='bold')
ax1.set_ylabel('Number of Products', fontweight='bold')
ax1.set_title('Level 2 Distribution Before vs After',
              fontweight='bold', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(top_l2, rotation=45, ha='right', fontsize=8)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, h,
                     f'{int(h):,}', ha='center', va='bottom', fontsize=7)

# Level 3 Comparison Plot
ax2 = fig.add_subplot(2, 3, 2)

l3_raw_vc  = df_raw['L3'].value_counts()
l3_bal_vc  = df_bal['L3'].value_counts()
top_l3     = l3_raw_vc.head(20).index
x          = np.arange(len(top_l3))

bars1 = ax2.bar(x - width/2, l3_raw_vc[top_l3].values,
                width, label='Raw', alpha=0.8, color='#e74c3c')
bars2 = ax2.bar(x + width/2,
                [l3_bal_vc.get(c, 0) for c in top_l3],
                width, label='Balanced', alpha=0.8, color='#27ae60')

ax2.set_xlabel('Level 3 Categories', fontweight='bold')
ax2.set_ylabel('Number of Products', fontweight='bold')
ax2.set_title('Level 3 Distribution Before vs After',
              fontweight='bold', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(top_l3, rotation=45, ha='right', fontsize=7)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Histogram of Samples per Category
ax3 = fig.add_subplot(2, 3, 3)

ax3.hist(l3_raw_dist, bins=50, alpha=0.6, label='Raw',      color='#e74c3c', edgecolor='black')
ax3.hist(l3_bal_dist, bins=30, alpha=0.6, label='Balanced', color='#27ae60', edgecolor='black')

mean_raw = np.mean(l3_raw_dist)
mean_bal = np.mean(l3_bal_dist)
ax3.axvline(mean_raw, color='#e74c3c', linestyle='--', linewidth=2,
            label=f'Raw Mean: {mean_raw:.0f}')
ax3.axvline(mean_bal, color='#27ae60', linestyle='--', linewidth=2,
            label=f'Balanced Mean: {mean_bal:.0f}')

ax3.set_xlabel('Samples per Category', fontweight='bold')
ax3.set_ylabel('Frequency (log scale)', fontweight='bold')
ax3.set_title('Distribution of Samples per Category',
              fontweight='bold', fontsize=12)
ax3.set_yscale('log')
ax3.legend(fontsize=8)
ax3.grid(axis='y', alpha=0.3)

# Box Plot Comparison
ax4 = fig.add_subplot(2, 3, 4)

bp = ax4.boxplot([l3_raw_dist, l3_bal_dist],
                 labels=['Raw', 'Balanced'],
                 patch_artist=True, showfliers=False)

for patch, color in zip(bp['boxes'], ['#e74c3c', '#27ae60']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

stats_text = (
    f"Raw:\n"
    f"  Mean:   {np.mean(l3_raw_dist):.0f}\n"
    f"  Median: {np.median(l3_raw_dist):.0f}\n"
    f"  Std:    {np.std(l3_raw_dist):.0f}\n\n"
    f"Balanced:\n"
    f"  Mean:   {np.mean(l3_bal_dist):.0f}\n"
    f"  Median: {np.median(l3_bal_dist):.0f}\n"
    f"  Std:    {np.std(l3_bal_dist):.0f}"
)
ax4.text(1.55, ax4.get_ylim()[1] * 0.5, stats_text, fontsize=9,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax4.set_ylabel('Samples per Category', fontweight='bold')
ax4.set_title('Distribution Statistics', fontweight='bold', fontsize=12)
ax4.grid(axis='y', alpha=0.3)

# Gini Coefficient Comparison
ax5 = fig.add_subplot(2, 3, 5)

bars = ax5.bar(['Raw', 'Balanced'], [gini_raw, gini_bal],
               color=['#e74c3c', '#27ae60'], alpha=0.8,
               edgecolor='black', linewidth=2)

ax5.axhline(y=0.5, color='orange', linestyle='--', linewidth=2,
            label='Moderate Imbalance (0.5)')
ax5.set_ylim(0, 1)
ax5.set_ylabel('Gini Coefficient', fontweight='bold')
ax5.set_title('Gini Coefficient Comparison',
              fontweight='bold', fontsize=12)
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

for bar in bars:
    h = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width() / 2, h,
             f'{h:.4f}', ha='center', va='bottom',
             fontsize=12, fontweight='bold')

# Summary Statistics Table
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

summary = [
    ['Metric',                  'Raw',                         'Balanced',                      'Change'],
    ['Total Products',          f'{len(df_raw):,}',            f'{len(df_bal):,}',               f'{len(df_bal)/len(df_raw)*100:.1f}%'],
    ['L2 Categories',           f'{df_raw["L2"].nunique()}',   f'{df_bal["L2"].nunique()}',       '—'],
    ['L3 Categories',           f'{df_raw["L3"].nunique()}',   f'{df_bal["L3"].nunique()}',       '—'],
    ['L2 Imbalance Ratio',      '404:1',                       '5:1',                            '↓ 92%'],
    ['L3 Imbalance Ratio',      '3,715:1',                     '677:1',                          '↓ 82%'],
    ['Max Samples/L3',          f'{l3_raw_dist.max():,}',      f'{l3_bal_dist.max():,}',         f'{l3_bal_dist.max()/l3_raw_dist.max()*100:.1f}%'],
    ['Min Samples/L3',          f'{l3_raw_dist.min():,}',      f'{l3_bal_dist.min():,}',         '—'],
    ['Mean Samples/L3',         f'{np.mean(l3_raw_dist):.0f}', f'{np.mean(l3_bal_dist):.0f}',   f'{np.mean(l3_bal_dist)/np.mean(l3_raw_dist)*100:.1f}%'],
    ['Std Dev',                 f'{np.std(l3_raw_dist):.0f}',  f'{np.std(l3_bal_dist):.0f}',    f'{np.std(l3_bal_dist)/np.std(l3_raw_dist)*100:.1f}%'],
    ['Gini Coefficient (L3)',   f'{gini_raw:.4f}',             f'{gini_bal:.4f}',                f'{(gini_bal-gini_raw)/gini_raw*100:.1f}%'],
]

table = ax6.table(cellText=summary, cellLoc='center', loc='center',
                  colWidths=[0.35, 0.22, 0.22, 0.21])
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.9)

for j in range(4):
    table[(0, j)].set_facecolor('#3498db')
    table[(0, j)].set_text_props(weight='bold', color='white')

for i in range(1, len(summary)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')

ax6.set_title('Summary Statistics', fontweight='bold', fontsize=12, pad=20)

# Save output
fig.suptitle('Hierarchical Balancing Analysis: Before vs After',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved to: {OUTPUT_PATH}")
plt.show()

# Console Summary Summary
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print(f"  Products     : {len(df_raw):,} → {len(df_bal):,} ({len(df_bal)/len(df_raw)*100:.1f}% retained)")
print(f"  L2 Imbalance : 404:1 → 5:1   (Gini: 0.5903 → 0.1602)")
print(f"  L3 Imbalance : 3,715:1 → 677:1 (Gini: {gini_raw:.4f} → {gini_bal:.4f})")
print(f"  Std Dev      : {np.std(l3_raw_dist):.0f} → {np.std(l3_bal_dist):.0f}")
