import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score
import warnings
warnings.filterwarnings('ignore')

INPUT_FILE  = 'Naming_Robust_Final_clean.csv'
OUTPUT_FILE = 'Naming_Robust_Final_EVALUATED.csv'

print("Loading model and data...")
df = pd.read_csv(INPUT_FILE)
model = SentenceTransformer('all-MiniLM-L6-v2')
total = len(df)
print(f"Loaded {total} clusters.")

# ─── Helper: flatten list to single string ────────────────────────────────────
def flatten(text):
    return str(text).replace('||', ' ').replace('  ', ' ').strip()

# ─── Prepare reference texts ──────────────────────────────────────────────────
labels   = df['Fixed_Label'].tolist()
keywords = [flatten(k) for k in df['Keywords'].tolist()]
examples = [flatten(e) for e in df['Examples'].tolist()]
stage1   = df['Stage1_FewShot'].tolist()
stage2   = df['Stage2_Critic'].tolist()

# Semantic Coherence Score (cosine, all-MiniLM-L6-v2)
print("\nComputing semantic coherence scores...")
label_embs   = model.encode(labels,   convert_to_tensor=True, show_progress_bar=True)
keyword_embs = model.encode(keywords, convert_to_tensor=True, show_progress_bar=True)
example_embs = model.encode(examples, convert_to_tensor=True, show_progress_bar=True)
stage1_embs  = model.encode(stage1,   convert_to_tensor=True, show_progress_bar=True)
stage2_embs  = model.encode(stage2,   convert_to_tensor=True, show_progress_bar=True)

cosine_label_kw  = [util.cos_sim(label_embs[i], keyword_embs[i]).item() for i in range(total)]
cosine_label_ex  = [util.cos_sim(label_embs[i], example_embs[i]).item() for i in range(total)]
cosine_s1_s2     = [util.cos_sim(stage1_embs[i], stage2_embs[i]).item() for i in range(total)]

df['Cosine_Label_vs_Keywords'] = cosine_label_kw
df['Cosine_Label_vs_Examples'] = cosine_label_ex
df['Cosine_Stage1_vs_Stage2']  = cosine_s1_s2

print(f"  Cosine Label vs Keywords : {np.mean(cosine_label_kw):.4f}")
print(f"  Cosine Label vs Examples : {np.mean(cosine_label_ex):.4f}")
print(f"  Cosine Stage1 vs Stage2  : {np.mean(cosine_s1_s2):.4f}")

# BERTScore (roberta-large)
print("\nComputing BERTScore: Label vs Keywords...")
P1, R1, F1_kw = bert_score(labels, keywords, lang='en', model_type='roberta-large', verbose=True)
df['BERTScore_P_vs_Keywords'] = P1.tolist()
df['BERTScore_R_vs_Keywords'] = R1.tolist()
df['BERTScore_F1_vs_Keywords'] = F1_kw.tolist()

print(f"\n  BERTScore P  vs Keywords : {P1.mean():.4f}")
print(f"  BERTScore R  vs Keywords : {R1.mean():.4f}")
print(f"  BERTScore F1 vs Keywords : {F1_kw.mean():.4f}")

print("\nComputing BERTScore: Label vs Examples...")
P2, R2, F1_ex = bert_score(labels, examples, lang='en', model_type='roberta-large', verbose=True)
df['BERTScore_P_vs_Examples'] = P2.tolist()
df['BERTScore_R_vs_Examples'] = R2.tolist()
df['BERTScore_F1_vs_Examples'] = F1_ex.tolist()

print(f"\n  BERTScore P  vs Examples : {P2.mean():.4f}")
print(f"  BERTScore R  vs Examples : {R2.mean():.4f}")
print(f"  BERTScore F1 vs Examples : {F1_ex.mean():.4f}")

print("\nComputing BERTScore: Stage1 vs Stage2 (Critic Stability)...")
P3, R3, F1_s = bert_score(stage1, stage2, lang='en', model_type='roberta-large', verbose=True)
df['BERTScore_F1_Stage1_vs_Stage2'] = F1_s.tolist()

print(f"\n  BERTScore F1 Stage1 vs Stage2 : {F1_s.mean():.4f}")

# Per-Quality-Tier Breakdown
print("\n--- BERTScore F1 by Quality Tier ---")
for tier in ['Excellent (≥0.60)', 'Good (0.45-0.60)', 'Acceptable (0.30-0.45)', 'Low (<0.30)']:
    mask = df['Quality'] == tier
    if mask.sum() > 0:
        avg = df.loc[mask, 'BERTScore_F1_vs_Keywords'].mean()
        print(f"  {tier:30s}: {avg:.4f}  (n={mask.sum()})")

# Final Summary
print("\n" + "="*60)
print("FINAL EVALUATION SUMMARY")
print("="*60)
print(f"Total clusters evaluated     : {total}")
print(f"\nSemantic Coherence (cosine):")
print(f"  Label vs Keywords          : {np.mean(cosine_label_kw):.4f}")
print(f"  Label vs Examples          : {np.mean(cosine_label_ex):.4f}")
print(f"  Stage1 vs Stage2 stability : {np.mean(cosine_s1_s2):.4f}")
print(f"\nBERTScore (roberta-large):")
print(f"  F1 Label vs Keywords       : {F1_kw.mean():.4f}")
print(f"  F1 Label vs Examples       : {F1_ex.mean():.4f}")
print(f"  F1 Stage1 vs Stage2        : {F1_s.mean():.4f}")
print(f"\nQuality Distribution:")
for tier, count in df['Quality'].value_counts().items():
    print(f"  {tier:30s}: {count:4d}  ({count/total*100:.1f}%)")

# Save Result
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Saved to {OUTPUT_FILE}")
