import ollama
import pandas as pd
import numpy as np
import re
import time
import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
MODEL_NAME          = "qwen2.5:7b"
INPUT_FILE          = r'C:\Users\goura\Documents\Master Thesis\Cluster_Profiles_for_LLM.csv'
OUTPUT_FILE         = 'Naming_Robust_Finalcsv'
EMBED_MODEL         = 'all-MiniLM-L6-v2'
MAX_RETRIES         = 2
MAX_ROOTS_IN_PROMPT = 5

# ── CHANGE THIS ONE LINE TO SWITCH BETWEEN TEST AND FULL ──────────
SAMPLE_SIZE = None        # ← 20 for quick test | None for full 411
# ─────────────────────────────────────────────────────────────────

# Prompts and Roles

PROMPT_FEWSHOT = """
Role: You are an automated product catalog tagging system.
Task: Generate a hierarchical taxonomy path (Root > Parent > Leaf).

HIERARCHY RULES:
- Root   = Broadest domain (e.g. Electronics, Software, Office Supplies)
- Parent = Sub-domain within Root (broader than Leaf, narrower than Root)
- Leaf   = Most specific product category (ALWAYS the narrowest)
- Leaf is ALWAYS more specific than Parent
- Parent is ALWAYS more specific than Root

STRICT RULES:
- Do NOT use brand names (HP, Xerox, Sony, Epson, Logitech, Intel, Corel, Canon...)
- Do NOT use colors (black, silver, white...)
- Do NOT use sizes or specs (cm, gb, mhz, dpi...)
- Use GENERIC category names only

Examples:
Input: [mouse, keyboard, usb, receiver, wireless]
Output: Electronics > Computer Accessories > Input Devices

Input: [norton, antivirus, license, year, symantec]
Output: Software > Security Software > Antivirus Licenses

Input: [toner, cartridge, printer, xerox]
Output: Office Supplies > Printing Equipment > Toner Cartridges

Input: [corel, paintshop, studio, videostudio, pro]
Output: Software > Graphics & Design Software > Image Editing Software

Input: [paper, photo, sheets, epson, m²]
Output: Office Supplies > Printing Materials > Photo Paper

Input: [monitor, touch, display, pixels, cm]
Output: Electronics > Displays & Monitors > Touch Screen Monitors

Now Classify:
- Keywords: {kw}
- Examples: {ex}

Return ONLY the path in format: Root > Parent > Leaf
No explanation. No extra text.
"""

PROMPT_CRITIC_V3 = """
Task: Check and MINIMALLY fix this product taxonomy path.

Input Path: {path}
Keywords: {kw}
Examples: {ex}

# Hierarchy Rules: Root > Parent > Leaf
- Root   = Broadest domain (Electronics, Software, Office Supplies, Computers...)
- Parent = Sub-domain within Root (broader than Leaf, narrower than Root)
- Leaf   = Most specific product type (ALWAYS the narrowest and most precise)

CORRECT example:
  Software > Graphics & Design Software > Image Editing Software
  Root=Software ✅ | Parent=sub-domain ✅ | Leaf=specific type ✅

WRONG — do NOT do this:
  Software > Image Editing > Graphics Software
  (Leaf is broader than Parent — order is flipped) ❌

# Existing Roots (reuse if applicable):
{existing_roots}

SIMILAR CLUSTERS already labeled (style reference only):
{rag_context}

# Fix Rules:

1. ROOT wrong?
   - Physical hardware / devices  → "Electronics" or "Computers"
   - Paper / ink / office print   → "Office Supplies"
   - Programs / apps / licenses   → "Software"
   - Cables / networking          → "Electronics" or "Networking"
   - Mounts / brackets / stands   → "Electronics" or "Home & Garden"
   → Fix Root only. Keep Parent and Leaf unchanged unless also broken.

2. HIERARCHY ORDER wrong?
   - Leaf must be MORE specific than Parent
   - If flipped → swap Parent and Leaf only
   - Do NOT rewrite the entire path

3. LEAF is generic or vague?
   - "Products", "Items", "Accessories" alone → replace with specific name
   - Use keywords and examples to determine the right name

4. MORE THAN 3 levels?
   - Keep only the first 3 levels (Root > Parent > Leaf)

5. PATH already correct?
   → Return it UNCHANGED.
   → Do NOT simplify or rewrite a correct path.
   → Specific is ALWAYS better than generic.

Return ONLY the final path: Root > Parent > Leaf
Nothing else. No explanation. No extra text.
"""

# ─────────────────────────────────────────────
# LOAD EMBEDDING MODEL
# ─────────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)
print("Ready.\n")

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def call_local_llm(prompt_text):
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'user', 'content': prompt_text}
        ])
        return response['message']['content'].strip()
    except Exception as e:
        print(f"  LLM Error: {e}")
        return "Error"

def enforce_structure(label):
    """Force exactly 3 levels."""
    if not label or label.lower() in ['error', 'nan', '']:
        return None
    label = re.sub(r'\s*>\s*', ' > ', label.strip())
    parts = [p.strip() for p in label.split('>') if p.strip()]
    if len(parts) >= 3:
        return ' > '.join(parts[:3])
    elif len(parts) == 2:
        return f"{parts[0]} > {parts[1]} > Products"
    return None

def compute_coherence(label, examples_text):
    """Semantic similarity — reporting only, NOT a gate."""
    try:
        emb_label = embedder.encode([label])
        emb_ex    = embedder.encode([examples_text])
        return float(cosine_similarity(emb_label, emb_ex)[0][0])
    except:
        return 0.0

def call_with_retry(prompt, max_retries=MAX_RETRIES):
    """Retry only on structural failure."""
    current_prompt = prompt
    last_raw = "Error"
    for attempt in range(max_retries):
        raw      = call_local_llm(current_prompt)
        last_raw = raw
        label    = enforce_structure(raw)
        if label:
            return label, attempt + 1
        print(f"    ⚠ Attempt {attempt+1}: bad structure → retrying...")
        current_prompt += "\nCRITICAL: Return ONLY 'Root > Parent > Leaf'. Nothing else."
    fallback = enforce_structure(last_raw) or 'Unknown > Unknown > Unknown'
    return fallback, max_retries

def build_rag_context(kw, labeled_so_far, top_k=3):
    """Find most similar already-labeled clusters for Critic reference."""
    if not labeled_so_far:
        return "  None yet."
    query_emb  = embedder.encode([kw])
    kw_list    = [r['Keywords'] for r in labeled_so_far]
    label_list = [r['Fixed_Label'] for r in labeled_so_far]
    kw_embs    = embedder.encode(kw_list)
    sims       = cosine_similarity(query_emb, kw_embs)[0]
    top_idx    = np.argsort(sims)[::-1][:top_k]
    lines      = [f"  [{kw_list[i]}] → {label_list[i]}" for i in top_idx]
    return "\n".join(lines)

def pick_better_label(stage1, stage2, ex):
    """Keep Stage 2 unless it scores more than 0.03 below Stage 1."""
    score1 = compute_coherence(stage1, ex)
    score2 = compute_coherence(stage2, ex)
    if score2 >= score1 - 0.03:
        return stage2, score2, "Stage2"
    else:
        print(f"    ℹ Stage1 kept (s1={score1:.3f} > s2={score2:.3f})")
        return stage1, score1, "Stage1_kept"

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
try:
    df = pd.read_csv(INPUT_FILE)
    if SAMPLE_SIZE:
        df = df.head(SAMPLE_SIZE)
    print(f"Loaded {len(df)} clusters.\n")
except FileNotFoundError:
    print(f"ERROR: {INPUT_FILE} not found.")
    sys.exit()

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
results    = []
embeddings = []
start_time = time.time()

try:
    for idx, row in df.iterrows():
        cid = row['Cluster_ID']
        kw  = row['Keywords']
        ex  = row['Examples']

        print(f"[{idx+1}/{len(df)}] Cluster {cid} | {kw[:60]}")

        # ── Stage 1: Few-Shot ──────────────────────────────────────
        stage1, attempts1 = call_with_retry(
            PROMPT_FEWSHOT.format(kw=kw, ex=ex))
        print(f"  Stage 1: {stage1}")

        # ── Build existing roots context ───────────────────────────
        root_counts = {}
        for r in results:
            root_counts[r['Root']] = root_counts.get(r['Root'], 0) + 1
        top_roots = sorted(root_counts, key=root_counts.get, reverse=True)
        top_roots = top_roots[:MAX_ROOTS_IN_PROMPT]
        existing_roots_str = "\n".join(
            [f"  - {r} ({root_counts[r]} clusters)" for r in top_roots]
        ) if top_roots else "  None yet."

        # ── RAG: similar already-labeled clusters ──────────────────
        rag_context = build_rag_context(kw, results, top_k=3)

        # ── Stage 2: Critic (minimal fix) ─────────────────────────
        stage2, attempts2 = call_with_retry(
            PROMPT_CRITIC_V3.format(
                path=stage1,
                kw=kw,
                ex=ex,
                existing_roots=existing_roots_str,
                rag_context=rag_context))

        # ── Pick better label ──────────────────────────────────────
        final_label, final_score, chosen = pick_better_label(stage1, stage2, ex)
        total_attempts = attempts1 + attempts2

        print(f"  Stage 2: {stage2}")
        print(f"  Final:   [{chosen}] → {final_label}")
        print(f"  Score:   {final_score:.4f} | Attempts: {total_attempts}\n")

        # ── Parse levels ───────────────────────────────────────────
        parts  = [p.strip() for p in final_label.split('>')]
        root   = parts[0] if len(parts) > 0 else 'Unknown'
        parent = parts[1] if len(parts) > 1 else 'Unknown'
        leaf   = parts[2] if len(parts) > 2 else 'Unknown'

        embeddings.append(embedder.encode([kw + " " + ex])[0])

        results.append({
            "Cluster_ID":      cid,
            "Keywords":        kw,
            "Examples":        ex,
            "Stage1_FewShot":  stage1,
            "Stage2_Critic":   stage2,
            "Fixed_Label":     final_label,
            "Label_Source":    chosen,
            "Root":            root,
            "Parent":          parent,
            "Leaf":            leaf,
            "Coherence_Score": round(final_score, 4),
            "Total_Attempts":  total_attempts
        })

except KeyboardInterrupt:
    print("\n[!] Interrupted. Saving partial results...")

# ─────────────────────────────────────────────
# QUALITY BANDS + SAVE
# ─────────────────────────────────────────────
def score_band(s):
    if s >= 0.60: return "Excellent (≥0.60)"
    if s >= 0.45: return "Good (0.45-0.60)"
    if s >= 0.30: return "Acceptable (0.30-0.45)"
    return "Low (<0.30)"

duration  = (time.time() - start_time) / 60
final_df  = pd.DataFrame(results)
final_df['Quality'] = final_df['Coherence_Score'].apply(score_band)
final_df.to_csv(OUTPUT_FILE, index=False)

if embeddings:
    np.save('quicktest_v4_centroids.npy', np.array(embeddings))

stage1_kept = (final_df['Label_Source'] == 'Stage1_kept').sum()
stage2_used = (final_df['Label_Source'] == 'Stage2').sum()

print(f"\n{'='*55}")
print(f"QUICK TEST COMPLETE — v4  ({len(final_df)} clusters)")
print(f"{'='*55}")
print(f"Time:             {duration:.2f} minutes")
print(f"Avg coherence:    {final_df['Coherence_Score'].mean():.4f}")
print(f"Min coherence:    {final_df['Coherence_Score'].min():.4f}")
print(f"Max coherence:    {final_df['Coherence_Score'].max():.4f}")
print(f"Unique roots:     {final_df['Root'].nunique()}")
print(f"Stage 1 kept:     {stage1_kept} clusters")
print(f"Stage 2 used:     {stage2_used} clusters")
print(f"\nQuality bands:")
print(final_df['Quality'].value_counts().to_string())
print(f"\nLabel Preview:")
print(final_df[[
    'Cluster_ID', 'Stage1_FewShot', 'Stage2_Critic',
    'Fixed_Label', 'Label_Source', 'Coherence_Score'
]].to_string(index=False))
print(f"\n💾 Saved: {OUTPUT_FILE}")
