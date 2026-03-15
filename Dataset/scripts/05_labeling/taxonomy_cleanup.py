import pandas as pd
import re

INPUT_FILE  = 'Naming_Robust_Final.csv'
OUTPUT_FILE = 'Naming_Robust_Final_clean1.csv'

# ─── Known brand names to remove from Leaf ───────────────────────────────────
BRAND_NAMES = {
    'xerox', 'corel', 'hp', 'epson', 'oki', 'kaspersky', 'symantec',
    'logitech', 'fujitsu', 'lenovo', 'dell', 'canon', 'samsung', 'sony',
    'acer', 'asus', 'brother', 'lexmark', 'netgear', 'philips', 'sharp',
    'eaton', 'tripp', 'chief', 'vogel', 'projecta', 'da-lite', 'zyxel',
    'trend micro', 'eset', 'bigben', 'iiquu', 'renewd', 'varta', 'duracell',
    'herma', 'supermicro', 'vertiv', 'avocent', 'allied telesis', 'digicom',
    'toshiba', 'kensington', 'targus', 'sandberg', 'techly', 'newstar',
    'dataflex', 'speck', 'cellularline', 'marmitek', 'datalex', 'iris','paintshop', 'windows', 'microsoft', 'adobe'
}

# ─── Generic leaf terms to enrich ─────────────────────────────────────────────
GENERIC_LEAVES = {'products', 'units', 'items', 'accessories', 'parts'}

# Parent → better Leaf lookup when leaf is too generic
GENERIC_LEAF_LOOKUP = {
    'alarm clocks':         'Digital Alarm Clocks',
    'cabinets':             'Server Rack Cabinets',
    'screen protectors':    'Mobile Screen Protectors',
    'portable dvd players': 'Portable DVD Players',
    'rechargeable batteries': 'Li-Ion Rechargeable Batteries',
    'projectors':           'Projector Accessories',
    'mounting kits':        'Wall & Ceiling Mounting Kits',
    'digital photo frames': 'Digital Photo Frames',
}

# ─── Hardware roots wrongly assigned to Office Supplies ───────────────────────
HARDWARE_KEYWORDS_IN_PARENT = {
    'servers', 'networking', 'computer hardware', 'motherboard',
    'graphics cards', 'power supplies', 'storage devices', 'rack'
}

df = pd.read_csv(INPUT_FILE)

fixes = {
    'brand_removed': 0,
    'root_corrected': 0,
    'spec_dump_cleaned': 0,
    'generic_leaf_enriched': 0,
    'redundant_parent_leaf_fixed': 0,
}

def clean_leaf(leaf, parent):
    """Apply all cleaning rules to a Leaf value."""
    # ── Rule 3: Remove spec dumps (comma-separated tokens with numbers/units) ──
    if re.search(r'\b(black|white|gb|ghz|kg|mhz|dpi|cm|mm|pc|usm|ef)\b', leaf.lower()):
        leaf = re.sub(r'\s*[\(\[].*[\)\]]', '', leaf).strip()
        leaf = re.sub(r',\s*[A-Z][a-zA-Z0-9\s]*$', '', leaf).strip()

    # ── Rule 1: Remove brand names from Leaf ──────────────────────────────────
    leaf_lower = leaf.lower()
    for brand in BRAND_NAMES:
        pattern = r'\b' + re.escape(brand) + r'\b'
        if re.search(pattern, leaf_lower):
            leaf_cleaned = re.sub(pattern, '', leaf, flags=re.IGNORECASE).strip()
            leaf_cleaned = re.sub(r'\s{2,}', ' ', leaf_cleaned).strip(' -,')
            if len(leaf_cleaned) > 3:
                leaf = leaf_cleaned
                leaf_lower = leaf.lower()

    # ── Rule 4: Enrich generic leaves ─────────────────────────────────────────
    if leaf.strip().lower() in GENERIC_LEAVES:
        parent_lower = parent.strip().lower()
        if parent_lower in GENERIC_LEAF_LOOKUP:
            leaf = GENERIC_LEAF_LOOKUP[parent_lower]

    # ── Rule 5: Fix redundant Parent == Leaf ─────────────────────────────────
    if leaf.strip().lower() == parent.strip().lower():
        parent_lower = parent.strip().lower()
        if parent_lower in GENERIC_LEAF_LOOKUP:
            leaf = GENERIC_LEAF_LOOKUP[parent_lower]
        else:
            leaf = leaf + ' Products'

    return leaf.strip()


cleaned_roots  = []
cleaned_parents = []
cleaned_leaves  = []
cleaned_labels  = []

for _, row in df.iterrows():
    root   = str(row['Root']).strip()
    parent = str(row['Parent']).strip()
    leaf   = str(row['Leaf']).strip()

    original_leaf = leaf

    # ── Rule 2: Correct root domain for hardware misclassified as Office Supplies
    if root.lower() == 'office supplies':
        parent_lower = parent.lower()
        if any(kw in parent_lower for kw in HARDWARE_KEYWORDS_IN_PARENT):
            root = 'Electronics'
            fixes['root_corrected'] += 1

    # Apply leaf cleaning rules
    new_leaf = clean_leaf(leaf, parent)

    # Track what was fixed
    if new_leaf != original_leaf:
        leaf_lower = original_leaf.lower()
        
        # Determine exactly which rule fired for the counter
        if any(b in leaf_lower for b in BRAND_NAMES):
            fixes['brand_removed'] += 1
        elif re.search(r'\b(black|white|gb|ghz|kg|mhz|dpi|cm|mm|pc|usm|ef)\b', leaf_lower):
            fixes['spec_dump_cleaned'] += 1
        elif leaf_lower in GENERIC_LEAVES:
            fixes['generic_leaf_enriched'] += 1
        elif original_leaf.strip().lower() == parent.strip().lower(): 
            fixes['redundant_parent_leaf_fixed'] += 1
        # Catch edge case: If brand removal *caused* a redundancy that was then fixed
        elif new_leaf.lower() == parent.lower() + " products":
            fixes['redundant_parent_leaf_fixed'] += 1

    fixed_label = f"{root} > {parent} > {new_leaf}"

    cleaned_roots.append(root)
    cleaned_parents.append(parent)
    cleaned_leaves.append(new_leaf)
    cleaned_labels.append(fixed_label)

df['Root']        = cleaned_roots
df['Parent']      = cleaned_parents
df['Leaf']        = cleaned_leaves
df['Fixed_Label'] = cleaned_labels

df.to_csv(OUTPUT_FILE, index=False)

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"✅ Saved to {OUTPUT_FILE}")
print(f"\nPost-Processing Summary:")
print(f"  Brand names removed from Leaf      : {fixes['brand_removed']}")
print(f"  Root corrected to Electronics      : {fixes['root_corrected']}")
print(f"  Spec dumps cleaned from Leaf       : {fixes['spec_dump_cleaned']}")
print(f"  Generic leaves enriched            : {fixes['generic_leaf_enriched']}")
print(f"  Redundant Parent=Leaf fixed        : {fixes['redundant_parent_leaf_fixed']}")
total = sum(fixes.values())
print(f"\n  Total clusters fixed                : {total} / {len(df)}")
print(f"  Clean pass rate                     : {(len(df)-total)/len(df)*100:.1f}%")