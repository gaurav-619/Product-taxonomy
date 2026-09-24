import pandas as pd
import ollama
import json
import os

# Configuration
MODEL_NAME = "qwen2.5:7b"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '..', '..', '..')

RAW_DATA_FILE = os.path.join(PROJECT_ROOT, 'Dataset', 'data', 'icecat_hierarchically_balanced.csv')
TAXONOMY_FILE = os.path.join(PROJECT_ROOT, 'Naming_Robust_Final_clean.csv')

def load_data():
    print("Loading datasets...")
    df_raw = pd.read_csv(RAW_DATA_FILE)
    df_tax = pd.read_csv(TAXONOMY_FILE)
    return df_raw, df_tax

def get_representative_products(df_raw, df_tax, cluster_id, top_n=10):
    """Get the titles and descriptions of products belonging to a specific cluster."""
    # Assuming df_raw has 'Cluster_ID' - if not, we use the original mapping or simulate it
    # We will grab products based on keywords matching if Cluster_ID is not directly on df_raw,
    # but the user has cluster assignments. Let's assume the user has a way to map them.
    # For demonstration, let's use the 'Examples' column from Naming_Robust_Final_clean.csv directly!
    
    row = df_tax[df_tax['Cluster_ID'] == cluster_id].iloc[0]
    category_name = f"{row['Root']} > {row['Parent']} > {row['Leaf']}"
    examples = row['Examples'].split(' || ')
    
    return category_name, examples

def generate_attribute_schema(category_name, products):
    """Use LLM to generate an attribute schema based on product examples."""
    prompt = f"""
Role: You are an E-commerce Data Architect.
Task: Extract a structured attribute schema for the product category: "{category_name}"

Here are representative product titles for this category:
{chr(10).join(f"- {p}" for p in products)}

Analyze these products and define the key attributes a buyer would use to filter them, or a seller would use to describe them.

Output the schema as a JSON object where:
- The keys are the attribute names.
- The values are an object containing 'type' ('categorical', 'numeric', or 'boolean') and 'values' (a list of examples or unit).

Example Output for "Memory Modules":
{{
  "Memory Type": {{"type": "categorical", "values": ["DDR3", "DDR4", "DDR5"]}},
  "Capacity": {{"type": "numeric", "unit": "GB"}},
  "Speed": {{"type": "numeric", "unit": "MHz"}},
  "Form Factor": {{"type": "categorical", "values": ["DIMM", "SO-DIMM"]}}
}}

Return ONLY the raw JSON object. Do not include markdown formatting or explanations.
"""

    print(f"Generating schema for: {category_name}...")
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'user', 'content': prompt}
        ])
        
        # Clean output
        content = response['message']['content'].strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        schema = json.loads(content)
        return schema
    except Exception as e:
        print(f"Error generating schema: {e}")
        return None

def main():
    if not os.path.exists(TAXONOMY_FILE):
        print(f"Cannot find {TAXONOMY_FILE}")
        return
        
    df_tax = pd.read_csv(TAXONOMY_FILE)
    
    # Pick a few diverse clusters to test
    # Memory Modules, Printers, Camera Case
    test_clusters = [6, 21, 5] 
    
    schemas = {}
    for cid in test_clusters:
        if cid in df_tax['Cluster_ID'].values:
            cat_name, products = get_representative_products(None, df_tax, cid)
            schema = generate_attribute_schema(cat_name, products)
            if schema:
                schemas[cat_name] = schema
                print(f"✅ Schema for {cat_name}:")
                print(json.dumps(schema, indent=2))
                print("-" * 50)
                
    # Save the output
    output_path = os.path.join(BASE_DIR, 'attribute_schemas.json')
    with open(output_path, 'w') as f:
        json.dump(schemas, f, indent=4)
    print(f"Saved generated schemas to {output_path}")

if __name__ == "__main__":
    main()
