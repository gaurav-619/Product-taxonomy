import pandas as pd
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, SKOS
import re
import os

def export_skos(input_csv, output_ttl):
    """
    Converts a flat CSV taxonomy (Root > Parent > Leaf) into a formal SKOS RDF Concept Scheme.
    """
    print(f"Loading taxonomy from {input_csv}...")
    df = pd.read_csv(input_csv)

    g = Graph()
    TAXA = Namespace("http://thesis.jadhav.dev/taxonomy/")
    g.bind("taxa", TAXA)
    g.bind("skos", SKOS)

    # Create the concept scheme
    scheme = TAXA["ProductTaxonomy"]
    g.add((scheme, RDF.type, SKOS.ConceptScheme))
    g.add((scheme, RDFS.label, Literal("Icecat Product Taxonomy (SKOS)")))

    seen = set()

    def to_uri(name):
        # Create a valid URI component
        return TAXA[re.sub(r'[^a-zA-Z0-9]', '_', str(name)).strip('_')]

    for _, row in df.iterrows():
        root, parent, leaf = row['Root'], row['Parent'], row['Leaf']
        
        root_uri   = to_uri(root)
        parent_uri = to_uri(f"{root}_{parent}")
        leaf_uri   = to_uri(f"{root}_{parent}_{leaf}")
        
        # 1. Add Root Level
        if root not in seen:
            g.add((root_uri, RDF.type, SKOS.Concept))
            g.add((root_uri, SKOS.prefLabel, Literal(root, lang="en")))
            g.add((root_uri, SKOS.topConceptOf, scheme))
            seen.add(root)
        
        # 2. Add Parent Level
        parent_key = f"{root}>{parent}"
        if parent_key not in seen:
            g.add((parent_uri, RDF.type, SKOS.Concept))
            g.add((parent_uri, SKOS.prefLabel, Literal(parent, lang="en")))
            g.add((parent_uri, SKOS.broader, root_uri))
            seen.add(parent_key)
        
        # 3. Add Leaf Level
        leaf_key = f"{root}>{parent}>{leaf}"
        if leaf_key not in seen:
            g.add((leaf_uri, RDF.type, SKOS.Concept))
            g.add((leaf_uri, SKOS.prefLabel, Literal(leaf, lang="en")))
            g.add((leaf_uri, SKOS.broader, parent_uri))
            
            # Attach metadata (cluster ID and coherence score from the thesis pipeline)
            g.add((leaf_uri, TAXA.clusterID, Literal(int(row['Cluster_ID']))))
            g.add((leaf_uri, TAXA.coherenceScore, Literal(float(row['Coherence_Score']))))
            seen.add(leaf_key)

    print(f"Exporting SKOS representation to {output_ttl}...")
    g.serialize(destination=output_ttl, format="turtle")
    
    print(f"Export complete. Generated {len(g)} RDF triples.")
    print(f"   Roots:   {df['Root'].nunique()}")
    print(f"   Parents: {df['Parent'].nunique()}")  
    print(f"   Leaves:  {df['Leaf'].nunique()}")

if __name__ == "__main__":
    # Assuming script is run from project root or this file's dir
    # Try finding the file relative to the project root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(base_dir, '..', '..', '..')
    input_file = os.path.join(project_root, 'Naming_Robust_Final_clean.csv')
    
    # Fallback to local if not found
    if not os.path.exists(input_file):
        input_file = 'Naming_Robust_Final_clean.csv'
        
    output_file = os.path.join(base_dir, 'taxonomy.ttl')
    
    if os.path.exists(input_file):
        export_skos(input_file, output_file)
    else:
        print(f"Error: Could not find {input_file}")
