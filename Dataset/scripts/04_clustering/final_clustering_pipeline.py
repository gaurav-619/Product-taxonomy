import numpy as np

from sklearn.cluster import AgglomerativeClustering

from sklearn.feature_extraction.text import TfidfVectorizer



# --- CONFIG ---

INPUT_CSV = r'C:\Users\goura\Documents\Master Thesis\Dataset\data\icecat_hierarchically_balanced.csv'

INPUT_EMBEDDINGS = r'C:\Users\goura\Documents\Master Thesis\Dataset\data\umap_5d.npy'

K_TARGET = 411



print("Loading Data...")

df = pd.read_csv(INPUT_CSV)

X = np.load(INPUT_EMBEDDINGS)

if len(df) != len(X): df = df.iloc[:len(X)]



# Use Title for readability

text_col = 'Title' if 'Title' in df.columns else 'product_name'

df['text_clean'] = df[text_col].fillna('')



print("Running Final Ward Clustering...")

model = AgglomerativeClustering(n_clusters=K_TARGET, linkage='ward')

df['Cluster_ID'] = model.fit_predict(X)



print("Generating Profiles...")

profile_data = []



for cid in sorted(df['Cluster_ID'].unique()):

    subset = df[df['Cluster_ID'] == cid]

    

    # 1. Get Keywords

    if len(subset) > 1:

        tfidf = TfidfVectorizer(stop_words='english', max_features=5)

        try:

            tfidf.fit_transform(subset['text_clean'])

            kw = ", ".join(tfidf.get_feature_names_out())

        except: kw = "N/A"

    else:

        kw = "Single Item"

        

    # 2. Get 3 Example Products

    examples = subset[text_col].head(3).tolist()

    example_str = " || ".join([str(e) for e in examples])

    

    profile_data.append({

        'Cluster_ID': cid,

        'Size': len(subset),

        'Keywords': kw,

        'Examples': example_str

    })



pd.DataFrame(profile_data).to_csv('Cluster_Profiles_for_LLM.csv', index=False)

print("Saved 'Cluster_Profiles_for_LLM.csv")