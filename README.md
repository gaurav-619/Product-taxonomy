# Unsupervised & Agentic Product Taxonomy Generation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning system developed for a Master's Thesis to construct, validate, and serve e-commerce product taxonomies from raw, noisy catalog data. The system combines **hierarchical stratified sampling**, **dense sentence embeddings**, **manifold dimensionality reduction**, **Ward's agglomerative clustering**, and an **LLM-driven taxonomy naming agent with automated critic verification**.

---

## 📌 Executive Summary & Core Contributions

In large-scale e-commerce catalogs, category taxonomies suffer from three systemic challenges:
1. **Severe Class Imbalance:** Head categories (e.g., generic cables, basic adapters) dominate catalogs, starving niche product categories of representation.
2. **Taxonomy Drift & High Maintenance Costs:** Manual taxonomy construction by human catalog specialists does not scale with fast-moving inventory.
3. **Lack of Downstream Utility:** Traditional clustering projects stop at cluster generation without proving utility in search ranking, attribute extraction, or semantic web interoperability.

### Key Contributions of this Work
- **Hierarchical Stratified Balancing:** Curated 489,000 raw Icecat product listings down to a balanced dataset of 35,607 items across 4 Root categories, mitigating extreme head/tail frequency skews.
- **Hybrid Embedding & Manifold Clustering:** Mapped multi-field product text (title, brand, specs) using `all-MiniLM-L6-v2` (384-d) reduced via UMAP (5-d) into 411 leaf clusters via Ward's hierarchical agglomerative clustering.
- **Autonomous LLM Taxonomy Naming & Critic Loop:** Extracted TF-IDF cluster keywords and centroid exemplars to prompt an LLM agent that synthesized a strict 3-tier taxonomy (`Root > Parent > Leaf`) validated by an automated structural critic.
- **Margin-Based Confidence Inference:** Built an inference engine assigning incoming products using cosine distance to 411 centroids with margin-based out-of-distribution (OOD) rejection.
- **Downstream E-Commerce Integrations:**
  - **Search Relevance:** Benchmarked baseline BM25 retrieval against taxonomy-filtered reranking, measuring Mean Reciprocal Rank (MRR).
  - **Dynamic Attribute Extraction:** Formulated category-specific schema profiles ready for Shopify/Magento metadata ingestion.
  - **Knowledge Graph Export:** Serialized the entire taxonomy into W3C-compliant **SKOS RDF triples** (`taxonomy.ttl`) for enterprise semantic web interoperability.

---

## 🏗️ Architecture & Pipeline Flow

```
                      +-----------------------------+
                      |   Raw Icecat Product Data   | (489,000 items)
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Hierarchical Balancing    | (35,607 items, 4 Root partitions)
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Dense Semantic Embeddings  | (SBERT: all-MiniLM-L6-v2, 384-d)
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  UMAP Manifold Reduction    | (384-d -> 5-d manifold projection)
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Agglomerative Clustering   | (Ward Linkage, K = 411 Leaf Clusters)
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   LLM Taxonomy Naming       | (Centroid Exemplars + Keywords)
                      |   + Structural Critic Loop  | (Root > Parent > Leaf validation)
                      +--------------+--------------+
                                     |
              +----------------------+----------------------+
              |                                             |
              v                                             v
+---------------------------+                 +---------------------------+
|  Real-Time Inference      |                 |  Production Enhancements  |
|  - Cosine Centroid Match  |                 |  - Taxonomy Search (MRR)  |
|  - Margin Confidence OOD  |                 |  - Attribute Schemas      |
+---------------------------+                 |  - SKOS RDF Export (.ttl) |
                                              +---------------------------+
```

---

## 📊 Empirical Evaluation Results

The pipeline was quantitatively evaluated across internal geometric clustering metrics, external ground-truth agreement, and taxonomy structural integrity.

| Evaluation Metric | Value | Category | Interpretation |
| :--- | :---: | :--- | :--- |
| **Silhouette Score** | **0.656** | Geometric / Internal | Strong separation between manifold clusters ([-1, 1]) |
| **Davies-Bouldin Index** | **0.399** | Geometric / Internal | Low intra-cluster spread relative to centroid distance (lower is better) |
| **Calinski-Harabasz Index**| **36,264.14** | Geometric / Internal | High ratio of between-cluster to within-cluster dispersion |
| **Mean Cluster Purity** | **74.28%** | Internal Consistency | High semantic alignment of products within leaf clusters |
| **Adjusted Rand Index (ARI)**| **0.409** | External Benchmark | Alignment with external ground-truth categories |
| **Normalized Mutual Info (NMI)**| **0.793** | External Benchmark | High mutual dependence with human-labeled catalog tiers |
| **Homogeneity Score** | **0.820** | External Benchmark | Each cluster predominantly contains a single product type |
| **Completeness Score** | **0.768** | External Benchmark | Members of a given class are predominantly assigned to the same cluster |
| **V-Measure Score** | **0.793** | External Benchmark | Harmonic mean of homogeneity and completeness |
| **Product Catalog Coverage** | **100.0%** | Structural | Full partition coverage without unassigned noise points |

---

## 🖥️ Interactive Streamlit Dashboard

The repository includes a comprehensive interactive web application built with Streamlit and D3.js:

1. **🏷️ Assign a New Product:**
   - Input product titles and descriptions.
   - Computes real-time SBERT embeddings, projects into UMAP space, and matches against the 411 cluster centroids.
   - Computes confidence scores via margin heuristics:
     - 🟢 **High Confidence:** $\Delta \ge 0.0008$
     - 🟡 **Medium Confidence:** $0.0003 \le \Delta < 0.0008$
     - 🟠 **Low Confidence:** $0.0001 \le \Delta < 0.0003$
     - 🔴 **Very Low / OOD:** $\Delta < 0.0001$ (flags for human review / anomaly queue).
2. **🌳 Interactive Taxonomy Tree:**
   - Collapsible D3.js tree visualization rendering the complete 3-level taxonomy hierarchy.
3. **🔍 Explore the Taxonomy:**
   - Deep-dive browser across all 411 leaf clusters, showing top TF-IDF keywords, parent categories, and representative product exemplars.
4. **📈 Evaluation & Metrics:**
   - Visualizations of clustering distributions, silhouette benchmarks, and stability analyses.
5. **📖 Methodology & Design Decisions:**
   - Complete architectural walkthrough covering why HDBSCAN was supplemented with Ward Linkage, SBERT model selection, and prompt design.
6. **🚀 Product Discovery & Enhancements:**
   - **Search Relevance Benchmark:** Compares standard BM25 retrieval against taxonomy-filtered retrieval with MRR gains.
   - **Dynamic Attribute Extraction:** Category-specific schema definitions (e.g., RAM capacity for memory, sensor resolution for cameras).
   - **SKOS Knowledge Graph Export:** Preview and instant download of the complete taxonomy in W3C SKOS RDF (`taxonomy.ttl`).

---

## 📂 Repository Structure

```
├── app.py                          # Streamlit application entrypoint
├── requirements.txt                # Production dependency specification
├── requirements_freeze.txt         # Full environment freeze archive
├── Final_Evaluation_Results.csv    # Official empirical evaluation metrics
├── Naming_Robust_Final_clean.csv   # The 411 production-ready cluster taxonomy
├── calculate_structural_metrics.py # Script for computing graph & tree metrics
├── Images/                         # Curated thesis figures, plots, and diagrams
└── Dataset/
    ├── data/
    │   └── cluster_centroids_411.npy # (8 KB) Precomputed 5D UMAP centroids
    └── scripts/
        ├── 01_exploration/         # Exploratory data analysis on raw Icecat catalog
        ├── 02_preprocessing/       # Data cleaning, text normalization & stratified balancing
        ├── 03_embedding/           # SBERT vectorization & UMAP manifold learning
        ├── 04_clustering/          # Agglomerative clustering, Ward linkage & tuning
        ├── 05_labeling/            # Autonomous LLM prompt pipeline & Critic validation
        ├── 06_evaluation/          # Benchmarks, purity, ARI/NMI & silhouette computation
        └── 07_enhancements/        # Search relevance (MRR), schemas, and SKOS export
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/gaurav-619/Product-taxonomy.git
cd Product-taxonomy
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv

# On Linux / macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Interactive Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this repository to your GitHub account (`gaurav-619/Product-taxonomy`).
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"** and select:
   - **Repository:** `gaurav-619/Product-taxonomy`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**. Streamlit Cloud will automatically install dependencies from `requirements.txt` and launch your live application.

---

## 🔬 Citation & Academic Reference

```bibtex
@mastersthesis{jadhav2026taxonomy,
  author       = {Gourav Suresh Jadhav},
  title        = {Unsupervised and Agentic Product Taxonomy Generation from E-Commerce Catalogs},
  school       = {Master Thesis},
  year         = {2026}
}
```
