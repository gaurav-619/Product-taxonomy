# LLM-Based E-Commerce Product Taxonomy & Discovery

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning pipeline that uses unsupervised clustering to construct a structured product taxonomy from raw catalog text, evaluates its structural and semantic properties, and demonstrates downstream utility in product discovery.

Developed as a Master's Thesis project, the system combines **label-aware stratified sampling**, **dense sentence embeddings**, **manifold dimensionality reduction**, **hybrid HDBSCAN/Ward agglomerative clustering**, and a **two-stage LLM labelling and critic verification pipeline**.

---

## 📌 Context & Problem Statement

In large-scale e-commerce catalogs, taxonomy construction and maintenance face distinct challenges:
1. **Severe Class Imbalance:** Head categories (e.g., standard cables, common peripherals) dominate catalogs, starving niche product categories of representation in unconstrained clustering.
2. **Vocabulary Diversity & Catalog Drift:** Unstructured titles and sparse specifications create noisy representations that challenge manual taxonomy curation.
3. **Downstream Utility Gap:** Clustering benchmarks often conclude with geometric metrics without evaluating practical applications such as search retrieval, metadata schemas, or semantic interoperability.

### System Overview & Technical Decisions
- **Label-Aware Stratified Sampling:** Sampled 489,000 raw Icecat product listings down to a balanced dataset of 35,607 items across 4 Root categories (`Electronics`, `Office Supplies`, `Software`, `Home & Office`) using reference labels to ensure balanced representation across the distribution tail.
- **Dense Embeddings & Manifold Projection:** Mapped multi-field product text (title, brand, specifications) into dense vectors using `all-MiniLM-L6-v2` (384 dimensions), followed by UMAP projection (5 dimensions) to preserve local and global manifold structure.
- **Hybrid Clustering Architecture:** Used **HDBSCAN** for density-based structure discovery, identifying a candidate granularity of $K = 411$ clusters under the selected configuration, followed by **Ward Agglomerative Hierarchical Clustering** constrained to $K = 411$ to produce a full partition with zero unassigned noise points.
- **Two-Stage LLM Labelling & Critic Verification:** Extracted TF-IDF cluster keywords and centroid exemplars, prompted Qwen 2.5 to generate candidate taxonomy labels (`Root > Parent > Leaf`), and verified them against parent-level sibling consistency and depth constraints using an automated critic.
- **Confidence-Aware Product Assignment:** Developed an inference engine using cosine distance to the 411 cluster centroids in 5D UMAP space, using confidence margins to distinguish clear assignments from ambiguous cases requiring human review.
- **Product Discovery & Practical Extensions:** Evaluated the downstream utility of the taxonomy through offline search relevance ranking (MRR), category-specific attribute schemas, and a standards-compliant SKOS/RDF taxonomy export.

---

## 🏗️ Architecture & Pipeline Flow

```
                      +-----------------------------+
                      |   Raw Icecat Product Data   | (489,000 items)
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Label-Aware Stratified      | (35,607 items, 4 Root partitions;
                      | Sampling                    |  combats head/tail imbalance)
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
                      | HDBSCAN Structure Discovery | (Explores cluster density;
                      |                             |  identifies candidate K = 411)
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Ward Agglomerative          | (K = 411 clusters; guarantees
                      | Clustering                  |  100% catalog coverage)
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Two-Stage LLM Labelling     | (TF-IDF keywords + exemplars ->
                      | & Critic Verification       |  Candidate label -> Critic review)
                      +--------------+--------------+
                                     |
              +----------------------+----------------------+
              |                                             |
              v                                             v
+---------------------------+                 +---------------------------+
| Confidence-Aware          |                 | Product Discovery &       |
| Inference Engine          |                 | Practical Extensions      |
| - Cosine centroid match   |                 | - Offline Search (MRR)    |
| - Margin ambiguity check  |                 | - Attribute Schemas       |
| - Potential drift flag    |                 | - SKOS/RDF Export (.ttl)  |
+---------------------------+                 +---------------------------+
```

---

## 🏷️ Two-Stage LLM Labelling & Critic Verification

Rather than treating LLM naming as an unconstrained single-prompt step, labelling was structured as a generation-and-verification loop:

```
TF-IDF Keywords + Centroid Exemplars
                 │
                 ▼
       Candidate Generation (Qwen 2.5)
                 │
                 ▼
        Proposed Taxonomy Label
         (Root > Parent > Leaf)
                 │
                 ▼
     Structural & Consistency Critic
  - Parent-child logical containment
  - Sibling name overlap / duplication
  - Specificity and format rules
                 │
        ┌────────┴────────┐
        ▼                 ▼
     Accept           Regenerate
   (92% on 1st pass)  (Modified prompt)
```

- **Critic Acceptance Rate:** The automated critic accepted **92%** of generated labels on the initial pass without requiring regeneration under the defined structural/consistency checks.
- **Structural Compliance:** 100% of final clusters met the required 3-level format (`Root > Parent > Leaf`) with no missing hierarchy levels.
- *Note:* Structural compliance validates hierarchy formatting and sibling distinctness; it is distinct from manual domain-expert semantic validation.

---

## 📊 Empirical Evaluation Results

The pipeline was quantitatively evaluated across internal geometric clustering metrics, external reference category alignment, and taxonomy structural properties.

| Evaluation Metric | Value | Metric Type | Description |
| :--- | :---: | :--- | :--- |
| **Silhouette Score** | **0.656** | Geometric / Internal | Internal cluster separation on 5D UMAP manifold ([-1, 1]) |
| **Davies-Bouldin Index** | **0.399** | Geometric / Internal | Average similarity ratio of each cluster with its most similar cluster |
| **Calinski-Harabasz Index**| **36,264.14** | Geometric / Internal | Ratio of between-cluster to within-cluster dispersion |
| **Mean Cluster Purity** | **74.28%** | Internal Consistency | Semantic alignment against reference catalog labels |
| **Adjusted Rand Index (ARI)**| **0.409** | External Benchmark | Degree of agreement with reference catalog partition |
| **Normalized Mutual Info (NMI)**| **0.793** | External Benchmark | Mutual information shared with reference hierarchy tiers |
| **Homogeneity Score** | **0.820** | External Benchmark | Extent to which each cluster contains only data points of a single class |
| **Completeness Score** | **0.768** | External Benchmark | Extent to which all data points of a given class are assigned to the same cluster |
| **V-Measure Score** | **0.793** | External Benchmark | Harmonic mean of homogeneity and completeness |
| **Catalog Partition Coverage** | **100.0%** | Structural Property | Full coverage under Ward clustering (zero unassigned noise points) |

---

## 🎯 Confidence-Aware Inference & Drift Flagging

When assigning newly observed products to the taxonomy, the inference engine applies cosine distance against the 411 centroids in 5D UMAP space. Confidence is determined by the distance margin between the nearest and second-nearest centroids ($\Delta = d_2 - d_1$):

```
       High Margin (Δ ≥ 0.0008)         ──>  🟢 High Confidence: Automated assignment
  Moderate Margin (0.0003 ≤ Δ < 0.0008) ──>  🟡 Moderate Confidence: Standard assignment
     Small Margin (0.0001 ≤ Δ < 0.0003) ──>  🟠 Low Confidence: Ambiguous boundary case
      Near-Zero Margin (Δ < 0.0001)     ──>  🔴 Flagged for Review: Potential out-of-taxonomy product
```

- **Ambiguous Assignments:** A small margin indicates a product sitting equidistant between two related categories (e.g., *USB Hubs* vs. *Docking Stations*), flagging it for human taxonomist verification.
- **Potential Taxonomy Drift:** Products far from all existing centroids with near-zero margins do not trigger automatic category creation; instead, they serve as a detection mechanism flagging candidate out-of-taxonomy items for human inspection.

---

## 🚀 Product Discovery Layer (Extensions)

To test the utility of the generated taxonomy beyond geometric evaluation, three practical extensions were implemented:

1. **🔍 Offline Search Relevance Evaluation:**
   - Evaluated baseline BM25 lexical keyword retrieval against taxonomy-aware reranking (filtering search candidate pools by predicted category).
   - Evaluated using **Mean Reciprocal Rank (MRR)** to assess whether category scoping ranks relevant products higher in the result set.
2. **📋 Category Attribute Schemas:**
   - Generated structured, category-specific attribute templates (e.g., RAM capacity, form factor, and bus speed for memory modules; sensor resolution, mount type, and ISO range for cameras).
   - Designed to support structured catalog filtering and marketplace product listing requirements.
3. **🔗 SKOS/RDF Representation:**
   - Serialized the complete 3-tier hierarchy into a W3C-compliant **SKOS (Simple Knowledge Organization System)** concept scheme using RDF Turtle syntax (`taxonomy.ttl`).
   - Represents categories using `skos:Concept`, `skos:broader`, `skos:narrower`, and `skos:prefLabel` for semantic web and knowledge graph tooling.

---

## 🖥️ Interactive Streamlit Dashboard

The repository includes an interactive web dashboard for inspecting taxonomy structure, evaluation results, and inference behaviour:

- **🏷️ Assign a New Product:** Input any product title and description to inspect real-time SBERT encoding, centroid distance, predicted 3-level taxonomy path, and confidence margin classification.
- **🌳 Interactive Taxonomy Tree:** Collapsible D3.js visualization rendering all 411 leaf clusters across the 4 root branches.
- **🔍 Explore the Taxonomy:** Catalog browser displaying cluster keywords, parent-level groupings, and centroid exemplar products.
- **📈 Evaluation & Metrics:** Interactive charts of cluster size distributions, silhouette analysis, and stability metrics across configuration runs.
- **📖 Methodology Pipeline:** Documentation of design decisions (HDBSCAN vs. Ward, SBERT model selection, and prompt templates).
- **🚀 Product Discovery Layer:**
  - *Tab 1: Search Relevance:* Offline BM25 vs. taxonomy-filtered retrieval benchmark and MRR comparison.
  - *Tab 2: Category Attribute Schemas:* Structured JSON schema inspector across categories.
  - *Tab 3: SKOS Ontology Export:* Direct preview and download of `taxonomy.ttl`.

---

## 📂 Repository Structure

```
├── app.py                          # Streamlit application entrypoint
├── requirements.txt                # Production dependency specification
├── requirements_freeze.txt         # Full environment freeze archive
├── Final_Evaluation_Results.csv    # Empirical evaluation metrics table
├── Naming_Robust_Final_clean.csv   # The 411 cluster taxonomy definitions
├── Images/                         # Thesis architecture diagrams and evaluation plots
└── Dataset/
    ├── data/
    │   ├── cluster_centroids_411.npy # (8 KB) Precomputed 5D UMAP centroids
    │   └── umap_fitted_5d.pkl        # (Git LFS) Fitted UMAP projection model
    └── scripts/
        ├── 01_exploration/         # Exploratory data analysis on raw Icecat catalog
        ├── 02_preprocessing/       # Cleaning, normalization & label-aware stratified balancing
        ├── 03_embedding/           # SBERT vectorization & UMAP manifold learning
        ├── 04_clustering/          # Agglomerative clustering, Ward linkage & tuning
        ├── 05_labeling/            # Two-stage LLM labelling & critic verification
        ├── 06_evaluation/          # Benchmark metrics computation (ARI, NMI, purity, silhouette)
        └── 07_enhancements/        # Product discovery: search relevance, schemas & SKOS export
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

## ☁️ Deployment on Streamlit Community Cloud

1. Fork or push this repository to GitHub.
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a **New app** and select:
   - **Repository:** `gaurav-619/Product-taxonomy`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**. Dependencies from `requirements.txt` and model artifacts via Git LFS are resolved automatically.

---

## 🔍 Limitations & Future Work

To maintain technical transparency, the following boundaries of this work should be noted:
- **Offline Evaluation:** Search relevance and retrieval benchmarks were conducted offline on sampled catalog queries rather than an online A/B testing environment.
- **Confidence vs. Formal OOD:** Margin-based confidence scores identify boundary ambiguity and anomalous products relative to existing centroids, but do not constitute a formally calibrated out-of-distribution detector.
- **Human Usability Validation:** While computational metrics (ARI, NMI, purity, silhouette) and automated critic checks were extensive, a formal user-centred usability study (e.g., card-sorting and tree-testing with domain experts) represents the recommended next step for taxonomy governance.

---

## 🔬 Citation

```bibtex
@mastersthesis{jadhav2026taxonomy,
  author       = {Gourav Suresh Jadhav},
  title        = {LLM-Based E-Commerce Product Taxonomy & Discovery},
  school       = {Master Thesis},
  year         = {2026}
}
```
