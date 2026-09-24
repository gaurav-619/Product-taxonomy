# Streamlit Demo — Product Taxonomy Assignment

This folder contains the Streamlit demonstration for the Master Thesis product taxonomy pipeline.

## What this app does

- Accepts a product title and optional description
- Encodes text with SBERT and projects it through a saved UMAP model
- Matches the product to the nearest taxonomy cluster centroid
- Returns a confidence-tiered taxonomy assignment
- Provides an interactive D3 tree, search interface, and evaluation dashboard

## How to run

From the repository root:

```bash
pip install -r requirements.txt
streamlit run Dataset/scripts/04_clustering/newproduct/app.py
```

If you prefer a local Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run Dataset/scripts/04_clustering/newproduct/app.py
```

## Required artifacts

The app expects the following files to exist:

- `Dataset/data/umap_fitted_5d.pkl`
- `Dataset/data/cluster_centroids_411.npy`
- `Naming_Robust_Final_clean.csv` (at the repository root)

## Main app pages

- **Assign a New Product** — automatic cluster assignment with confidence and alternative candidates
- **Interactive Taxonomy Tree** — collapsible tree visualization of the taxonomy structure
- **Explore the Taxonomy** — search, browse, and inspect clusters and leaves
- **Evaluation Results** — charts and metrics for clustering quality, LLM alignment, and taxonomy balance
- **Methodology Pipeline** — overview of the pipeline decisions and system architecture

## Troubleshooting

- If the app fails while loading models, verify that the artifact files exist and are readable.
- If the taxonomy CSV is missing, place `Naming_Robust_Final_clean.csv` at the repository root.
- If Streamlit does not open, copy the local URL from the terminal into your browser.

## Notes

This demo is designed to showcase both the technical pipeline and the user-facing experience for taxonomy assignment and inspection.