# Master Thesis: Product Taxonomy Generation

This repository contains the implementation and results for a Master Thesis focused on autonomous product taxonomy generation and labeling using Large Language Models (LLMs) and clustering techniques.

## Project Structure

- `Dataset/scripts/`: 
  - `01_exploration/`: Initial data analysis and visualization.
  - `02_preprocessing/`: Data cleaning and balancing scripts.
  - `03_embedding/`: Text embedding generation using MiniLM and UMAP projection.
  - `04_clustering/`: Hierarchical and HDBSCAN clustering implementations.
  - `05_labeling/`: LLM-based labeling and taxonomy tree construction.
  - `06_evaluation/`: Multi-metric evaluation framework (BERTScore, Coherence, etc.).
- `Images/`: Key visualizations and pipeline architectures.
- `Complete_Thesis_Documentation.md`: Comprehensive documentation of the thesis methodology and results.

## Setup

1. **Environment**: Python 3.12+ recommended.
2. **Dependencies**: Install required packages using:
   ```bash
   pip install -r requirements.txt
   ```
3. **Data**: Large dataset files are excluded from this repository due to size limits.

## Usage

Each directory in `Dataset/scripts/` corresponds to a stage in the pipeline. Refer to the documentation in each script for specific execution details.
