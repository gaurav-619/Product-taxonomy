"""
Root entrypoint for Streamlit deployment.
Launches the full interactive Product Taxonomy demo.
"""
import os
import runpy
import sys

TARGET_APP = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Dataset",
    "scripts",
    "04_clustering",
    "newproduct",
    "app.py"
)

if __name__ == "__main__":
    if not os.path.exists(TARGET_APP):
        raise FileNotFoundError(f"Target application not found at: {TARGET_APP}")
    # Forward execution to the full application script
    runpy.run_path(TARGET_APP, run_name="__main__")
