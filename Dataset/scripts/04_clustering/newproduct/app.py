# app_enhanced.py — Automated Product Taxonomy Generation
# Master Thesis — Gourav Suresh Jadhav

# ENHANCED VERSION — D3 Tree · Methodology Page · Session State · Charts

import os
import json
import random
import time
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Product Taxonomy — Thesis Demo",
    page_icon="🏷️",
    layout="wide"
)

# ── GLOBAL STYLE ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1  { font-size: 1.7rem !important; font-weight: 700 !important; color: #0f172a; }
    h2  { font-size: 1.2rem !important; font-weight: 600 !important; color: #0f172a; }
    h3  { font-size: 1rem   !important; }
    .stMetric label { font-size: 0.75rem !important; color: #64748b; }
    .stMetric [data-testid="metric-container"] {
        background: #f8fafc; padding: 10px 14px;
        border-radius: 8px; border: 1px solid #e2e8f0;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0 !important; border-radius: 8px;
    }
    .stButton button { border-radius: 6px; font-size: 0.82rem; }
    footer { visibility: hidden; }
    .stTabs [data-baseweb="tab"] { font-size: 0.85rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── PATHS ─────────────────────────────────────────────────────────────────────
# Using relative paths for deployment compatibility
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, '..', '..', '..', 'data')
UMAP_MODEL  = os.path.join(DATA_DIR, 'umap_fitted_5d.pkl')
CENTROIDS   = os.path.join(DATA_DIR, 'cluster_centroids_411.npy')
TAXONOMY    = os.path.join(BASE_DIR, '..', '..', '..', '..', 'Naming_Robust_Final_clean.csv')

HIGH_CONF   = 0.0008
MEDIUM_CONF = 0.0003
LOW_CONF    = 0.0001

# ── ROOT COLOURS ──────────────────────────────────────────────────────────────
ROOT_COLORS = {
    "Electronics"    : ("#0e6fa3", "#e0f2fe"),
    "Office Supplies": ("#1a7f37", "#dcfce7"),
    "Software"       : ("#7c3aed", "#ede9fe"),
    "Home & Office"  : ("#b45309", "#fef3c7"),
}
ROOT_HEX = {
    "Electronics"    : "#0e6fa3",
    "Office Supplies": "#1a7f37",
    "Software"       : "#7c3aed",
    "Home & Office"  : "#b45309",
}

def root_badge(root):
    fg, bg = ROOT_COLORS.get(root, ("#374151", "#f3f4f6"))
    return (
        f"<span style='background:{bg}; color:{fg}; font-size:0.78rem; "
        f"font-weight:700; padding:3px 10px; border-radius:12px; "
        f"border:1px solid {fg}40'>{root}</span>"
    )

# ── LOAD ARTIFACTS ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    try:
        sbert      = SentenceTransformer('all-MiniLM-L6-v2')
        umap_model = joblib.load(UMAP_MODEL)
        centroids  = np.load(CENTROIDS)
        taxonomy   = pd.read_csv(TAXONOMY)[['Cluster_ID', 'Root', 'Parent', 'Leaf']]
        return sbert, umap_model, centroids, taxonomy
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        st.stop()

@st.cache_data(show_spinner=False)
def load_full_taxonomy():
    try:
        df = pd.read_csv(TAXONOMY)
        df['Examples_List'] = df['Examples'].fillna('').apply(
            lambda x: [e.strip() for e in x.split('||') if e.strip()]
        )
        df['Keywords_Clean'] = df['Keywords'].fillna('')
        return df
    except Exception as e:
        st.error(f"Failed to load taxonomy: {e}")
        st.stop()

@st.cache_data(show_spinner=False)
def build_tree_cache():
    """Cache the tree JSON to avoid rebuilding on every page load."""
    tax = load_full_taxonomy()
    return build_tree_json(tax)

with st.spinner("Loading taxonomy system..."):
    sbert, umap_model, centroids, taxonomy_df = load_artifacts()

# ── SESSION STATE INIT ────────────────────────────────────────────────────────
if 'title' not in st.session_state:
    st.session_state.title = ""
if 'description' not in st.session_state:
    st.session_state.description = ""
if 'assignment_result' not in st.session_state:
    st.session_state.assignment_result = None

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_label(cluster_id):
    row = taxonomy_df[taxonomy_df['Cluster_ID'] == cluster_id]
    if len(row) == 0:
        return "Unknown", "Unknown", "Unknown"
    r = row.iloc[0]
    return r['Root'], r['Parent'], r['Leaf']

def assign_to_taxonomy(title, description=""):
    text      = (title + " " + description).strip()
    embedding = sbert.encode([text])
    umap_vec  = umap_model.transform(embedding)
    distances = cosine_distances(umap_vec, centroids)[0]
    top3_idx  = distances.argsort()[:3]
    top3_dist = distances[top3_idx]
    margin    = float(top3_dist[1] - top3_dist[0])

    candidates = []
    for cid, dist in zip(top3_idx, top3_dist):
        root, parent, leaf = get_label(int(cid))
        candidates.append({
            'path'    : f"{root} > {parent} > {leaf}",
            'root'    : root, 'parent': parent, 'leaf': leaf,
            'distance': round(float(dist), 6)
        })

    if candidates[0]['path'] == candidates[1]['path']:
        tier, stars, color = "High Confidence", "★★★★★", "#1a7f37"
        note = "Dual cluster confirmed"
    elif margin >= HIGH_CONF:
        tier, stars, color = "High Confidence", "★★★★★", "#1a7f37"
        note = "Auto-assign"
    elif margin >= MEDIUM_CONF:
        tier, stars, color = "Medium Confidence", "★★★☆☆", "#9a6700"
        note = "Auto-assign — log for review"
    elif margin >= LOW_CONF:
        tier, stars, color = "Low Confidence", "★★☆☆☆", "#b35900"
        note = "Flag for manual review"
    else:
        tier, stars, color = "Very Low — Potential New Category", "★☆☆☆☆", "#b91c1c"
        note = "Not well represented in training data"

    return {
        'candidates': candidates, 'margin': round(margin, 6),
        'tier': tier, 'stars': stars, 'color': color, 'note': note
    }

def quality_color(q):
    return {
        'Excellent (≥0.60)'     : '#1a7f37',
        'Good (0.45-0.60)'      : '#0e6fa3',
        'Acceptable (0.30-0.45)': '#9a6700',
        'Low (<0.30)'           : '#b91c1c'
    }.get(q, '#555')

def thesis_banner():
    # Removed thesis metadata banner per request
    pass

# ── D3 TREE BUILDER ───────────────────────────────────────────────────────────
def build_tree_json(tax_df, highlight_root=None, highlight_parent=None, highlight_leaf=None):
    """Build hierarchy dict for D3 from taxonomy dataframe."""
    tree = {"name": "Taxonomy", "children": []}
    for root_name, root_grp in tax_df.groupby('Root'):
        root_color = ROOT_HEX.get(root_name, "#64748b")
        root_node = {
            "name": root_name,
            "color": root_color,
            "count": int(root_grp['Cluster_ID'].nunique()),
            "highlighted": (root_name == highlight_root),
            "children": []
        }
        for parent_name, parent_grp in root_grp.groupby('Parent'):
            parent_node = {
                "name": parent_name,
                "color": root_color,
                "count": int(parent_grp['Cluster_ID'].nunique()),
                "highlighted": (root_name == highlight_root and parent_name == highlight_parent),
                "children": []
            }
            # Group by unique leaf names to avoid duplicates
            for leaf_name, leaf_grp in parent_grp.groupby('Leaf'):
                leaf_highlighted = (
                    root_name == highlight_root and
                    parent_name == highlight_parent and
                    leaf_name == highlight_leaf
                )
                parent_node["children"].append({
                    "name": leaf_name,
                    "color": root_color,
                    "cluster_id": int(leaf_grp['Cluster_ID'].iloc[0]),  # Take first cluster ID
                    "count": len(leaf_grp),  # Number of clusters with this leaf
                    "highlighted": leaf_highlighted,
                    "children": []
                })
            root_node["children"].append(parent_node)
        tree["children"].append(root_node)
    return tree

def render_d3_tree(tree_data, height=700, highlight_path=None):
    """Render an interactive collapsible D3 tree via st.components."""
    tree_json = json.dumps(tree_data)
    highlight_label = highlight_path or ""

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ margin: 0; background: #f8fafc; font-family: 'Inter', sans-serif; overflow: auto; }}
  .node circle {{
    stroke-width: 2px;
    cursor: pointer;
    transition: r 0.2s, stroke-width 0.2s;
  }}
  .node circle:hover {{ stroke-width: 3px; }}
  .node text {{
    font-size: 11px;
    fill: #1e293b;
    font-family: 'Inter', sans-serif;
    pointer-events: none;
  }}
  .node.highlighted circle {{
    stroke-width: 4px !important;
    filter: drop-shadow(0 0 8px rgba(255,200,0,0.8));
  }}
  .node.highlighted text {{
    font-weight: 700;
    fill: #b45309;
  }}
  .link {{
    fill: none;
    stroke: #cbd5e1;
    stroke-width: 1.5px;
    transition: stroke 0.3s;
  }}
  .link.highlighted-link {{
    stroke: #f59e0b;
    stroke-width: 2.5px;
  }}
  .tooltip {{
    position: absolute;
    background: #0f172a;
    color: #f1f5f9;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 12px;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.2s;
    max-width: 220px;
    line-height: 1.5;
    z-index: 999;
  }}
  #controls {{
    position: absolute;
    top: 10px;
    right: 10px;
    display: flex;
    gap: 6px;
    flex-direction: column;
  }}
  #controls button {{
    background: #0f172a;
    color: #94a3b8;
    border: none;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 11px;
    cursor: pointer;
    font-family: 'Inter', sans-serif;
  }}
  #controls button:hover {{ background: #1e293b; color: #f1f5f9; }}
  #legend {{
    position: absolute;
    top: 10px;
    left: 10px;
    background: rgba(255,255,255,0.95);
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 11px;
    line-height: 2;
  }}
  #legend .dot {{
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
  }}
  #search-bar {{
    position: absolute;
    bottom: 10px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    gap: 6px;
    background: white;
    padding: 8px 12px;
    border-radius: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    width: 340px;
  }}
  #search-bar input {{
    border: none; outline: none; font-size: 12px;
    width: 100%; font-family: 'Inter', sans-serif; color: #1e293b;
  }}
  #node-count {{
    position: absolute;
    bottom: 60px;
    right: 10px;
    font-size: 10px;
    color: #94a3b8;
    text-align: right;
  }}
</style>
</head>
<body>
<div id="legend">
  <div><span class="dot" style="background:#0e6fa3"></span>Electronics</div>
  <div><span class="dot" style="background:#1a7f37"></span>Office Supplies</div>
  <div><span class="dot" style="background:#7c3aed"></span>Software</div>
  <div><span class="dot" style="background:#b45309"></span>Home & Office</div>
</div>
<div id="controls">
  <button onclick="expandAll()">Expand All</button>
  <button onclick="collapseAll()">Collapse All</button>
  <button onclick="resetZoom()">Reset View</button>
</div>
<div id="node-count"></div>
<div id="search-bar">
  <span style="color:#94a3b8">🔍</span>
  <input type="text" id="search-input" placeholder="Search leaf category..." oninput="searchNodes(this.value)"/>
</div>
<div class="tooltip" id="tooltip"></div>
<svg id="tree-svg"></svg>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
const treeData = {tree_json};
const highlightPath = "{highlight_label}";

const width  = window.innerWidth;
const height = {height};
const margin = {{top: 40, right: 180, bottom: 60, left: 60}};

const svg = d3.select("#tree-svg")
  .attr("width", width)
  .attr("height", height);

const g = svg.append("g").attr("transform", `translate(${{margin.left}},${{margin.top}})`);

// Zoom
const zoom = d3.zoom()
  .scaleExtent([0.2, 3])
  .on("zoom", e => g.attr("transform", e.transform));
svg.call(zoom);

const treeLayout = d3.tree().size([height - margin.top - margin.bottom, width - margin.left - margin.right - 100]);

let root = d3.hierarchy(treeData);

// Collapse all by default except root level
root.children.forEach(collapse);

function collapse(d) {{
  if (d.children) {{
    d._children = d.children;
    d._children.forEach(collapse);
    d.children = null;
  }}
}}

function expand(d) {{
  if (d._children) {{
    d.children = d._children;
    d._children = null;
  }}
  if (d.children) d.children.forEach(expand);
}}

// Auto-expand highlighted path
if (highlightPath) {{
  const parts = highlightPath.split(" > ");
  function expandPath(node, depth) {{
    if (depth >= parts.length) return;
    if (node._children) {{
      node.children = node._children;
      node._children = null;
    }}
    if (node.children) {{
      node.children.forEach(child => {{
        if (child.data.name === parts[depth]) expandPath(child, depth + 1);
        else if (child.data.name === parts[depth]) expandPath(child, depth + 1);
      }});
    }}
  }}
  if (root._children) {{ root.children = root._children; root._children = null; }}
  root.children.forEach(child => {{
    if (child.data.name === parts[0]) expandPath(child, 1);
  }});
}}

const tooltip = document.getElementById("tooltip");

function update(source) {{
  treeLayout(root);
  const nodes = root.descendants();
  const links = root.links();

  document.getElementById("node-count").textContent =
    `${{nodes.length}} nodes visible`;

  // Links
  const link = g.selectAll(".link").data(links, d => d.target.id);

  const linkEnter = link.enter().append("path")
    .attr("class", d => {{
      const isHighlighted = highlightPath &&
        d.target.data.highlighted && d.source.data.highlighted;
      return "link" + (isHighlighted ? " highlighted-link" : "");
    }})
    .attr("d", d => {{
      const o = {{x: source.x0 || source.x, y: source.y0 || source.y}};
      return diagonal({{source: o, target: o}});
    }});

  link.merge(linkEnter).transition().duration(400)
    .attr("d", diagonal)
    .attr("class", d => {{
      const isHl = highlightPath && d.target.data.highlighted && d.source.data.highlighted;
      return "link" + (isHl ? " highlighted-link" : "");
    }});

  link.exit().transition().duration(400)
    .attr("d", d => {{
      const o = {{x: source.x, y: source.y}};
      return diagonal({{source: o, target: o}});
    }}).remove();

  // Nodes
  const node = g.selectAll(".node").data(nodes, d => d.id || (d.id = Math.random()));

  const nodeEnter = node.enter().append("g")
    .attr("class", d => "node" + (d.data.highlighted ? " highlighted" : ""))
    .attr("transform", d => `translate(${{source.y0||source.y}},${{source.x0||source.x}})`)
    .on("click", (e, d) => {{ toggle(d); update(d); }})
    .on("mouseover", (e, d) => {{
      tooltip.style.opacity = 1;
      const depth = d.depth;
      const depthLabel = ["Root", "Category Root", "Parent", "Leaf"][Math.min(depth, 3)];
      const count = d.data.count ? `${{d.data.count}} clusters` : "";
      const cid   = d.data.cluster_id != null ? `Cluster #${{d.data.cluster_id}}` : "";
      tooltip.innerHTML = `<b>${{d.data.name}}</b><br><span style="color:#94a3b8">${{depthLabel}} · ${{count || cid}}</span>`;
      tooltip.style.left = (e.pageX + 12) + "px";
      tooltip.style.top  = (e.pageY - 28) + "px";
    }})
    .on("mousemove", e => {{
      tooltip.style.left = (e.pageX + 12) + "px";
      tooltip.style.top  = (e.pageY - 28) + "px";
    }})
    .on("mouseout", () => tooltip.style.opacity = 0);

  // Circles
  nodeEnter.append("circle")
    .attr("r", 0)
    .attr("fill", d => d.data.highlighted ? "#fbbf24" : (d._children ? d.data.color || "#64748b" : "#fff"))
    .attr("stroke", d => d.data.color || "#64748b");

  node.merge(nodeEnter).transition().duration(400)
    .attr("transform", d => `translate(${{d.y}},${{d.x}})`)
    .attr("class", d => "node" + (d.data.highlighted ? " highlighted" : ""));

  node.merge(nodeEnter).select("circle").transition().duration(400)
    .attr("r", d => {{
      if (d.depth === 0) return 10;
      if (d.depth === 1) return 8;
      if (d.depth === 2) return 5;
      return 4;
    }})
    .attr("fill", d => {{
      if (d.data.highlighted) return "#fbbf24";
      if (d._children) return d.data.color || "#64748b";
      return "#fff";
    }})
    .attr("stroke", d => d.data.color || "#64748b");

  // Labels
  nodeEnter.append("text")
    .attr("dy", "0.35em")
    .attr("x", d => d.children || d._children ? -12 : 10)
    .attr("text-anchor", d => d.children || d._children ? "end" : "start")
    .text(d => {{
      const name = d.data.name;
      return name.length > 28 ? name.slice(0, 26) + "…" : name;
    }})
    .style("font-weight", d => d.data.highlighted ? "700" : (d.depth <= 1 ? "600" : "400"))
    .style("fill", d => d.data.highlighted ? "#b45309" : (d.depth === 0 ? "#0f172a" : "#334155"));

  node.merge(nodeEnter).select("text")
    .attr("x", d => d.children || d._children ? -12 : 10)
    .attr("text-anchor", d => d.children || d._children ? "end" : "start")
    .text(d => {{
      const name = d.data.name;
      return name.length > 28 ? name.slice(0, 26) + "…" : name;
    }});

  node.exit().transition().duration(400)
    .attr("transform", d => `translate(${{source.y}},${{source.x}})`).remove();

  nodes.forEach(d => {{ d.x0 = d.x; d.y0 = d.y; }});
}}

function diagonal(d) {{
  return `M${{d.source.y}},${{d.source.x}}C${{(d.source.y + d.target.y)/2}},${{d.source.x}} ${{(d.source.y + d.target.y)/2}},${{d.target.x}} ${{d.target.y}},${{d.target.x}}`;
}}

function toggle(d) {{
  if (d.children) {{ d._children = d.children; d.children = null; }}
  else {{ d.children = d._children; d._children = null; }}
}}

function expandAll() {{
  root.each(d => {{
    if (d._children) {{ d.children = d._children; d._children = null; }}
  }});
  update(root);
}}

function collapseAll() {{
  root.children && root.children.forEach(collapse);
  update(root);
}}

function resetZoom() {{
  svg.transition().duration(400).call(zoom.transform, d3.zoomIdentity.translate(margin.left, margin.top));
}}

function searchNodes(query) {{
  if (!query) {{
    g.selectAll(".node circle").style("opacity", 1);
    g.selectAll(".node text").style("opacity", 1);
    return;
  }}
  const q = query.toLowerCase();
  g.selectAll(".node").each(function(d) {{
    const match = d.data.name.toLowerCase().includes(q);
    d3.select(this).select("circle").style("opacity", match ? 1 : 0.15);
    d3.select(this).select("text").style("opacity", match ? 1 : 0.15)
      .style("font-weight", match ? "700" : "400");
  }});
}}

update(root);

// Auto-fit
svg.call(zoom.transform, d3.zoomIdentity.translate(margin.left + 10, margin.top));
</script>
</body>
</html>
"""
    components.html(html, height=height + 80, scrolling=False)


# ── METHODOLOGY PIPELINE ──────────────────────────────────────────────────────
def render_methodology_pipeline():
    html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #f8fafc; font-family: 'Inter', sans-serif; padding: 20px; }

  .pipeline { display: flex; align-items: stretch; gap: 0; overflow-x: auto; padding: 10px 0; }

  .stage {
    display: flex; flex-direction: column; align-items: center;
    min-width: 130px; flex: 1;
    opacity: 0;
    animation: fadeIn 0.5s ease forwards;
  }
  .stage:nth-child(1)  { animation-delay: 0.1s; }
  .stage:nth-child(2)  { animation-delay: 0.25s; }
  .stage:nth-child(3)  { animation-delay: 0.4s; }
  .stage:nth-child(4)  { animation-delay: 0.55s; }
  .stage:nth-child(5)  { animation-delay: 0.7s; }
  .stage:nth-child(6)  { animation-delay: 0.85s; }
  .stage:nth-child(7)  { animation-delay: 1.0s; }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .arrow { display: flex; align-items: center; color: #94a3b8; font-size: 18px; padding: 0 4px; margin-top: -24px; }

  .box {
    border-radius: 10px;
    padding: 14px 12px;
    text-align: center;
    width: 100%;
    border: 2px solid transparent;
    transition: transform 0.2s, box-shadow 0.2s;
    cursor: default;
  }
  .box:hover { transform: translateY(-4px); box-shadow: 0 8px 20px rgba(0,0,0,0.12); }

  .box .icon { font-size: 1.8rem; margin-bottom: 6px; }
  .box .title { font-size: 0.82rem; font-weight: 700; margin-bottom: 3px; }
  .box .sub   { font-size: 0.7rem; color: #64748b; line-height: 1.4; }

  .tag {
    font-size: 0.65rem; font-weight: 700; letter-spacing: .06em;
    text-transform: uppercase; margin-top: 8px; padding: 3px 8px;
    border-radius: 4px; display: inline-block;
  }

  /* colours */
  .c1 { background: #eff6ff; border-color: #3b82f6; }
  .c1 .title { color: #1d4ed8; }
  .c1 .tag { background: #dbeafe; color: #1d4ed8; }

  .c2 { background: #f0fdf4; border-color: #22c55e; }
  .c2 .title { color: #15803d; }
  .c2 .tag { background: #dcfce7; color: #15803d; }

  .c3 { background: #fdf4ff; border-color: #a855f7; }
  .c3 .title { color: #7e22ce; }
  .c3 .tag { background: #f3e8ff; color: #7e22ce; }

  .c4 { background: #fff7ed; border-color: #f97316; }
  .c4 .title { color: #c2410c; }
  .c4 .tag { background: #ffedd5; color: #c2410c; }

  .c5 { background: #fef2f2; border-color: #ef4444; }
  .c5 .title { color: #b91c1c; }
  .c5 .tag { background: #fee2e2; color: #b91c1c; }

  .c6 { background: #ecfdf5; border-color: #10b981; }
  .c6 .title { color: #065f46; }
  .c6 .tag { background: #d1fae5; color: #065f46; }

  .c7 { background: #f0f9ff; border-color: #0ea5e9; }
  .c7 .title { color: #0369a1; }
  .c7 .tag { background: #e0f2fe; color: #0369a1; }

  .metrics-row {
    display: flex; gap: 12px; margin-top: 24px; flex-wrap: wrap;
  }
  .metric-card {
    flex: 1; min-width: 120px; background: white;
    border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 12px 16px; text-align: center;
    animation: fadeIn 0.5s ease forwards; opacity: 0;
    animation-delay: 1.2s;
  }
  .metric-card .val { font-size: 1.4rem; font-weight: 700; color: #0f172a; }
  .metric-card .lbl { font-size: 0.7rem; color: #94a3b8; margin-top: 2px; }

  .finding-box {
    background: #fffbeb; border: 2px solid #fbbf24; border-radius: 10px;
    padding: 14px 18px; margin-top: 20px;
    animation: fadeIn 0.5s ease forwards; opacity: 0; animation-delay: 1.5s;
  }
  .finding-box .badge {
    font-size: 0.65rem; font-weight: 700; letter-spacing: .06em;
    text-transform: uppercase; background: #fef3c7; color: #b45309;
    padding: 2px 8px; border-radius: 4px; display: inline-block; margin-bottom: 6px;
  }
  .finding-box p { font-size: 0.82rem; color: #374151; line-height: 1.6; margin-top: 4px; }
</style>
</head>
<body>

<div class="pipeline">
  <div class="stage">
    <div class="box c1">
      <div class="icon">🗃️</div>
      <div class="title">Icecat Dataset</div>
      <div class="sub">489,185 B2B products. Stratified sampling to 35,607.</div>
      <span class="tag">Input</span>
    </div>
  </div>
  <div class="arrow">→</div>

  <div class="stage">
    <div class="box c2">
      <div class="icon">🧠</div>
      <div class="title">SBERT Encoding</div>
      <div class="sub">all-MiniLM-L6-v2<br>384-dimensional dense embeddings</div>
      <span class="tag">Embedding</span>
    </div>
  </div>
  <div class="arrow">→</div>

  <div class="stage">
    <div class="box c3">
      <div class="icon">📐</div>
      <div class="title">UMAP Reduction</div>
      <div class="sub">384D → 5D manifold. Preserves local + global structure.</div>
      <span class="tag">Dimensionality</span>
    </div>
  </div>
  <div class="arrow">→</div>

  <div class="stage">
    <div class="box c4">
      <div class="icon">🌳</div>
      <div class="title">Ward AHC</div>
      <div class="sub">Agglomerative clustering. K=411 optimal cut. 100% coverage.</div>
      <span class="tag">Clustering</span>
    </div>
  </div>
  <div class="arrow">→</div>

  <div class="stage">
    <div class="box c5">
      <div class="icon">🤖</div>
      <div class="title">Qwen 2.5 7B</div>
      <div class="sub">Few-shot + RAG Critic + Gating. Labels Root›Parent›Leaf.</div>
      <span class="tag">LLM Naming</span>
    </div>
  </div>
  <div class="arrow">→</div>

  <div class="stage">
    <div class="box c6">
      <div class="icon">🔍</div>
      <div class="title">Critic Review</div>
      <div class="sub">Auto quality gate. 92% acceptance. BERTScore F₁ 0.8231.</div>
      <span class="tag">Validation</span>
    </div>
  </div>
  <div class="arrow">→</div>

  <div class="stage">
    <div class="box c7">
      <div class="icon">🏷️</div>
      <div class="title">Taxonomy</div>
      <div class="sub">4 Roots · 165 Parents · 302 Leaves. NMI 0.7933 vs Icecat.</div>
      <span class="tag">Output</span>
    </div>
  </div>
</div>

<div class="metrics-row">
  <div class="metric-card"><div class="val">0.6565</div><div class="lbl">Silhouette Score</div></div>
  <div class="metric-card"><div class="val">0.7933</div><div class="lbl">NMI vs Icecat</div></div>
  <div class="metric-card"><div class="val">0.8231</div><div class="lbl">BERTScore F₁</div></div>
  <div class="metric-card"><div class="val">100%</div><div class="lbl">Product Coverage</div></div>
  <div class="metric-card"><div class="val">100%</div><div class="lbl">3-Level Compliance</div></div>
  <div class="metric-card"><div class="val">92%</div><div class="lbl">Critic Acceptance</div></div>
</div>

<div class="finding-box">
  <span class="badge">⚡ Emergent Finding</span>
  <p><strong>Home & Office</strong> — discovered as an entirely new 4th root domain, never present in any exemplar or rule. The pipeline independently detected that cable protectors and desk organisers occupy a distinct semantic region from computing devices — a genuine unsupervised cross-domain discovery.</p>
</div>

</body>
</html>
"""
    components.html(html, height=480, scrolling=False)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    "<div style='text-align:center; padding:14px 0 6px 0'>"
    "<span style='font-size:2rem'>🏷️</span><br>"
    "<b style='font-size:1rem; color:#0f172a'>Product Taxonomy</b><br>"
    "<span style='font-size:0.78rem; color:#888'>Master Thesis Demo</span><br>"
    "<span style='font-size:0.75rem; color:#aaa'>Gourav Suresh Jadhav · 2026</span>"
    "</div>",
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🔍 Assign Product", "🌳 Taxonomy Tree"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Development tools
if st.sidebar.checkbox("🛠️ Dev Tools", value=False, key="dev_tools"):
    if st.sidebar.button("🔄 Clear Cache", key="clear_cache"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Cache cleared!")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:0.78rem; color:#64748b; line-height:2.1'>"
    "<b>Pipeline</b><br>"
    "SBERT all-MiniLM-L6-v2 · 384D<br>"
    "UMAP 5D · Ward AHC · 411 clusters<br>"
    "Qwen 2.5 7B + Critic labeling<br>"
    "100% structural compliance"
    "</div>",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:0.7rem; color:#94a3b8; text-align:center; padding:8px 0'>"
    "🎯 <b>Key Achievements</b><br>"
    "• 411 clusters from 35K products<br>"
    "• BERTScore F₁: 0.823<br>"
    "• NMI vs Icecat: 0.793<br>"
    "• 4 root domains discovered"
    "</div>",
    unsafe_allow_html=True
)

# Show last assigned result in sidebar if available
if st.session_state.assignment_result:
    r   = st.session_state.assignment_result
    top = r['candidates'][0]
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<div style='font-size:0.72rem; color:#64748b; margin-bottom:4px'>"
        f"<b>Last assigned</b></div>"
        f"<div style='font-size:0.78rem; color:#0f172a; font-weight:600; "
        f"word-wrap:break-word'>{st.session_state.assignment_title}</div>"
        f"<div style='font-size:0.72rem; color:{r['color']}; margin-top:2px'>"
        f"{r['stars']} {r['tier']}</div>"
        f"<div style='font-size:0.72rem; color:#64748b; margin-top:4px'>"
        f"{top['root']} → {top['parent']} → {top['leaf']}</div>",
        unsafe_allow_html=True
    )


if page == "🔍 Assign Product":

    thesis_banner()
    st.title("🔍 Assign a New Product")

    # How it works
    st.markdown(
        "<div style='display:flex; gap:0; margin-bottom:20px; "
        "border:1px solid #e2e8f0; border-radius:10px; overflow:hidden'>"
        "<div style='flex:1; padding:14px 18px; background:#f0f4ff; border-right:1px solid #e2e8f0'>"
        "<div style='font-size:0.7rem; color:#6366f1; font-weight:700; text-transform:uppercase; letter-spacing:.06em'>Step 1</div>"
        "<div style='font-size:0.9rem; font-weight:600; color:#0f172a; margin-top:3px'>Enter product title</div>"
        "<div style='font-size:0.76rem; color:#64748b; margin-top:2px'>Any product — seen or unseen</div></div>"
        "<div style='flex:1; padding:14px 18px; background:#f0fdf4; border-right:1px solid #e2e8f0'>"
        "<div style='font-size:0.7rem; color:#16a34a; font-weight:700; text-transform:uppercase; letter-spacing:.06em'>Step 2</div>"
        "<div style='font-size:0.9rem; font-weight:600; color:#0f172a; margin-top:3px'>Pipeline runs</div>"
        "<div style='font-size:0.76rem; color:#64748b; margin-top:2px'>SBERT → UMAP → Ward centroid</div></div>"
        "<div style='flex:1; padding:14px 18px; background:#faf5ff'>"
        "<div style='font-size:0.7rem; color:#7c3aed; font-weight:700; text-transform:uppercase; letter-spacing:.06em'>Step 3</div>"
        "<div style='font-size:0.9rem; font-weight:600; color:#0f172a; margin-top:3px'>Taxonomy path returned</div>"
        "<div style='font-size:0.76rem; color:#64748b; margin-top:2px'>Root → Parent → Leaf · confidence scored</div></div>"
        "</div>",
        unsafe_allow_html=True
    )

    col_input, col_examples = st.columns([2, 1])

    with col_input:
        title = st.text_input(
            "Product Title",
            value=st.session_state.title,
            placeholder="e.g. Cisco Catalyst 9200 48-Port Gigabit Network Switch"
        )
        description = st.text_area(
            "Description *(optional — improves accuracy)*",
            value=st.session_state.description,
            placeholder="e.g. Managed switch, 48 GbE ports, PoE+, stackable",
            height=68
        )
        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            assign_btn = st.button("Assign →", type="primary", use_container_width=True)
        with col_btn2:
            surprise_btn = st.button("🎲 Surprise Me", use_container_width=True)

    with col_examples:
        st.markdown("**Quick Examples**")
        examples = [
            {
                "title": "HP LaserJet Pro M404n Monochrome Laser Printer",
                "description": "Professional monochrome laser printer, 38ppm, 1200dpi, built-in ethernet, 256MB RAM, 250-sheet tray"
            },
            {
                "title": "Samsung 970 EVO Plus 1TB NVMe M.2 Internal SSD",
                "description": "NVMe M.2 solid state drive, 1TB capacity, sequential read 3500MB/s, sequential write 3300MB/s, V-NAND technology"
            },
            {
                "title": "Logitech MX Master 3 Wireless Mouse",
                "description": "Advanced wireless mouse, 4000 DPI sensor, ergonomic design, USB-C charging, multi-device connectivity, Bluetooth"
            },
            {
                "title": "Apple iPhone 15 Pro 256GB Titanium",
                "description": "Smartphone with A17 Pro chip, 6.1 inch Super Retina XDR display, 48MP camera system, titanium design, USB-C"
            },
            {
                "title": "Canon EOS R50 Mirrorless Camera Body Black",
                "description": "Mirrorless digital camera, 24.2MP APS-C CMOS sensor, 4K video, Dual Pixel CMOS AF II, Wi-Fi, Bluetooth, lightweight body"
            },
            {
                "title": "Microsoft Office 365 Business Premium 1 Year License",
                "description": "Cloud-based productivity suite license, includes Word Excel PowerPoint Teams, 1TB OneDrive storage, 1 year subscription"
            },
            {
                "title": "APC Smart-UPS 1500VA LCD 230V Rack Mount",
                "description": "Uninterruptible power supply 1500VA 1000W, rack mount 2U, LCD display, pure sinewave output, network management card slot"
            },
            {
                "title": "Cisco Catalyst 2960-X 48 Port Gigabit Switch",
                "description": "Managed network switch, 48 Gigabit Ethernet ports, 4 SFP uplinks, LAN Base software, stackable, rack mountable 1U"
            },
        ]
        selected_example = st.selectbox(
            "Choose an example:",
            ["Select an example..."] + [ex['title'] for ex in examples],
            key="example_select"
        )
        col_clear, col_select = st.columns(2)
        with col_select:
            if selected_example != "Select an example...":
                selected = next((x for x in examples if x['title'] == selected_example), None)
                if selected is not None:
                    st.session_state.title = selected['title']
                    st.session_state.description = selected['description']

                st.session_state.title = selected_example
                st.session_state.description = ""
                st.success(f"✅ Loaded: {selected_example}")
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.title = ""
                st.session_state.description = ""
                st.session_state.assignment_result = None
                st.rerun()

    # Surprise Me — fixed with session state
    if surprise_btn:
        tax_temp = load_full_taxonomy()
        sample   = tax_temp[tax_temp['Examples_List'].apply(len) > 0].sample(1).iloc[0]
        if sample['Examples_List']:
            chosen = random.choice(sample['Examples_List'])
            st.session_state.title = chosen
            st.session_state.description = ""
            st.rerun()

    st.markdown("---")

    # Run assignment
    if assign_btn and title.strip():
        ph    = st.empty()
        steps = ["Product text", "SBERT 384D", "UMAP 5D", "Ward centroid", "Taxonomy path"]
        for i in range(len(steps) + 1):
            html = ("<div style='display:flex; align-items:center; gap:4px; "
                    "margin-bottom:12px; justify-content:center'>")
            for j, s in enumerate(steps):
                active = j < i
                bg = "#1e40af" if active else "#eff6ff"
                fg = "#ffffff" if active else "#1e40af"
                html += (
                    f"<span style='background:{bg}; color:{fg}; "
                    f"border:1px solid #bfdbfe; border-radius:6px; "
                    f"padding:5px 12px; font-size:0.78rem; font-weight:600'>{s}</span>"
                )
                if j < len(steps) - 1:
                    html += "<span style='color:#94a3b8'>→</span>"
            html += "</div>"
            ph.markdown(html, unsafe_allow_html=True)
            time.sleep(0.25)

        result = assign_to_taxonomy(title, description)
        st.session_state.assignment_result = result
        st.session_state.assignment_title  = title
        ph.empty()

    # Show result (from session state — persists across reruns)
    if st.session_state.assignment_result:
        result = st.session_state.assignment_result
        top    = result['candidates'][0]
        c      = result['color']
        fg, bg = ROOT_COLORS.get(top['root'], ("#374151", "#f3f4f6"))

        # Banner
        st.markdown(
            f"<div style='background:{bg}; border:2px solid {fg}40; "
            f"padding:20px 24px; border-radius:10px; margin-bottom:16px'>"
            f"<div style='font-size:0.73rem; color:{fg}; font-weight:700; "
            f"text-transform:uppercase; letter-spacing:.06em; margin-bottom:8px'>Assigned to Taxonomy</div>"
            f"<div style='font-size:1.35rem; font-weight:700; color:#0f172a'>"
            f"{top['root']}"
            f"<span style='color:#94a3b8; font-weight:400'> → </span>"
            f"{top['parent']}"
            f"<span style='color:#94a3b8; font-weight:400'> → </span>"
            f"<span style='color:{fg}'>{top['leaf']}</span>"
            f"</div>"
            f"<div style='margin-top:10px; display:flex; align-items:center; gap:10px'>"
            f"{root_badge(top['root'])}"
            f"<span style='font-size:0.86rem; color:{c}; font-weight:600'>"
            f"{result['stars']} {result['tier']}</span>"
            f"<span style='font-size:0.76rem; color:#94a3b8'>{result['note']}</span>"
            f"</div></div>",
            unsafe_allow_html=True
        )

        # Alternative candidates
        st.markdown("##### Alternative candidates")
        max_dist = max(cand['distance'] for cand in result['candidates']) or 1
        medals   = ["🥇", "🥈", "🥉"]
        for i, cand in enumerate(result['candidates']):
            bar_pct = max(4, int((1 - cand['distance'] / max_dist) * 100))
            bar_col = fg if i == 0 else "#cbd5e1"
            label   = f"{cand['root']} → {cand['parent']} → {cand['leaf']}"
            st.markdown(
                f"<div style='margin-bottom:10px'>"
                f"<div style='font-size:0.82rem; color:#374151; margin-bottom:3px'>{medals[i]} {label}</div>"
                f"<div style='background:#f1f5f9; border-radius:4px; height:10px; width:100%'>"
                f"<div style='background:{bar_col}; height:10px; border-radius:4px; width:{bar_pct}%'></div></div>"
                f"<div style='font-size:0.72rem; color:#94a3b8; margin-top:2px'>distance: {cand['distance']}</div></div>",
                unsafe_allow_html=True
            )

        # Tip to see in tree
        st.markdown("---")
        st.info(
            f"💡 Switch to **🌳 Taxonomy Tree** in the sidebar to see "
            f"**{top['leaf']}** highlighted in the full tree."
        )

        if c == "#b91c1c":
            st.warning("**Very low confidence** — this product may be underrepresented. Flagged as a potential new taxonomy node.")
        elif c == "#1a7f37":
            st.success("**High confidence** — product mapped clearly to an existing taxonomy node.")
        else:
            st.warning("**Moderate confidence** — result is plausible but flagged for review.")

    elif assign_btn:
        st.warning("Please enter a product title.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TAXONOMY TREE (NEW!)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌳 Taxonomy Tree":

    thesis_banner()
    st.title("🌳 Interactive Taxonomy Tree")
    st.markdown(
        "**411 clusters** across **4 root domains** discovered entirely unsupervised. "
        "Click any node to expand / collapse. Hover for details."
    )

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Root Domains",      "4")
    c2.metric("Parent Categories", "165")
    c3.metric("Leaf Categories",   "302")
    c4.metric("Total Clusters",    "411")
    st.markdown("---")

    tax = load_full_taxonomy()

    # If a product was assigned, offer to highlight it
    highlight_root   = None
    highlight_parent = None
    highlight_leaf   = None
    highlight_path   = None

    if st.session_state.assignment_result:
        top = st.session_state.assignment_result['candidates'][0]
        highlight_root   = top['root']
        highlight_parent = top['parent']
        highlight_leaf   = top['leaf']
        highlight_path   = f"{highlight_root} > {highlight_parent} > {highlight_leaf}"

        fg, bg = ROOT_COLORS.get(highlight_root, ("#374151", "#f3f4f6"))
        st.markdown(
            f"<div style='background:{bg}; border:1.5px solid {fg}40; border-radius:8px; "
            f"padding:10px 16px; margin-bottom:12px; display:flex; align-items:center; gap:12px'>"
            f"<span style='font-size:1.2rem'>📍</span>"
            f"<div>"
            f"<div style='font-size:0.72rem; color:{fg}; font-weight:700; text-transform:uppercase; "
            f"letter-spacing:.05em'>Last assigned product is highlighted below</div>"
            f"<div style='font-size:0.9rem; font-weight:600; color:#0f172a; margin-top:2px'>"
            f"{st.session_state.assignment_title}</div>"
            f"<div style='font-size:0.78rem; color:#64748b'>"
            f"{highlight_root} → {highlight_parent} → <b style='color:{fg}'>{highlight_leaf}</b></div>"
            f"</div></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background:#f1f5f9; border-radius:8px; padding:10px 16px; "
            "margin-bottom:12px; font-size:0.82rem; color:#64748b'>"
            "💡 Assign a product first on the <b>🔍 Assign Product</b> page "
            "and its node will be automatically highlighted here."
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("**Click nodes to expand · Hover for details · Use controls top-right to expand/collapse all**")

    # Use cached tree data for faster loading
    base_tree = build_tree_cache()
    if highlight_path:
        tree_data = build_tree_json(
            load_full_taxonomy(),
            highlight_root=highlight_root,
            highlight_parent=highlight_parent,
            highlight_leaf=highlight_leaf
        )
    else:
        tree_data = base_tree

    render_d3_tree(tree_data, height=680, highlight_path=highlight_path)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EXPLORE TAXONOMY (unchanged but improved)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗂️ Explore Taxonomy":

    thesis_banner()
    st.title("🗂️ Explore the Taxonomy")
    st.markdown(
        "**411 clusters** — discovered unsupervised from 35,607 Icecat products "
        "and labeled by Qwen 2.5 7B with no human annotation."
    )
    st.markdown("---")

    tax = load_full_taxonomy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Clusters",    "411")
    c2.metric("Root Categories",   str(tax['Root'].nunique()))
    c3.metric("Parent Categories", str(tax['Parent'].nunique()))
    c4.metric("Leaf Categories",   str(tax['Leaf'].nunique()))
    st.markdown("---")

    tab1, tab2 = st.tabs(["🔎 Search Clusters", "🌲 Browse by Root"])

    with tab1:
        search_query = st.text_input(
            "Search by product type, keyword, or category name",
            placeholder="e.g. printer · switch · antivirus · camera · UPS",
            key="search"
        )

        if search_query.strip():
            q    = search_query.strip().lower()
            mask = (
                tax['Examples'].str.lower().str.contains(q, na=False) |
                tax['Keywords_Clean'].str.lower().str.contains(q, na=False) |
                tax['Leaf'].str.lower().str.contains(q, na=False) |
                tax['Parent'].str.lower().str.contains(q, na=False) |
                tax['Root'].str.lower().str.contains(q, na=False)
            )
            results = tax[mask].copy()

            if len(results) == 0:
                st.warning(f"No clusters found for **'{search_query}'**")
            else:
                st.success(
                    f"**{len(results)} cluster(s)** matched across "
                    f"**{results['Root'].nunique()} root categories**"
                )
                for _, row in results.iterrows():
                    qc        = quality_color(row.get('Quality', ''))
                    score     = row.get('Coherence_Score', 0)
                    score_str = f"{score:.3f}" if isinstance(score, float) else ""
                    with st.expander(f"{row['Root']}  ›  {row['Parent']}  ›  **{row['Leaf']}**"):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            rfg, rbg = ROOT_COLORS.get(row['Root'], ("#374151", "#f3f4f6"))
                            st.markdown(
                                f"<div style='background:{rbg}; padding:10px 14px; border-radius:6px; "
                                f"font-size:0.88rem; line-height:2'>"
                                f"<b style='color:{rfg}'>{row['Root']}</b><br>"
                                f"&nbsp;&nbsp;▸ {row['Parent']}<br>"
                                f"&nbsp;&nbsp;&nbsp;&nbsp;▸ <b style='color:#0066cc'>{row['Leaf']}</b></div>"
                                f"<p style='margin-top:8px; font-size:0.78rem; color:#555'>"
                                f"🔑 <code>{row['Keywords_Clean']}</code></p>"
                                f"<span style='background:{qc}; color:#fff; font-size:0.72rem; "
                                f"padding:2px 8px; border-radius:4px'>Coherence {score_str}</span>",
                                unsafe_allow_html=True
                            )
                        with col2:
                            st.markdown("<p style='font-size:0.82rem; font-weight:600; margin-bottom:4px'>Products in this cluster</p>", unsafe_allow_html=True)
                            for ex in row['Examples_List'][:6]:
                                st.markdown(
                                    f"<p style='font-size:0.82rem; color:#444; margin:2px 0; "
                                    f"padding-left:8px; border-left:3px solid #dee2e6'>· {ex}</p>",
                                    unsafe_allow_html=True
                                )
        else:
            st.markdown(
                "<p style='font-size:0.85rem; color:#888'>Try: "
                "<b>printer</b> · <b>switch</b> · <b>laptop</b> · "
                "<b>antivirus</b> · <b>monitor</b> · <b>UPS</b></p>",
                unsafe_allow_html=True
            )

    with tab2:
        root_counts = (
            tax.groupby('Root').size()
            .reset_index(name='n')
            .sort_values('n', ascending=False)
        )

        rcols = st.columns(len(root_counts))
        for i, (_, rrow) in enumerate(root_counts.iterrows()):
            rfg, rbg = ROOT_COLORS.get(rrow['Root'], ("#374151", "#f3f4f6"))
            with rcols[i]:
                st.markdown(
                    f"<div style='background:{rbg}; border:1.5px solid {rfg}40; "
                    f"border-radius:8px; padding:12px; text-align:center; margin-bottom:12px'>"
                    f"<div style='font-size:0.92rem; font-weight:700; color:{rfg}'>{rrow['Root']}</div>"
                    f"<div style='font-size:1.4rem; font-weight:700; color:#0f172a; margin:4px 0'>{rrow['n']}</div>"
                    f"<div style='font-size:0.72rem; color:#94a3b8'>clusters</div></div>",
                    unsafe_allow_html=True
                )

        st.markdown("---")
        selected_root = st.selectbox(
            "Select a root category to explore",
            ["— select —"] + root_counts['Root'].tolist(),
            format_func=lambda x: x if x == "— select —" else
                f"{x}  ({root_counts[root_counts['Root']==x]['n'].values[0]} clusters)"
        )

        if selected_root != "— select —":
            fr       = tax[tax['Root'] == selected_root]
            rfg, rbg = ROOT_COLORS.get(selected_root, ("#374151", "#f3f4f6"))
            pc       = (fr.groupby('Parent').size().reset_index(name='n').sort_values('n', ascending=False))
            st.caption(f"{selected_root} · {len(fr)} clusters · {fr['Parent'].nunique()} parent categories")

            selected_parent = st.selectbox(
                "Select a parent category",
                ["All"] + pc['Parent'].tolist(),
                format_func=lambda x: x if x == "All" else
                    f"{x}  ({pc[pc['Parent']==x]['n'].values[0]} clusters)"
            )

            filtered = fr if selected_parent == "All" else fr[fr['Parent'] == selected_parent]

            if selected_parent == "All":
                for parent_name, group in filtered.groupby('Parent'):
                    tree_html = (
                        f"<div style='background:#fafafa; border:1px solid #e2e8f0; "
                        f"border-radius:8px; padding:12px 16px; margin-bottom:8px'>"
                        f"<div style='font-size:0.92rem; font-weight:700; color:#0f172a; margin-bottom:6px'>"
                        f"▸ {parent_name} <span style='font-size:0.75rem; color:#94a3b8; font-weight:400'>({len(group)} clusters)</span></div>"
                    )
                    for _, row in group.iterrows():
                        score     = row.get('Coherence_Score', 0)
                        score_str = f"{score:.3f}" if isinstance(score, float) else ""
                        tree_html += (
                            f"<div style='padding-left:18px; font-size:0.84rem; color:#0066cc; margin:3px 0'>"
                            f"└─ {row['Leaf']} <span style='color:#d1d5db; font-size:0.72rem'>{score_str}</span></div>"
                        )
                    tree_html += "</div>"
                    st.markdown(tree_html, unsafe_allow_html=True)
            else:
                for _, row in filtered.iterrows():
                    qc        = quality_color(row.get('Quality', ''))
                    score     = row.get('Coherence_Score', 0)
                    score_str = f"{score:.3f}" if isinstance(score, float) else ""
                    with st.expander(f"**{row['Leaf']}**  ·  Cluster {int(row['Cluster_ID'])}  ·  Coherence {score_str}"):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.markdown(
                                f"<p style='font-size:0.8rem; color:#555'>🔑 <code>{row['Keywords_Clean']}</code></p>"
                                f"<span style='background:{qc}; color:#fff; font-size:0.72rem; padding:2px 8px; "
                                f"border-radius:4px'>{row.get('Quality','')}</span>",
                                unsafe_allow_html=True
                            )
                        with col2:
                            for ex in row['Examples_List'][:5]:
                                st.markdown(
                                    f"<p style='font-size:0.82rem; color:#444; margin:2px 0; "
                                    f"padding-left:8px; border-left:3px solid #dee2e6'>· {ex}</p>",
                                    unsafe_allow_html=True
                                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EVALUATION RESULTS (with charts)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Evaluation Results":

    thesis_banner()
    st.title("📊 Evaluation Results")
    st.markdown(
        "Multi-dimensional validation — internal cluster geometry, "
        "external alignment with the Icecat reference taxonomy, and LLM labeling quality."
    )
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Input Products",  "489,185")
    c2.metric("After Balancing", "35,607")
    c3.metric("Clusters (K)",    "411")
    c4.metric("Coverage",        "100%")
    c5.metric("NMI vs Icecat",   "0.7933")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🔬 Cluster Quality", "🤖 LLM Labeling", "🏗️ Taxonomy Structure"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Internal Validity")
            internal_df = pd.DataFrame({
                'Metric' : ['Silhouette', 'Davies-Bouldin', 'CH Index (÷1000)', 'Mean Purity'],
                'Score'  : [0.6565, 0.3992, 36.26, 0.7428],
                'Target' : [0.5, 0.6, 20.0, 0.6],
            })
            # Bar chart
            chart_data = pd.DataFrame({
                'Achieved': [0.6565, 0.3992, 0.7428],
                'Benchmark': [0.5, 0.6, 0.6]
            }, index=['Silhouette', 'DB Index', 'Purity'])
            st.bar_chart(chart_data, height=200)
            st.dataframe(
                pd.DataFrame({
                    'Metric' : ['Silhouette Coefficient', 'Davies-Bouldin Index', 'Calinski-Harabasz', 'Mean Cluster Purity'],
                    'Score'  : [0.6565, 0.3992, 36264.14, 0.7428],
                    'Verdict': ['✅ Strong separation', '✅ Low overlap', '✅ High dispersion', '✅ 74% agreement']
                }),
                use_container_width=True, hide_index=True
            )

        with col2:
            st.markdown("##### External Alignment vs. Icecat")
            ext_chart = pd.DataFrame({
                'Score': [0.7933, 0.8204, 0.7679, 0.4094, 1.0]
            }, index=['NMI', 'Homogeneity', 'Completeness', 'ARI', 'Coverage'])
            st.bar_chart(ext_chart, height=200)
            st.dataframe(
                pd.DataFrame({
                    'Metric' : ['NMI / V-Measure', 'Homogeneity', 'Completeness', 'ARI', 'Coverage'],
                    'Score'  : [0.7933, 0.8204, 0.7679, 0.4094, 1.0000],
                    'Verdict': ['79% expert structure recovered', 'Clusters semantically pure',
                                'Expert categories represented', 'Expected at K=411', 'Every product assigned']
                }),
                use_container_width=True, hide_index=True
            )

        st.markdown("---")
        st.markdown("##### Algorithm Comparison")
        algo_df = pd.DataFrame({
            'Algorithm' : ['K-Means', 'HDBSCAN (EOM)', 'Ward AHC ✅'],
            'Silhouette': [0.5821, 0.6712, 0.6565],
            'DB Index'  : [0.5614, 0.3241, 0.3992],
            'Coverage'  : ['100%', '84.2%', '100%'],
        })
        algo_chart = pd.DataFrame({
            'Silhouette': [0.5821, 0.6712, 0.6565],
            'DB Index (inverted)': [1-0.5614, 1-0.3241, 1-0.3992]
        }, index=['K-Means', 'HDBSCAN', 'Ward AHC'])
        st.bar_chart(algo_chart, height=220)
        st.dataframe(algo_df, use_container_width=True, hide_index=True)
        st.caption("Ward selected — near-HDBSCAN geometric quality with guaranteed 100% product coverage.")

    with tab2:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("##### LLM Labeling — Method Progression")
            llm_chart = pd.DataFrame({
                'BERTScore F₁': [0.7784, 0.8103, 0.8231],
                '3-Level Compliance (÷100)': [0.613, 0.946, 1.0]
            }, index=['Zero-Shot', 'Few-Shot', 'Few-Shot + Critic'])
            st.bar_chart(llm_chart, height=250)
            st.dataframe(
                pd.DataFrame({
                    'Method'            : ['Zero-Shot', 'Few-Shot', 'Few-Shot + Critic + Gate'],
                    'BERTScore F₁'      : [0.7784, 0.8103, 0.8231],
                    '3-Level Compliance': ['61.3%', '94.6%', '100%'],
                    'Note'              : ['Baseline', 'Exemplars enforce format', '✅ Final — all constraints met']
                }),
                use_container_width=True, hide_index=True
            )
        with col2:
            st.metric("BERTScore F₁",       "0.8231")
            st.metric("3-Level Compliance", "100%")
            st.metric("Critic Acceptance",  "92.0%")
            st.metric("Good / Excellent",   "62.6%")

        st.markdown("---")
        st.markdown(
            "<div style='background:#fffbeb; border:2px solid #fbbf24; "
            "border-radius:10px; padding:18px 22px'>"
            "<div style='font-size:0.74rem; color:#b45309; font-weight:700; "
            "text-transform:uppercase; letter-spacing:.06em; margin-bottom:6px'>Emergent Finding</div>"
            "<div style='font-size:1.05rem; font-weight:700; color:#0f172a'>Home & Office — discovered as a 4th root domain</div>"
            "<div style='font-size:0.86rem; color:#555; margin-top:6px'>"
            "This root was <b>never present in any few-shot exemplar or post-processing rule</b>. "
            "The pipeline independently recognised that cable protectors and desk organisers "
            "occupy a distinct semantic region from computing devices — purely from embedding "
            "space geometry. Genuine unsupervised cross-domain discovery."
            "</div></div>",
            unsafe_allow_html=True
        )

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(
                pd.DataFrame({
                    'Root Category'         : ['Electronics', 'Office Supplies', 'Software', 'Home & Office (emergent)'],
                    'Clusters'              : [270, 83, 57, 1],
                    'Share'                 : ['65.7%', '20.2%', '13.9%', '0.2%']
                }),
                use_container_width=True, hide_index=True
            )
            root_chart = pd.DataFrame({'Clusters': [270, 83, 57, 1]},
                                       index=['Electronics', 'Office Supplies', 'Software', 'Home & Office'])
            st.bar_chart(root_chart, height=200)
        with col2:
            st.dataframe(
                pd.DataFrame({
                    'Property'  : ['Total clusters', 'Unique Roots', 'Unique Parents',
                                   'Unique Leaves', 'Parent:Leaf ratio', 'Structural compliance'],
                    'Value'     : ['411', '4', '165', '302', '1:1.83', '100%']
                }),
                use_container_width=True, hide_index=True
            )

    st.markdown("---")
    # Footer removed as requested
    st.markdown("", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — METHODOLOGY (NEW!)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Methodology":

    thesis_banner()
    st.title("⚙️ Methodology Pipeline")
    st.markdown(
        "End-to-end pipeline — from raw Icecat product data to a fully labeled "
        "3-level taxonomy with confidence scoring."
    )
    st.markdown("---")

    st.markdown("##### Pipeline Architecture")
    render_methodology_pipeline()

    st.markdown("---")
    st.markdown("##### Design Decisions")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "<div style='background:#f0f9ff; border-left:4px solid #0ea5e9; padding:14px 18px; "
            "border-radius:0 8px 8px 0; margin-bottom:12px'>"
            "<b style='color:#0369a1'>Why SBERT over TF-IDF?</b><br>"
            "<span style='font-size:0.83rem; color:#334155; margin-top:4px; display:block'>"
            "TF-IDF treats words as independent features. SBERT captures semantic similarity — "
            "'network switch' and 'managed switch' are close in embedding space, not orthogonal."
            "</span></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='background:#f0fdf4; border-left:4px solid #22c55e; padding:14px 18px; "
            "border-radius:0 8px 8px 0; margin-bottom:12px'>"
            "<b style='color:#15803d'>Why UMAP over PCA?</b><br>"
            "<span style='font-size:0.83rem; color:#334155; margin-top:4px; display:block'>"
            "PCA is linear and loses non-linear manifold structure. UMAP preserves both local "
            "neighborhood and global cluster separation — critical for Ward linkage quality."
            "</span></div>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            "<div style='background:#fdf4ff; border-left:4px solid #a855f7; padding:14px 18px; "
            "border-radius:0 8px 8px 0; margin-bottom:12px'>"
            "<b style='color:#7e22ce'>Why Ward AHC over K-Means?</b><br>"
            "<span style='font-size:0.83rem; color:#334155; margin-top:4px; display:block'>"
            "K-Means assumes spherical clusters. Ward AHC minimizes within-cluster variance at "
            "each merge — produces more coherent, varied-shape clusters. And unlike HDBSCAN, "
            "100% coverage is guaranteed."
            "</span></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='background:#fff7ed; border-left:4px solid #f97316; padding:14px 18px; "
            "border-radius:0 8px 8px 0; margin-bottom:12px'>"
            "<b style='color:#c2410c'>Why Qwen 2.5 7B + Critic?</b><br>"
            "<span style='font-size:0.83rem; color:#334155; margin-top:4px; display:block'>"
            "Zero-shot LLMs fail structurally (61.3% compliance). Few-shot exemplars fix format. "
            "The RAG Critic adds a quality gate — 92% auto-accepted, 8% regenerated — reaching "
            "100% structural compliance and BERTScore 0.8231."
            "</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("##### Confidence Scoring Logic")
    st.markdown(
        "<div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; "
        "padding:16px 20px; font-size:0.84rem; color:#334155'>"
        "Classification confidence is determined by the <b>margin between the top-1 and top-2 "
        "centroid distances</b> in UMAP space. A large margin = the product sits clearly closest "
        "to one cluster. A small margin = product is ambiguous between two candidate nodes.<br><br>"
        "<b>Thresholds</b><br>"
        "★★★★★ High · margin ≥ 0.0008 — auto-assign<br>"
        "★★★☆☆ Medium · margin ≥ 0.0003 — auto-assign, log for review<br>"
        "★★☆☆☆ Low · margin ≥ 0.0001 — flag for manual review<br>"
        "★☆☆☆☆ Very Low — potential new category node"
        "</div>",
        unsafe_allow_html=True
    )



