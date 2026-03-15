import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 13)
ax.set_ylim(0, 8)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Input box
ax.add_patch(FancyBboxPatch((4.0, 6.6), 5.0, 0.9,
                             boxstyle="round,pad=0.1",
                             facecolor="#eafaf1", edgecolor="#2a7a4b", linewidth=2))
ax.text(6.5, 7.1, "Cluster Profile",
        color="#1a1a1a", fontsize=10, fontweight="bold", ha="center")
ax.text(6.5, 6.8, "Keywords  ·  Example Products  ·  Cluster ID",
        color="#555555", fontsize=8, ha="center")

branches = [
    {
        "x": 1.6, "color": "#2c5f8a",
        "method": "Method 1",
        "name": "Zero-Shot",
        "desc": "Role-based prompt\nSingle category name\nNo examples provided",
        "out": '"Optical Discs\nand Drives"',
    },
    {
        "x": 6.5, "color": "#6b3fa0",
        "method": "Method 2",
        "name": "Few-Shot",
        "desc": "Three exemplar pairs\nHierarchical path format\nPattern-based learning",
        "out": '"Electronics > Storage\nDevices > Optical Disc Drives"',
    },
    {
        "x": 11.4, "color": "#b03a2e",
        "method": "Method 3",
        "name": "Critic-Refinement",
        "desc": "Method 2 path as input\nHS/UNSPSC alignment\nSpecificity critique",
        "out": '"Electronics > Storage Devices\n> Optical Disc Drives > CD/DVD Drives"',
    },
]

for b in branches:
    x, color = b["x"], b["color"]

    # Arrow from input
    ax.annotate("", xy=(x, 5.7), xytext=(6.5, 6.6),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                connectionstyle="arc3,rad=0.0"))

    # Method header box
    ax.add_patch(FancyBboxPatch((x - 1.4, 4.7), 2.8, 0.85,
                                boxstyle="round,pad=0.1",
                                facecolor="#f7f7f7", edgecolor=color, linewidth=2))
    ax.text(x, 5.18, b["method"], color=color,
            fontsize=8, fontweight="bold", ha="center")
    ax.text(x, 4.88, b["name"], color="#1a1a1a",
            fontsize=9, fontweight="bold", ha="center")

    # Description box
    ax.add_patch(FancyBboxPatch((x - 1.4, 3.1), 2.8, 1.45,
                                boxstyle="round,pad=0.1",
                                facecolor="#fafafa", edgecolor="#cccccc", linewidth=1.2))
    ax.text(x, 3.83, b["desc"], color="#444444",
            fontsize=7.8, ha="center", va="center", linespacing=1.6)

    # Arrow to output
    ax.annotate("", xy=(x, 2.8), xytext=(x, 3.1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))

    # Output box
    ax.add_patch(FancyBboxPatch((x - 1.4, 1.6), 2.8, 1.05,
                                boxstyle="round,pad=0.1",
                                facecolor="#f7f7f7", edgecolor=color, linewidth=1.8))
    ax.text(x, 2.13, b["out"], color="#1a1a1a",
            fontsize=7.5, ha="center", va="center", linespacing=1.4)

# Method 2 feeds into Method 3
ax.annotate("", xy=(10.0, 5.12), xytext=(7.9, 5.12),
            arrowprops=dict(arrowstyle="-|>", color="#888888",
                            lw=1.5, linestyle="dashed"))
ax.text(8.95, 5.35, "feeds into", color="#888888",
        fontsize=7.5, ha="center", style="italic")

# Title and caption
ax.text(6.5, 7.75,
        "Figure 3.7: Three-Stage LLM Prompting Strategy Pipeline",
        color="#1a1a1a", fontsize=10, fontweight="bold", ha="center")
ax.text(6.5, 0.2,
        "Method 3 (Critic-Refinement) receives Method 2 output as input for iterative path refinement.",
        color="#555555", fontsize=8, ha="center", style="italic")

plt.tight_layout(pad=1.0)
plt.savefig("Figure_3_7_Prompting_Strategies.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.show()
print("Saved: Figure_3_7_Prompting_Strategies.png")
