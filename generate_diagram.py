"""
Generate Architecture_Diagram.png for the Supply Chain Disruption Response Agent.
Uses matplotlib to create a professional system architecture diagram.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Colour palette ──────────────────────────────────────────────────
COL_PERCEPTION  = "#4A90D9"   # Blue  – knowledge sources
COL_REASONING   = "#2ECC71"   # Green – LangGraph nodes
COL_EXECUTION   = "#E74C3C"   # Red   – action targets
COL_HUMAN       = "#F39C12"   # Orange – human in the loop
COL_EDGE        = "#555555"   # Dark grey – arrows
COL_BG          = "#FAFAFA"   # Light background
COL_CLUSTER_BG  = "#FFFFFF"   # Cluster background

def draw_box(ax, x, y, w, h, text, color, fontsize=7.5, bold=False, alpha=0.85):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="#333333",
        linewidth=1.2, alpha=alpha, zorder=3
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color="white",
            zorder=4, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label="", color=COL_EDGE, style="-|>",
               linestyle="-", connectionstyle="arc3,rad=0"):
    """Draw an arrow between two points with optional label."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color,
        linewidth=1.3, linestyle=linestyle,
        connectionstyle=connectionstyle,
        mutation_scale=12, zorder=2
    )
    ax.add_patch(arrow)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.01, my, label, fontsize=6, color=color,
                ha="left", va="center", style="italic", zorder=5)

def draw_cluster(ax, x, y, w, h, title, color):
    """Draw a labelled cluster rectangle."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015",
        facecolor=COL_CLUSTER_BG, edgecolor=color,
        linewidth=2, linestyle="--", alpha=0.3, zorder=1
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h + 0.015, title,
            ha="center", va="bottom", fontsize=9,
            fontweight="bold", color=color, zorder=5)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(16, 11))
    fig.patch.set_facecolor(COL_BG)
    ax.set_facecolor(COL_BG)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    # Title
    ax.text(0.5, 1.01, "Supply Chain Disruption Response Agent — System Architecture",
            ha="center", va="bottom", fontsize=14, fontweight="bold", color="#222")

    # ── Clusters ────────────────────────────────────────────────────
    # Perception layer (left column)
    draw_cluster(ax, 0.0, 0.22, 0.22, 0.72, "Perception Layer\n(Knowledge Sources)", COL_PERCEPTION)
    # Reasoning layer (center)
    draw_cluster(ax, 0.28, 0.05, 0.42, 0.89, "Reasoning Layer — LangGraph StateGraph", COL_REASONING)
    # Execution layer (right column)
    draw_cluster(ax, 0.76, 0.22, 0.22, 0.40, "Execution Layer\n(Action Targets)", COL_EXECUTION)

    # ── Perception boxes (left) ─────────────────────────────────────
    bw, bh = 0.18, 0.065
    sources = [
        (0.11, 0.84, "Disruption\nNews API"),
        (0.11, 0.74, "SOP Wiki\n(Markdown / RAG)"),
        (0.11, 0.64, "Inventory & PO\nDatabase (PostgreSQL)"),
        (0.11, 0.54, "Disruption\nHistory Log"),
        (0.11, 0.44, "Supplier Docs\n(ChromaDB)"),
        (0.11, 0.34, "Supplier Pricing\nAPI"),
    ]
    for sx, sy, stxt in sources:
        draw_box(ax, sx, sy, bw, bh, stxt, COL_PERCEPTION, fontsize=6.5)

    # ── Reasoning nodes (center column) ─────────────────────────────
    nw, nh = 0.20, 0.058
    nodes = [
        (0.49, 0.86, "1. classify_disruption\nLLM + fetch_alerts + SOP"),
        (0.49, 0.74, "2. assess_impact\nquery_inventory_db + history"),
        (0.49, 0.62, "3. find_alternatives\nsearch_supplier_docs + pricing"),
        (0.49, 0.50, "4. calculate_exposure\ncalculate_financial_impact"),
        (0.49, 0.38, "5. generate_plan\ndraft_response_plan (LLM)"),
        (0.49, 0.26, "6. human_review\n[INTERRUPT — await approval]"),
        (0.49, 0.14, "7. execute_actions\nsend_notification + update_PO"),
    ]
    for i, (nx, ny, ntxt) in enumerate(nodes):
        col = COL_HUMAN if i == 5 else COL_REASONING
        draw_box(ax, nx, ny, nw, nh, ntxt, col, fontsize=6.5, bold=(i == 5))

    # ── Execution boxes (right) ─────────────────────────────────────
    targets = [
        (0.87, 0.52, "Slack / Email\nNotifications"),
        (0.87, 0.38, "ERP System\n(PO Updates)"),
    ]
    for tx, ty, ttxt in targets:
        draw_box(ax, tx, ty, bw, bh, ttxt, COL_EXECUTION, fontsize=6.5)

    # ── Human actor (right side) ────────────────────────────────────
    draw_box(ax, 0.87, 0.26, bw, bh, "Procurement\nManager (Human)", COL_HUMAN, fontsize=6.5, bold=True)

    # ── START / END ─────────────────────────────────────────────────
    ax.annotate("START", xy=(0.49, 0.93), fontsize=8, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.04", fc="#333", ec="#333", alpha=0.9),
                color="white", zorder=5)
    ax.annotate("END", xy=(0.49, 0.065), fontsize=8, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.04", fc="#333", ec="#333", alpha=0.9),
                color="white", zorder=5)

    # ── Arrows: Perception → Reasoning ──────────────────────────────
    # News API → classify
    draw_arrow(ax, 0.20, 0.84, 0.39, 0.86, color=COL_PERCEPTION)
    # SOP Wiki → classify
    draw_arrow(ax, 0.20, 0.74, 0.39, 0.86, color=COL_PERCEPTION)
    # Inventory DB → assess_impact
    draw_arrow(ax, 0.20, 0.64, 0.39, 0.74, color=COL_PERCEPTION)
    # History → assess_impact
    draw_arrow(ax, 0.20, 0.54, 0.39, 0.74, color=COL_PERCEPTION)
    # Supplier Docs → find_alternatives
    draw_arrow(ax, 0.20, 0.44, 0.39, 0.62, color=COL_PERCEPTION)
    # Supplier Pricing → find_alternatives
    draw_arrow(ax, 0.20, 0.34, 0.39, 0.62, color=COL_PERCEPTION)

    # ── Arrows: Graph flow (center) ─────────────────────────────────
    # START → classify
    draw_arrow(ax, 0.49, 0.90, 0.49, 0.89)
    # classify → assess
    draw_arrow(ax, 0.49, 0.831, 0.49, 0.769, label="severity\nrouting")
    # assess → find_alternatives
    draw_arrow(ax, 0.49, 0.711, 0.49, 0.649)
    # find_alternatives → calculate_exposure (if found)
    draw_arrow(ax, 0.45, 0.591, 0.45, 0.529, label="alternatives\nfound")
    # find_alternatives → generate_plan (no alternatives - skip calc)
    draw_arrow(ax, 0.53, 0.591, 0.53, 0.409, label="no alternatives",
               linestyle="--", connectionstyle="arc3,rad=-0.15")
    # calculate → generate_plan
    draw_arrow(ax, 0.49, 0.471, 0.49, 0.409)
    # generate_plan → human_review
    draw_arrow(ax, 0.49, 0.351, 0.49, 0.289)
    # human_review → execute (approve)
    draw_arrow(ax, 0.45, 0.231, 0.45, 0.169, label="approve")
    # human_review → END (reject)
    draw_arrow(ax, 0.53, 0.231, 0.53, 0.09, label="reject",
               linestyle="--", connectionstyle="arc3,rad=-0.15")
    # human_review → generate_plan (modify loop)
    draw_arrow(ax, 0.39, 0.28, 0.33, 0.38, label="modify\n(iter < 3)",
               connectionstyle="arc3,rad=0.4", linestyle="--", color=COL_HUMAN)
    # execute → END
    draw_arrow(ax, 0.49, 0.111, 0.49, 0.09)

    # ── Arrows: Reasoning → Execution ───────────────────────────────
    # execute_actions → Slack
    draw_arrow(ax, 0.59, 0.15, 0.78, 0.52, color=COL_EXECUTION,
               connectionstyle="arc3,rad=-0.2")
    # execute_actions → ERP
    draw_arrow(ax, 0.59, 0.13, 0.78, 0.38, color=COL_EXECUTION,
               connectionstyle="arc3,rad=-0.15")

    # ── Arrows: Human ↔ human_review ────────────────────────────────
    draw_arrow(ax, 0.78, 0.26, 0.59, 0.26, label="decision", color=COL_HUMAN)

    # ── Legend ──────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=COL_PERCEPTION, label="Knowledge Sources"),
        mpatches.Patch(color=COL_REASONING, label="LangGraph Nodes"),
        mpatches.Patch(color=COL_HUMAN, label="Human-in-the-Loop"),
        mpatches.Patch(color=COL_EXECUTION, label="Action Targets"),
    ]
    ax.legend(handles=legend_patches, loc="lower left", fontsize=7,
              framealpha=0.9, edgecolor="#ccc")

    # ── Save ────────────────────────────────────────────────────────
    fig.savefig("c:/Users/Hassaan/Desktop/Capstone Lab/Architecture_Diagram.png",
                dpi=200, bbox_inches="tight", facecolor=COL_BG)
    plt.close(fig)
    print("Architecture_Diagram.png generated successfully.")


if __name__ == "__main__":
    main()
