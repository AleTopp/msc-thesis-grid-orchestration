import json

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
from pathlib import Path
import numpy as np
import re

colors = [
    "crimson", "darkgreen", "royalblue", "goldenrod",
    "purple", "darkorange", "deeppink", "teal", "brown"
]

def draw_graph(G: nx.Graph | nx.DiGraph, pdcs=None, paths=None, max_latency=None, output_path: Path | None = None, pos = None, view_mode: int = 1):
    if pdcs is None:
        pdcs = set()

    if output_path is None:
        output_path = Path("runtime_results") / "graph.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 10))

    try:
        if pos is None:
            pos = nx.nx_pydot.pydot_layout(G, prog="dot")
    except Exception:
        print("⚠️ Error with pydot layout, using spring layout instead.")
        pos = nx.spring_layout(G, seed=42)

    edge_labels = nx.get_edge_attributes(G, "latency")
    node_colors = []
    node_labels = {}
    node_edgecolors = []
    red_pair = 1

    for n in G.nodes:
        role = G.nodes[n].get("role")
        r_of = G.nodes[n].get("r_of", [])
        label = n

        if n in pdcs:
            color = "orange"
            label += f"\n{G.nodes[n].get('processing', 0)}"
            edge_color = "black"
        elif role == "CC":
            color = "red"
            label += "\n(CC)"
            edge_color = "black"
        elif role == "PMU":
            color = "lightgreen"
            label += "\n(PMU)"
            edge_color = "black"
        else:
            color = "lightblue"
            edge_color = "gray"
            
        if len(r_of) > 0:
            G.nodes[n]["pair"] = red_pair
            for nn in r_of:
                G.nodes[nn]["pair"] = red_pair
            
            red_pair += 1

        node_colors.append(color)
        node_labels[n] = label
        node_edgecolors.append(edge_color)
        
    for i, (n, pair) in enumerate(G.nodes.data("pair", default=None)):
        if pair:
            node_colors[i] = colors[(pair-1) % len(colors)]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        edgecolors=node_edgecolors,
        node_size=1100,
        linewidths=1.8,
    )

    nx.draw_networkx_edges(G, pos, width=1.2, edge_color="lightgray")
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.5)

    if paths and view_mode == 1:
        # Highlight each PMU -> CC path with a different color
        for i, (pmu, data) in enumerate(paths.items()):
            path = data["path"]
            color = colors[i % len(colors)]
            edges = list(zip(path, path[1:]))

            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges,
                width=2.8,
                edge_color=color,
            )
    elif paths and (view_mode == 2 or view_mode == 3):
        ax = plt.gca()
        
        paths_per_edge = {}
        for (pmu, data) in paths.items():
            path = data["path"]
            for u, v in zip(path, path[1:]):
                if (u, v) not in paths_per_edge.keys():
                    paths_per_edge[(u, v)] = set()
                    
                paths_per_edge[(u, v)].add(pmu)
        
        for (a, b), pmus in paths_per_edge.items():
            n_paths = len(pmus)
            
            offset_step = 3
            offsets = np.linspace(
                -offset_step * (n_paths - 1) / 2,
                offset_step * (n_paths - 1) / 2,
                n_paths,
            )
            
            for pmu, offset in zip(pmus, offsets):
                idx = vec_idx_from_pmu_name(pmu)
                color = colors[idx]
                if view_mode == 3 and G.nodes[pmu].get("pair", None) is not None:
                    color = colors[G.nodes[pmu].get("pair", -1) - 1]
                
                _draw_offset_edge(ax, pos, a, b, color, offset)

    if paths:
        # Text box con i ritardi PMU -> CC (invariato)
        all_pmus = [n for n in G.nodes if G.nodes[n].get("role") == "PMU"]
        text = "Latency PMU → CC:\n"
        if max_latency is not None:
            text += f"Max required latency: {float(max_latency):.2f} ms\n"
        else:
            text += "Max required latency: N/A\n"

        for pmu in all_pmus:
            if pmu in paths:
                delay = paths[pmu].get("delay", None)
                if delay is not None:
                    text += f"{pmu} → CC: {float(delay):.2f} ms ✔️\n"
                else:
                    text += f"{pmu} → CC: delay N/A\n"
            else:
                text += f"{pmu} → CC: no path available ✗\n"

        plt.gcf().text(
            0.05, 0.85, text,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
        )

    # Legend
    legend_elements = [
        Patch(facecolor="red", edgecolor="black", label="CC"),
        Patch(facecolor="lightgreen", edgecolor="black", label="PMU"),
        Patch(facecolor="orange", edgecolor="black", label="PDC (assigned)"),
        Patch(facecolor="lightblue", edgecolor="gray", label="Other Nodes"),
    ]
    plt.legend(handles=legend_elements, loc="lower left", fontsize=9, frameon=True)

    plt.title("Graph with role and selected path", fontsize=12)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    backend = matplotlib.get_backend().lower()
    if backend not in {"agg", "pdf", "ps", "svg"}:
        plt.show(block=False)
    else:
        print(
            f"ℹ️  Matplotlib backend '{backend}' is non-interactive.\n"
            f"   Graph saved to \"{output_path}\"."
        )

    plt.close()


def _draw_offset_path(ax, pos, path, color, offset, lw=2.8, alpha=0.9, node_r=10):
    """
    Disegna un percorso spostando lateralmente ogni arco di `offset` unità,
    così percorsi che condividono lo stesso arco rimangono visibili in parallelo.
    """
    for u, v in zip(path, path[1:]):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
 
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length == 0:
            continue
 
        # Vettore perpendicolare normalizzato
        ux, uy = dx / length, dy / length
        perp_x, perp_y = -uy, ux
 
        # Punti di inizio/fine traslati lateralmente
        sx = x1 + perp_x * offset + ux * node_r * 1.2
        sy = y1 + perp_y * offset + uy * node_r * 1.2
        ex = x2 + perp_x * offset - ux * node_r * 1.2
        ey = y2 + perp_y * offset - uy * node_r * 1.2
 
        ax.annotate(
            "",
            xy=(ex, ey),
            xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                alpha=alpha,
                mutation_scale=14,
            ),
        )

def _draw_offset_edge(ax, pos, u, v, color, offset, lw=2.8, alpha=0.9, node_r=10):
    """
    Disegna un arco spostandolo lateralmente di `offset` unità,
    così che rimangano visibili in parallelo.
    """
    x1, y1 = pos[u]
    x2, y2 = pos[v]

    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length == 0:
        return

    # Vettore perpendicolare normalizzato
    ux, uy = dx / length, dy / length
    perp_x, perp_y = -uy, ux

    # Punti di inizio/fine traslati lateralmente
    sx = x1 + perp_x * offset + ux * node_r * 1.2
    sy = y1 + perp_y * offset + uy * node_r * 1.2
    ex = x2 + perp_x * offset - ux * node_r * 1.2
    ey = y2 + perp_y * offset - uy * node_r * 1.2

    ax.annotate(
        "",
        xy=(ex, ey),
        xytext=(sx, sy),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            alpha=alpha,
            mutation_scale=14,
        ),
    )
    
def vec_idx_from_pmu_name(pmu_name: str) -> int:
    m = re.match(r"[r]?PMU(\d+)", pmu_name)
    if m:
        return int(m.group(1)) - 1
    return -1