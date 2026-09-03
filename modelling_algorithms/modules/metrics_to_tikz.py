"""
metrics_to_tikz.py

Legge un file metrics.json generato da `test3.py` e produce file .png
contenenti grafici per le metriche richieste.

Esempio d'uso:
python metrics_to_tikz.py \
  --metrics runtime_results/2026-08-06/15-30-45.526540/metrics.json \
  --outdir runtime_results/2026-08-06/15-30-45.526540
"""
from __future__ import annotations

import json
import os
import argparse
from typing import List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tikzplotlib

def compute_boxplot_stats(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    arr = np.array(values, dtype=float)
    q1 = float(np.percentile(arr, 25))
    q2 = float(np.percentile(arr, 50))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    # whiskers using 1.5*IQR rule
    lw = float(np.min(arr[arr >= q1 - 1.5 * iqr])) if arr.size else q1
    uw = float(np.max(arr[arr <= q3 + 1.5 * iqr])) if arr.size else q3
    return {
        "lower_whisker": lw,
        "lower_quartile": q1,
        "median": q2,
        "upper_quartile": q3,
        "upper_whisker": uw,
    }

def compute_group_means(alg_groups: Dict[str, List[List[float]]]) -> Dict[str, List[float]]:
    """Compute the mean per algorithm and x-position from grouped raw observations."""
    means: Dict[str, List[float]] = {}
    for algo, groups in alg_groups.items():
        avg_vals: List[float] = []
        for group in groups:
            if not group:
                avg_vals.append(np.nan)
            else:
                avg_vals.append(float(np.mean(group)))
        means[algo] = avg_vals
    return means


def write_grouped_lineplot_png(filename: str, title: str, xlabels: List[str], alg_groups: Dict[str, List[List[float]]], ylabel: str, xlabel: Optional[str] = None, y_limit: Optional[float] = None, log_y: bool = False):
    """Create a lineplot of the mean value for each algorithm across sizes."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    algos = list(alg_groups.keys())
    n_x = len(xlabels)
    fig_width = max(10, n_x * 1.8)
    fig_height = 6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    colors = plt.get_cmap('tab10').colors

    for a_idx, algo in enumerate(algos):
        means = compute_group_means({algo: alg_groups[algo]})[algo]
        x = np.arange(1, n_x + 1)
        valid = ~np.isnan(np.asarray(means, dtype=float))
        if np.any(valid):
            ax.plot(x[valid], np.asarray(means, dtype=float)[valid], label=algo, color=colors[a_idx % len(colors)], marker='o', linewidth=2)

    ax.set_xticks(np.arange(1, n_x + 1))
    ax.set_xticklabels(xlabels, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if y_limit is not None:
        ax.set_ylim(bottom=0, top=y_limit)
    if log_y:
        ax.set_yscale('log')
    if algos:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(4, len(algos)))
    plt.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    code = tikzplotlib.get_tikz_code(axis_width="10in", axis_height="6in")
    code = code.replace("begin{tikzpicture}", "begin{tikzpicture}[scale=\\tikzscale]", 1)
    code = code.replace("#", "\\#")
    with open(filename + ".tex", "w") as f:
        f.write(code)
    plt.close(fig)


def write_grouped_boxplot_png(filename: str, title: str, xlabels: List[str], alg_groups: Dict[str, List[List[float]]], ylabel: str, xlabel: Optional[str] = None, horiz_line: Optional[float] = None, y_limit: Optional[float] = None, log_y: bool = False, mean_lines: Optional[Dict[str, List[float]]] = None):
    """Create grouped boxplot PNG: one box per algorithm at each x (size)."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    algos = list(alg_groups.keys())
    n_algs = len(algos)
    n_x = len(xlabels)

    # positions
    total_width = 0.8
    width = total_width / max(1, n_algs)
    offsets = [ (i - (n_algs-1)/2.0) * width for i in range(n_algs) ]
    # Make figures wider so boxplots are readable when many sizes/algorithms
    fig_width = max(10, n_x * 2.0)
    fig_height = 6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)

    colors = plt.get_cmap('tab10').colors
    legend_handles = []

    for a_idx, algo in enumerate(algos):
        groups = alg_groups[algo]
        data = []
        positions = []
        for x_idx in range(n_x):
            grp = groups[x_idx] if x_idx < len(groups) else []
            # Matplotlib requires at least one value; use nan to skip
            if not grp:
                data.append([np.nan])
            else:
                data.append(list(grp))
            positions.append(x_idx + 1 + offsets[a_idx])

        # filter positions and data where not all nan
        plot_data = []
        plot_pos = []
        for d, p in zip(data, positions):
            if all(np.isnan(d)):
                # skip empty
                continue
            plot_data.append(d)
            plot_pos.append(p)

        if plot_data:
            bp = ax.boxplot(plot_data, positions=plot_pos, widths=width*0.9, patch_artist=True, manage_ticks=False)
            color = colors[a_idx % len(colors)]
            for box in bp['boxes']:
                box.set(facecolor=color, alpha=0.6)
            for whisker in bp['whiskers']:
                whisker.set(color=color)
            for median in bp['medians']:
                median.set(color='black')
            # create legend handle
            legend_handles.append(Line2D([0], [0], color=color, lw=6, alpha=0.6))

        if mean_lines is not None and algo in mean_lines:
            mean_vals = mean_lines[algo]
            x_vals = np.arange(1, n_x + 1)
            valid = ~np.isnan(np.asarray(mean_vals, dtype=float))
            if np.any(valid):
                ax.plot(x_vals[valid] + offsets[a_idx], np.asarray(mean_vals, dtype=float)[valid], color=colors[a_idx % len(colors)], linestyle='-', linewidth=2.5, marker='o', label=algo)

    ax.set_xticks([i+1 for i in range(n_x)])
    ax.set_xticklabels(xlabels, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if horiz_line is not None:
        ax.hlines(horiz_line, 0.5, n_x+0.5, colors='red', linestyles='--')

    if y_limit is not None:
        ax.set_ylim(bottom=0, top=y_limit)

    if log_y:
        ax.set_yscale('log')

    if legend_handles:
        ax.legend(legend_handles, algos, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(4, len(algos)))

    plt.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    # Keep the TikZ aspect ratio and physical size aligned with the PNG.
    # Without these options tikzplotlib/PGFPlots uses its own default axis size.
    code = tikzplotlib.get_tikz_code(
        axis_width="10in",
        axis_height="6in",
    )
    code = code.replace("begin{tikzpicture}", "begin{tikzpicture}[scale=\\tikzscale]", 1)
    code = code.replace("#", "\\#")
    with open(filename + ".tex", "w") as f:
        f.write(code)
    plt.close(fig)


def safe_extend(d: Dict, key, default):
    if key not in d:
        d[key] = default


def main():
    parser = argparse.ArgumentParser(description="Convert metrics.json to TikZ/PGFPlots .tex files")
    parser.add_argument("--metrics", default="runtime_results/2026-08-06/15-30-45.526540/metrics.json")
    parser.add_argument("--outdir", default="runtime_results/2026-08-06/15-30-45.526540/tikz")
    args = parser.parse_args()

    metrics_path = args.metrics
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    with open(metrics_path, "r") as f:
        data = json.load(f)

    # data is expected as a list of dicts (values of metrics_dict)
    # Organize: algorithm -> size -> aggregated lists
    store: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    for entry in data:
        size = int(entry.get("size"))
        algo = str(entry.get("algorithm", "unknown"))
        store.setdefault(algo, {})
        store[algo].setdefault(size, {
            "pdcs_nums": [],
            "pdcs_norm": [],
            "execution_times": [],
            "node_jain_indexes": [],
            "edge_jain_indexes": [],
            "ttf": [],
            "node_faults_to_failure": [],
            "edge_faults_to_failure": [],
            "latency_ePMU": [],
            "latency_rPMU": [],
            "flows_after_node_abs": [],
            "flows_after_node_norm": [],
            "flows_after_edge_abs": [],
            "flows_after_edge_norm": [],
        })

        s = store[algo][size]
        s["pdcs_nums"].extend(entry.get("pdcs_nums", []))
        # normalized by number of candidates (size)
        s["pdcs_norm"].extend([v / size for v in entry.get("pdcs_nums", []) if size != 0])
        s["execution_times"].extend(entry.get("execution_times", []))
        s["node_jain_indexes"].extend(entry.get("node_jain_indexes", []))
        s["edge_jain_indexes"].extend(entry.get("edge_jain_indexes", []))
        # Treat null (None) TTF and fault-to-failure values as 2*size + 1 per user request
        for key, fallback_key in [
            ("times_to_failure", "ttf"),
            ("node_faults_to_failure", "node_faults_to_failure"),
            ("edge_faults_to_failure", "edge_faults_to_failure"),
        ]:
            for v in entry.get(key, []):
                if v is None:
                    s[fallback_key].append(2 * size + 1)
                else:
                    s[fallback_key].append(v)
        # latencies stored in a nested structure under latency_distribution
        latdist = entry.get("latency_distribution", {})
        s["latency_ePMU"].extend(latdist.get("ePMUs", []))
        s["latency_rPMU"].extend(latdist.get("rPMUs", []))
        # flows after node failure: independent_data_flows_after_1_node_fail
        s["flows_after_node_abs"].extend(entry.get("independent_data_flows_after_1_node_fail", []))
        # normalize by size/2 as requested
        denom = size / 2 if size != 0 else 1
        s["flows_after_node_norm"].extend([v / denom * 100.0 for v in entry.get("independent_data_flows_after_1_node_fail", [])])
        # flows after edge failure
        s["flows_after_edge_abs"].extend(entry.get("independent_data_flows_after_1_edge_fail", []))
        s["flows_after_edge_norm"].extend([v / denom * 100.0 for v in entry.get("independent_data_flows_after_1_edge_fail", [])])

    # Collect global sizes and algorithms, grouping all "4_k(...)" variants together
    orig_algos = sorted(store.keys())
    all_sizes = sorted({sz for algo in store.values() for sz in algo.keys()})
    xlabels = [str(sz) for sz in all_sizes]

    # map display name -> list of original algos
    origs_by_display: Dict[str, List[str]] = {}
    for a in orig_algos:
        if a.startswith("4_k("):
            disp = "4_k-shortest-candidates"
        else:
            disp = a
        origs_by_display.setdefault(disp, []).append(a)

    all_algos = sorted(origs_by_display.keys())

    # helper to build alg_groups for a metric key by aggregating original algos for each display name
    def collect_metric(metric_key: str):
        alg_groups: Dict[str, List[List[float]]] = {}
        for disp, origs in origs_by_display.items():
            groups: List[List[float]] = []
            for sz in all_sizes:
                combined: List[float] = []
                for orig in origs:
                    sizes_dict = store.get(orig, {})
                    combined.extend(sizes_dict.get(sz, {}).get(metric_key, []))
                groups.append(combined)
            alg_groups[disp] = groups
        return alg_groups

    # latency ePMU
    alg_groups_e = collect_metric("latency_ePMU")
    png_dir = os.path.join(outdir, "png")
    os.makedirs(png_dir, exist_ok=True)
    write_grouped_boxplot_png(os.path.join(png_dir, "latency_ePMU_by_size.png"),
                              title="",#"Latency ePMU by size",
                              xlabels=xlabels,
                              alg_groups=alg_groups_e,
                              ylabel="ePMU latency (ms)",
                              xlabel="graph size (PMU no.)")

    alg_groups_r = collect_metric("latency_rPMU")
    write_grouped_boxplot_png(os.path.join(png_dir, "latency_rPMU_by_size.png"),
                              title="",#"Latency rPMU by size",
                              xlabels=xlabels,
                              alg_groups=alg_groups_r,
                              ylabel="rPMU latency (ms)",
                              xlabel="graph size (PMU no.)")

    # PDC distribution raw and normalized
    alg_groups_pdcs = collect_metric("pdcs_nums")
    write_grouped_boxplot_png(os.path.join(png_dir, "pdcs_nums_by_size.png"),
                              title="",#"PDCs number by size",
                              xlabels=xlabels,
                              alg_groups=alg_groups_pdcs,
                              ylabel="# of placed PDCs",
                              xlabel="graph size (PMU no.)")

    alg_groups_pdcs_norm = collect_metric("pdcs_norm")
    write_grouped_boxplot_png(os.path.join(png_dir, "pdcs_norm_by_size.png"),
                              title="",#"PDCs / num_candidates by size",
                              xlabels=xlabels,
                              alg_groups=alg_groups_pdcs_norm,
                              ylabel="# of placed PDCs / candidates",
                              xlabel="graph size (PMU no.)")

    # execution time
    alg_groups_time = collect_metric("execution_times")
    execution_time_means = compute_group_means(alg_groups_time)
    write_grouped_boxplot_png(os.path.join(png_dir, "execution_time_by_size.png"),
                              title="",#"Execution time by size",
                              xlabels=xlabels,
                              alg_groups=alg_groups_time,
                              ylabel="execution time (ms)",
                              xlabel="graph size (PMU no.)",
                              log_y=True,
                              mean_lines=execution_time_means)
    write_grouped_lineplot_png(os.path.join(png_dir, "execution_time_mean_by_size.png"),
                              title="",#"Mean execution time by size",
                              xlabels=xlabels,
                              alg_groups=alg_groups_time,
                              ylabel="mean execution time (ms)",
                              xlabel="graph size (PMU no.)",
                              log_y=True)

    # jain
    alg_groups_node_jain = collect_metric("node_jain_indexes")
    write_grouped_boxplot_png(os.path.join(png_dir, "node_jain_index_by_size.png"),
                              title="",#"Node Jain index by size",
                              xlabels=xlabels,
                              alg_groups=alg_groups_node_jain,
                              ylabel="node Jain index",
                              xlabel="graph size (PMU no.)")

    alg_groups_edge_jain = collect_metric("edge_jain_indexes")
    write_grouped_boxplot_png(os.path.join(png_dir, "edge_jain_index_by_size.png"),
                              title="",#"Edge Jain index by size",
                              xlabels=xlabels,
                              alg_groups=alg_groups_edge_jain,
                              ylabel="edge Jain index",
                              xlabel="graph size (PMU no.)")

    # time to failure and analogous fault-to-failure metrics
    alg_groups_ttf = collect_metric("ttf")
    write_grouped_boxplot_png(os.path.join(png_dir, "ttf_by_size.png"),
                              title="",#"Time To Failure by size",
                              xlabels=xlabels,
                              alg_groups=alg_groups_ttf,
                              ylabel="# failures to lose observability",
                              xlabel="graph size (PMU no.)",
                              horiz_line=1)

    alg_groups_node_faults_ttf = collect_metric("node_faults_to_failure")
    write_grouped_boxplot_png(os.path.join(png_dir, "node_faults_to_failure_by_size.png"),
                              title="",#"Node faults to failure by size",
                              xlabels=xlabels,
                              alg_groups=alg_groups_node_faults_ttf,
                              ylabel="# node faults to lose observability",
                              xlabel="graph size (PMU no.)",
                              horiz_line=1)

    alg_groups_edge_faults_ttf = collect_metric("edge_faults_to_failure")
    write_grouped_boxplot_png(os.path.join(png_dir, "edge_faults_to_failure_by_size.png"),
                              title="",#"Edge faults to failure by size",
                              xlabels=xlabels,
                              alg_groups=alg_groups_edge_faults_ttf,
                              ylabel="# edge faults to lose observability",
                              xlabel="graph size (PMU no.)",
                              horiz_line=1)

    # flows after 1st failure: absolute and normalized
    alg_groups_flows_abs = collect_metric("flows_after_node_abs")
    max_line = max((sz / 2 for sz in all_sizes), default=None)
    write_grouped_boxplot_png(os.path.join(png_dir, "flows_after_1node_abs_by_size.png"),
                              title="",#"Flows after 1 node fail (abs) by size",
                              xlabels=xlabels,
                              alg_groups=alg_groups_flows_abs,
                              ylabel="# independent flows after 1 node fail",
                              xlabel="graph size (PMU no.)",
                              horiz_line=max_line)

    alg_groups_flows_norm = collect_metric("flows_after_node_norm")
    write_grouped_boxplot_png(os.path.join(png_dir, "flows_after_1node_pct_by_size.png"),
                              title="",#"Flows after 1 node fail (% of size/2) by size",
                              xlabels=xlabels,
                              alg_groups=alg_groups_flows_norm,
                              ylabel="% of max flows after 1 node fail",
                              xlabel="graph size (PMU no.)",
                              y_limit=105)

    # flows after 1 edge fail (absolute and normalized)
    alg_groups_flows_edge_abs = collect_metric("flows_after_edge_abs")
    write_grouped_boxplot_png(os.path.join(png_dir, "flows_after_1edge_abs_by_size.png"),
                              title="",#"Flows after 1 edge fail (abs) by size",
                              xlabels=xlabels,
                              alg_groups=alg_groups_flows_edge_abs,
                              ylabel="# independent flows after 1 edge fail",
                              xlabel="graph size (PMU no.)",
                              horiz_line=max_line)

    alg_groups_flows_edge_norm = collect_metric("flows_after_edge_norm")
    write_grouped_boxplot_png(os.path.join(png_dir, "flows_after_1edge_pct_by_size.png"),
                              title="",#"Flows after 1 edge fail (% of size/2) by size",
                              xlabels=xlabels,
                              alg_groups=alg_groups_flows_edge_norm,
                              ylabel="% of max flows after 1 edge fail",
                              xlabel="graph size (PMU no.)",
                              y_limit=105)

    print(f"Produced TikZ files in: {outdir}")


if __name__ == "__main__":
    main()
