from _resiliency import PDC_PRIO_UNCHANGED, PDC_PRIO_FALSE, PDC_PRIO_TRUE, place_pdcs_resiliently, prefix_tree_from_pmu_paths, vec_idx_from_pmu_name
from placement_pdc import place_pdcs_greedy
from graph_model import create_graph
from visualizer import draw_graph
import networkx as nx
import numpy as np
import re

def main():
    # PMUs
    N, M = 4, 4
    v = np.ones((N+M, 1), dtype=int)
    R = build_simple_R(N, M)
    
    # Graph
    SEED = 42                   # Altri seed: 4, 67
    CANDIDATES = 1*(N+M)        # 2*(N+M) or 3*(N+M)
    MAX_CC_LINKS = N            # None or N
    EDGE_LAT_MIN = 1
    EDGE_LAT_MAX = 3
    
    # Algo
    MAX_LAT = 500
    CONSIDER_NEIGH = True
    PAR_CHI = False
    PDC_PRIO = PDC_PRIO_UNCHANGED
    OVERLAPPED_LINKS = [[("N16", "CC"), ("N13", "N9")]]
    DEBUG = True
    
    # All
    set = {"N": N, "M": M, "v": v, "R": R, "SEED": SEED, "CANDIDATES": CANDIDATES, "MAX_CC_LINKS": MAX_CC_LINKS, "EDGE_LAT_MIN": EDGE_LAT_MIN, "EDGE_LAT_MAX": EDGE_LAT_MAX, "MAX_LAT": MAX_LAT, "CONSIDER_NEIGH": CONSIDER_NEIGH, "PAR_CHI": PAR_CHI, "PDC_PRIO": PDC_PRIO}

    G = create_graph(seed=SEED, num_pmus=N+M, num_candidates=CANDIDATES, cc_max_links=MAX_CC_LINKS, edge_latency_min=EDGE_LAT_MIN, edge_latency_max=EDGE_LAT_MAX)
    set_simple_red_role(G, R)
    
    draw_graph(G, output_path="output-test/0.png")
    
    pos = None
    try:
        pos = nx.nx_pydot.pydot_layout(G, prog="dot")
    except:
        pos = nx.spring_layout(G, seed=42)

    (pdcs, pmu_paths, _) = place_pdcs_greedy(G, max_latency=MAX_LAT, flag_splitting=False)
    draw_graph(G, pdcs, pmu_paths, max_latency=MAX_LAT, output_path="output-test/1.0-graph-dario.png", pos=pos, params=set)
    T = build_tree(pmu_paths, R)
    draw_graph(T, pdcs, pmu_paths, max_latency=MAX_LAT, output_path="output-test/2.0-tree-dario.png", view_mode=3, params=set)

    (pdcs, pmu_paths) = place_pdcs_resiliently(G, max_latency=MAX_LAT, essentialPMUs=N, v=v, R=R, parchi_constraint=PAR_CHI, cc_successors_constraint=(not CONSIDER_NEIGH), pdc_prio=PDC_PRIO, overlapped_links=OVERLAPPED_LINKS, debug=DEBUG)
    draw_graph(G, pdcs, pmu_paths, max_latency=MAX_LAT, output_path="output-test/1.1b-graph-resilient.png", pos=pos, params=set)
    T = build_tree(pmu_paths, R)
    draw_graph(T, pdcs, pmu_paths, max_latency=MAX_LAT, output_path="output-test/2.1b-tree-resilient.png", view_mode=3, params=set)

def build_simple_R(N: int, M: int):
    top    = np.hstack([np.zeros((N, N)), np.eye(N, M)])
    bottom = np.hstack([np.eye(M, N), np.zeros((M, M))])
    return np.vstack([top, bottom])

def build_tree(pmu_paths, R) -> nx.DiGraph:
    T = prefix_tree_from_pmu_paths(pmu_paths)
    set_simple_red_role(T, R)
    return T

def set_simple_red_role(G: nx.Graph, R):
    for n, d in G.nodes(data=True):
        if not re.match(r"[r]?PMU(\d+)", n):
            if d.get("role", "candidate") != "PMU":
                continue
        
        i = vec_idx_from_pmu_name(n)
        
        r_of = d.get("r_of", [])
        for j in range(i+1, R.shape[0]):
            if R[i, j] == 1:
                r_of.append(f"PMU{j+1}")
        
        d["r_of"] = r_of

def find_seed():
    N, M = 4, 4
    
    for i in range(1000):
        G = create_graph(seed=i, num_pmus=N+M, num_candidates=2*(N+M))
        
        links = 0
        for n in G:
            links += G.number_of_edges("CC", n)
            
        if links > 4:
            continue
        
        print(f"FOUND. Seed: {i}")

if __name__ == "__main__":
    main()