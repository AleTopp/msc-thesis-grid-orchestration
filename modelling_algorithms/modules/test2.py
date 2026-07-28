from graph_model import DEFAULT_NODE_LATENCY, EDGE_LATENCY, NODE_LATENCY, NODE_REDUNDANT_OF, ROLE_PMU, create_graph
from visualizer import draw_graph, get_layout
from placement_pdc import place_pdcs_greedy
from resiliency import PDC_PRIO_FALSE, PDC_PRIO_TRUE, PDC_PRIO_UNCHANGED, place_pdcs_resiliently, vec_idx_from_pmu_name, calc_path_cost
from resiliency2 import *
import networkx as nx
import numpy as np
import re

G: nx.Graph = None

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
    PDC_PRIO = PDC_PRIO_FALSE
    DEBUG = True
    
    # All
    set = {"N": N, "M": M, "v": v, "R": R, "SEED": SEED, "CANDIDATES": CANDIDATES, "MAX_CC_LINKS": MAX_CC_LINKS, "EDGE_LAT_MIN": EDGE_LAT_MIN, "EDGE_LAT_MAX": EDGE_LAT_MAX, "MAX_LAT": MAX_LAT, "CONSIDER_NEIGH": CONSIDER_NEIGH, "PAR_CHI": PAR_CHI, "PDC_PRIO": PDC_PRIO}

    global G
    G = create_graph(seed=SEED, num_pmus=N+M, num_candidates=CANDIDATES, cc_max_links=MAX_CC_LINKS, edge_latency_min=EDGE_LAT_MIN, edge_latency_max=EDGE_LAT_MAX)
    set_simple_red_role(G, R)
    
    pos = get_layout(G)
    
    if False:
        G.remove_edge("PMU1", "N4")
        G.remove_edge("PMU5", "N4")
        G.add_node("N9", **{NODE_LATENCY: DEFAULT_NODE_LATENCY})
        G.add_edge("PMU1", "N9", **{EDGE_LATENCY: 2})
        G.add_edge("PMU5", "N9", **{EDGE_LATENCY: 2})
        G.add_edge("N4", "N9", **{EDGE_LATENCY: 2})
        pos["N9"] = (pos["N4"][0] - 30, pos["N4"][1] - 50)

    out_dir = "output-test-4b"
    
    draw_graph(G, max_latency=MAX_LAT, output_path=f"{out_dir}/0_graph.png", pos=pos, params=set)
    
    # === From resiliency ===
    (pdcs, pmu_paths, _) = place_pdcs_greedy(G, max_latency=MAX_LAT, flag_splitting=True)
    draw_graph(G, pdcs, pmu_paths, max_latency=MAX_LAT, output_path=f"{out_dir}/00_greedy-dario.png", pos=pos, view_mode=3)

    (pdcs, pmu_paths) = place_pdcs_resiliently(G, max_latency=MAX_LAT, essentialPMUs=N, v=v, R=R, parchi_constraint=PAR_CHI, cc_successors_constraint=(not CONSIDER_NEIGH), pdc_prio=PDC_PRIO, debug=DEBUG)
    draw_graph(G, pdcs, pmu_paths, max_latency=MAX_LAT, output_path=f"{out_dir}/000b_resilient.png", pos=pos, view_mode=3)
        
    # === From resiliency2 ===
    (pdcs, pmu_paths) = place_pdcs_node_disjoint(G, max_latency=MAX_LAT, essential_pmus=N, R=R)
    draw_graph(G, paths=pmu_paths, pdcs=pdcs, max_latency=MAX_LAT,
        output_path=f"{out_dir}/1_place_pdcs_node_disjoint.png", pos=pos, view_mode=3)
    
    (pdcs, pmu_paths) = place_pdcs_edge_disjoint(G, max_latency=MAX_LAT, essential_pmus=N, R=R)
    draw_graph(G, paths=pmu_paths, pdcs=pdcs, max_latency=MAX_LAT,
        output_path=f"{out_dir}/2_place_pdcs_edge_disjoint.png", pos=pos, view_mode=3)
    
    (pdcs, pmu_paths) = place_pdcs_greedy_edge_removal(G, max_latency=MAX_LAT, essential_pmus=N, R=R)
    draw_graph(G, paths=pmu_paths, pdcs=pdcs, max_latency=MAX_LAT,
        output_path=f"{out_dir}/3_place_pdcs_greedy_edge_removal.png", pos=pos, view_mode=3)
    
    (pdcs, pmu_paths) = place_pdcs_greedy_edge_penalty(G, max_latency=MAX_LAT, essential_pmus=N, R=R)
    draw_graph(G, paths=pmu_paths, pdcs=pdcs, max_latency=MAX_LAT,
        output_path=f"{out_dir}/4_place_pdcs_greedy_edge_penalty.png", pos=pos, view_mode=3)
    
    (pdcs, pmu_paths) = place_pdcs_suurballe(G, max_latency=MAX_LAT, essential_pmus=N, R=R)
    draw_graph(G, paths=pmu_paths, pdcs=pdcs, max_latency=MAX_LAT,
        output_path=f"{out_dir}/5_place_pdcs_suurballe.png", pos=pos, view_mode=3)
    
    (pdcs, pmu_paths) = place_pdcs_min_cost_flow_overlap(G, max_latency=MAX_LAT, essential_pmus=N, R=R)
    draw_graph(G, paths=pmu_paths, pdcs=pdcs, max_latency=MAX_LAT,
        output_path=f"{out_dir}/6_place_pdcs_min_cost_flow_overlap.png", pos=pos, view_mode=3)
    
    (pdcs, pmu_paths) = place_pdcs_bhandari(G, max_latency=MAX_LAT, essential_pmus=N, R=R)
    draw_graph(G, paths=pmu_paths, pdcs=pdcs, max_latency=MAX_LAT,
        output_path=f"{out_dir}/7_place_pdcs_bhandari.png", pos=pos, view_mode=3)
    
    (pdcs, pmu_paths) = place_pdcs_k_shortest_candidates(G, max_latency=MAX_LAT, essential_pmus=N, R=R)
    draw_graph(G, paths=pmu_paths, pdcs=pdcs, max_latency=MAX_LAT,
        output_path=f"{out_dir}/8_place_pdcs_k_shortest_candidates.png", pos=pos, view_mode=3)
    
    
    
    
    
    
    

def build_simple_R(N: int, M: int):
    top    = np.hstack([np.zeros((N, N)), np.eye(N, M)])
    bottom = np.hstack([np.eye(M, N), np.zeros((M, M))])
    return np.vstack([top, bottom])

def set_simple_red_role(G: nx.Graph, R):
    for n, d in G.nodes(data=True):
        if not re.match(r"[r]?PMU(\d+)", n):
            if d.get(NODE_ROLE, ROLE_CANDIDATE) != ROLE_PMU:
                continue
        
        i = vec_idx_from_pmu_name(n)
        
        r_of = d.get(NODE_REDUNDANT_OF, [])
        for j in range(i+1, R.shape[0]):
            if R[i, j] == 1:
                r_of.append(LABEL_PMU(j+1))
        
        d[NODE_REDUNDANT_OF] = r_of
        
def calc(path):
    return calc_path_cost(G, path.split("-"))

if __name__ == "__main__":
    main()