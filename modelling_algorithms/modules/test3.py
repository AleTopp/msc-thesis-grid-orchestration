from graph_model import NODE_REDUNDANT_OF, ROLE_PMU, create_graph
from visualizer import draw_graph, get_layout
from placement_pdc import place_pdcs_greedy
from resiliency import PDC_PRIO_UNCHANGED, place_pdcs_resiliently, vec_idx_from_pmu_name
from resiliency2 import *
from collections import Counter
import networkx as nx
import numpy as np
import re, math, random, datetime

G: nx.Graph = None

def test_case(_params: dict[str, str], candidates_to_pmu_ratio: float = 1):
    x =  _params["num_candidates"] / (2 * candidates_to_pmu_ratio)
    
    N, M = math.ceil(x), math.floor(x)
    v = np.ones((N+M, 1), dtype=int)
    R = build_simple_R(N, M)
    
    G = create_graph(
        seed=_params["seed"], 
        num_pmus=N+M, 
        num_candidates=_params["num_candidates"], 
        cc_max_links=_params["cc_max_links"], 
        edge_latency_min=_params["edge_lat_min"], 
        edge_latency_max=_params["edge_lat_max"],
    )
    set_simple_red_role(G, R)
    
    params = {
        "seed": _params["seed"],
        "num_candidates": _params["num_candidates"],
        "cc_max_links": _params["cc_max_links"], 
        "edge_latency_min": _params["edge_lat_min"], 
        "edge_latency_max": _params["edge_lat_max"],
        "N": N,
        "M": M,
        "max_latency": _params["max_latency"],
        "flag_splitting": _params["flag_splitting"],
        "essentialPMUs": N,
        "v": v,
        "R": R,
        "parchi_constraint": _params["parchi_constraint"],
        "cc_successors_constraint": _params["cc_successors_constraint"],
        "pdc_prio": _params["pdc_prio"],
        "K": _params["K"],
        "out_dir": f"{_params["out_dir"].replace(" ", "/").replace(":", "-")}/nc{_params["num_candidates"]}_s{_params["seed"]}",
    }
    #print(f"params: {params}")
    exec_placing(G, params)
    
def exec_placing(G: nx.Graph, params: dict[str, str]):
    dir = params["out_dir"]
    params_draw = {
        "max_latency": params["max_latency"],
        "pos": get_layout(G),
        "view_mode": 3,
    }

    crashing_node = random.choice([n for n, role in G.nodes(data=NODE_ROLE, default=ROLE_CANDIDATE) if role == ROLE_CANDIDATE])
    crashing_edge = random.choice([e for e in G.edges()])
    
    # === Helper functions ===
    
    def change_out(func_name: str):
        params_draw["output_path"] = f"{dir}/{func_name}.png"
    
    def run_and_save(func: Callable, out_name: str, **params: dict[str, str]):
        result = None
        delta_time = -1
        if func is not None:
            before = datetime.datetime.now()
            try:
                result = func(G, **params)
            except KeyboardInterrupt:
                print(f"⚠⚠ Skipped function: {dir}/{out_name}.png ⚠⚠")
                result = None
            after = datetime.datetime.now()
            delta_time = after - before
        
        pdcs, pmu_paths = None, None
        if result is not None:
            pdcs = result[0]
            pmu_paths = result[1]
            
        change_out(out_name)
        draw_graph(G, pdcs=pdcs, paths=pmu_paths, **params_draw)
        if result is not None:
            evaluate_paths(G, pdcs, pmu_paths, delta_t=delta_time, name=out_name)
            crash_and_eval(pmu_paths, name=out_name)
        
    def evaluate_paths(G: nx.Graph, pdcs: set, pmu_paths: dict, delta_t: datetime.timedelta, name: str, output_path: str = f"{dir}/metrics.txt"):
        essentialPMUs = params["essentialPMUs"]
        R = params["R"]
        lines = []
        
        # === Valutazioni statiche ===
        lines.append(f"== Static evaluations for {name} ==")
        # 1) Numero di PDCs
        lines.append(f"Number of PDCs: {len(pdcs)}")
        # 2a) Percentuale di dati che arrivano
        flows = 100*sum(1 for v in pmu_paths.values() if v["path"])/R.shape[0]
        lines.append(f"% of data which arrives at CC: {round(flows, 2)}%")
        # 2b) Percentuale di dati resilienti
        ePMUs = [LABEL_PMU(i+1) for i in range(int(essentialPMUs))]
        groups_dict = build_redundant_pmu_groups(ePMUs, R)
        res = sum(
            1
            for group in groups_dict.values()
            if sum(1 for pmu in group if pmu in pmu_paths) > 1
        )
        lines.append(f"% of data which arrives in 2+ copies: {round(100*res/len(ePMUs), 2)}%")
        # 3) Tempo di convergenza
        lines.append(f"Execution time: {str(delta_t)}")
        # 4) Archi vs Numero di flussi di dati
        edge_counts = Counter(
            tuple(sorted((u, v)))
            for path in (val["path"] for val in pmu_paths.values())
            for u, v in zip(path[:-1], path[1:])
        )
        reverse_edges = {
            count: [edge for edge, value in edge_counts.items() if value == count]
            for count in set(edge_counts.values())
        }
        lines.append("Edges vs Number of flows:")
        for i, edges in reverse_edges.items():
            perc = round(100*len(edges) / G.number_of_edges(), 2)
            lines.append(f"{i} Flow{'s' if i > 1 else ' '} | {len(edges)} ({perc}%)")
        
        lines.append("")
        with open(output_path, mode='+a') as f:
            f.writelines([f"{l}\n" for l in lines])

    def crash_and_eval(pmu_paths: dict, name: str, output_path: str = f"{dir}/metrics.txt"):
        essentialPMUs = params["essentialPMUs"]
        R = params["R"]

        ePMUs = [LABEL_PMU(i+1) for i in range(int(essentialPMUs))]
        groups_dict = build_redundant_pmu_groups(ePMUs, R)

        # === Valutazioni dinamiche ===
        lines = [f"== Dynamic evaluations for {name} =="]
        flows_before = 100*sum(1 for v in pmu_paths.values() if v["path"])/R.shape[0]

        # 1) Nodo crashato
        lines.append(f"Crashing node: {crashing_node}")
        node_crash_results = {pmu: 1 for pmu in pmu_paths.keys()} # 0: dead, 1: alive
        for pmu, path in [(pmu, val["path"]) for pmu, val in pmu_paths.items()]:
            if crashing_node in path:
                node_crash_results[pmu] = 0   # Data from 'pmu' cannot reach CC anymore

        # 1a) Quanti flussi arrivano ancora al CC
        flows_now = 100*sum(v for v in node_crash_results.values())/R.shape[0]
        lines.append(f"% of data which still arrives at CC: {round(flows_now, 2)} (before was {round(flows_before, 2)}%)")

        # 1b) Quanti flussi indipendenti arrivano ancora al CC (in 1 o 2+ copie)
        res = sum(
            1
            for group in groups_dict.values()
            if sum(node_crash_results.get(pmu, 0) for pmu in group) > 0
        )
        lines.append(f"% of independent data which arrives in 1 copies: {round(100*res/len(ePMUs), 2)}%")
        res = sum(
            1
            for group in groups_dict.values()
            if sum(node_crash_results.get(pmu, 0) for pmu in group) > 1
        )
        lines.append(f"% of independent data which arrives in 2+ copies: {round(100*res/len(ePMUs), 2)}%")


        # 2) Arco crashato
        lines.append(f"\nCrashing edge: {crashing_edge}")
        edge_crash_results = {pmu: 1 for pmu in pmu_paths.keys()} # 0: dead, 1: alive
        for pmu, path in [(pmu, val["path"]) for pmu, val in pmu_paths.items()]:
            path_edges = zip(path[:-1], path[1:])
            if crashing_edge in path_edges:
                edge_crash_results[pmu] = 0   # Data from 'pmu' cannot reach CC anymore

        # 2a) Quanti flussi arrivano ancora al CC
        flows_now = 100*sum(v for v in edge_crash_results.values())/R.shape[0]
        lines.append(f"% of data which still arrives at CC: {round(flows_now, 2)}% (before was {round(flows_before, 2)}%)")

        # 2b) Quanti flussi indipendenti arrivano ancora al CC (in 1 o 2+ copie)
        res = sum(
            1
            for group in groups_dict.values()
            if sum(edge_crash_results.get(pmu, 0) for pmu in group) > 0
        )
        lines.append(f"% of independent data which arrives in 1 copies: {round(100*res/len(ePMUs), 2)}%")
        res = sum(
            1
            for group in groups_dict.values()
            if sum(edge_crash_results.get(pmu, 0) for pmu in group) > 1
        )
        lines.append(f"% of independent data which arrives in 2+ copies: {round(100*res/len(ePMUs), 2)}%")

        lines.append("----------------------------\n\n")
        with open(output_path, mode='+a') as f:
            f.writelines([f"{l}\n" for l in lines])
    
    # ==== Execution ====
    run_and_save(None, "0_graph")
    
    params_greedy = {
        "max_latency": params["max_latency"],
        "flag_splitting": params["flag_splitting"],
    }
    run_and_save(place_pdcs_greedy, "00_greedy", **params_greedy)
    
    params_resilient = {
        "max_latency": params["max_latency"],
        "essentialPMUs": params["essentialPMUs"],
        "v": params["v"],
        "R": params["R"],
        "parchi_constraint": params["parchi_constraint"],
        "cc_successors_constraint": params["cc_successors_constraint"],
        "pdc_prio": params["pdc_prio"],
    }
    run_and_save(place_pdcs_resiliently, "1_resilient", **params_resilient)

    params_others = {
        "max_latency": params["max_latency"],
        "essential_pmus": params["essentialPMUs"],
        "R": params["R"],
    }
    run_and_save(place_pdcs_greedy_edge_penalty, "2_greedy-edge-penalty", **params_others)
    run_and_save(place_pdcs_min_cost_flow_overlap, "3_tiered-min-cost", **params_others)
    run_and_save(place_pdcs_k_shortest_candidates, f"4_k({params["K"]})-shortest-candidates", **params_others, K=params["K"])


def main():
    STARTING_SEED = 42
    # Graph
    EDGE_LAT_MIN = 1
    EDGE_LAT_MAX = 3
    
    # Algo
    MAX_LAT = 500
    SPLITTING = False
    CONSIDER_NEIGH = True
    PAR_CHI = False
    PDC_PRIO = PDC_PRIO_UNCHANGED       # (Non cambia davvero nulla, bisognerebbe fare a ~~parità di latenza allora prio)
    
    # All
    params = {
        #"seed": SEED,
        #"num_candidates": CANDIDATES,
        #"cc_max_links": MAX_CC_LINKS,
        "edge_lat_min": EDGE_LAT_MIN,
        "edge_lat_max": EDGE_LAT_MAX,
        "max_latency": MAX_LAT,
        "flag_splitting": SPLITTING,
        "parchi_constraint": PAR_CHI,
        "cc_successors_constraint": (not CONSIDER_NEIGH),
        "pdc_prio": PDC_PRIO,
        #"K": int(CANDIDATES/2) + 1,
        "out_dir": f"output/{str(datetime.datetime.now())}",
    }
    
    random.seed(STARTING_SEED)

    # Random or specific seeds
    seeds = [random.randrange(0, 1000) for _ in range(10)]
    # seeds = [...]
    
    # Graph sizes (num_candidates) to check
    sizes = [8, 10]
    # sizes = [...]
    
    for size in sizes:
        params["num_candidates"] = size
        params["cc_max_links"] = math.floor(size/2)
        params["K"] = math.floor(size/2) + 1
        for seed in seeds:
            params["seed"] = seed
            test_case(params)

    
    

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

if __name__ == "__main__":
    main()