import json

from graph_model import NODE_REDUNDANT_OF, ROLE_PMU, create_graph
from visualizer import draw_graph, get_layout
from placement_pdc import _TimeoutException, place_pdcs_greedy, place_pdcs_random
from resiliency import PDC_PRIO_UNCHANGED, place_pdcs_resiliently, vec_idx_from_pmu_name
from resiliency2 import *
from collections import Counter
import networkx as nx
import numpy as np
import re, math, random, datetime

G: nx.Graph = None

def test_case(_params: dict[str, str], metrics: dict[str, list], skipped: list[tuple], candidates_to_pmu_ratio: float = 1):
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
    
    all_params = {
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
        "out_dir": f"{_params["out_dir"]}/nc{_params["num_candidates"]}_s{_params["seed"]}",
    }
    exec_placing(G, all_params, metrics, skipped)
    
def exec_placing(G: nx.Graph, all_params: dict[str, str], metrics: dict[str, list], skipped: list[tuple]):
    size = int(all_params["num_candidates"])
    seed = int(all_params["seed"])
    dir = all_params["out_dir"]
    params_draw = {
        "max_latency": all_params["max_latency"],
        "pos": get_layout(G),
        "view_mode": 3,
    }

    crashing_node = random.choice([n for n, role in G.nodes(data=NODE_ROLE, default=ROLE_CANDIDATE) if role == ROLE_CANDIDATE])
    crashing_edge = random.choice([e for e in G.edges()])
    
    crashing_seq_nodes = [n for n, role in G.nodes(data=NODE_ROLE, default=ROLE_CANDIDATE) if role == ROLE_CANDIDATE]
    crashing_seq_edges = [e for e in G.edges()]
    random.shuffle(crashing_seq_nodes)
    random.shuffle(crashing_seq_edges)
    crashing_seq_nodes = crashing_seq_nodes[:size]
    crashing_seq_edges = crashing_seq_edges[:size]
    
    crashing_seq = [*crashing_seq_nodes, *crashing_seq_edges]
    random.shuffle(crashing_seq)
    
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
            except (KeyboardInterrupt, _TimeoutException):
                print(f"⚠⚠ Skipped function: {dir}/{out_name}.png ⚠⚠")
                skipped.append((size, seed, out_name, "Timeout"))
                result = None
            after = datetime.datetime.now()
            delta_time = after - before
        
        pdcs, pmu_paths = None, None
        if result is not None:
            pdcs = result[0]
            pmu_paths = result[1]
        
        # Disegnare tutti i grafi comporta notevole quantità di tempo e spazio in più
        if func is None:
            change_out(out_name)
            draw_graph(G, pdcs=pdcs, paths=pmu_paths, **params_draw)
            
        if result is not None:
            with open(f"{dir}/{out_name}.json", mode='w') as f:
                s = json.dumps(pmu_paths, indent=2)
                f.write(s)
                
            evaluate_paths(G, pdcs, pmu_paths, delta_t=delta_time, name=out_name)
            crash_and_eval(pmu_paths, name=out_name)
        
    def evaluate_paths(G: nx.Graph, pdcs: set, pmu_paths: dict, delta_t: datetime.timedelta, name: str, output_path: str = f"{dir}/metrics.txt"):
        essentialPMUs = all_params["essentialPMUs"]
        R = all_params["R"]
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
        # 4) Jain index per la fairness degli edge
        all_edges = [tuple(sorted((u, v))) for u, v in G.edges()]
        edge_counts = Counter(
            tuple(sorted((u, v)))
            for path in (val["path"] for val in pmu_paths.values())
            for u, v in zip(path[:-1], path[1:])
        )
        values = [edge_counts.get(edge, 0) for edge in all_edges]
        sum_x = sum(values)
        sum_x2 = sum(x * x for x in values)
        edge_jain_index = (sum_x ** 2) / (len(values) * sum_x2) if sum_x2 > 0 else 1.0
        lines.append(f"Jain index (edge fairness): {round(edge_jain_index, 4)}")
        # 4b) Jain index per la fairness dei nodi
        all_nodes = [n for n in G.nodes()]
        node_counts = Counter(
            n
            for path in (val["path"] for val in pmu_paths.values())
            for n in path
        )
        values = [node_counts.get(node, 0) for node in all_nodes]
        sum_x = sum(values)
        sum_x2 = sum(x * x for x in values)
        node_jain_index = (sum_x ** 2) / (len(values) * sum_x2) if sum_x2 > 0 else 1.0
        lines.append(f"Jain index (node fairness): {round(node_jain_index, 4)}")
        # 5) Archi vs Numero di flussi di dati
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
        
        if flows == 100: # Se non arrivano tutti, scartiamo il seed
            rPMUs = [LABEL_PMU(i+1) for i in range(int(R.shape[0])) if LABEL_PMU(i+1) not in ePMUs]
            
            append_to_metrics(
                size = size,
                algorithm = name,
                
                seed = seed,
                pdcs_num = len(pdcs),
                execution_time = delta_t / datetime.timedelta(milliseconds=1),
                latency_epmus = [val["delay"] for pmu, val in pmu_paths.items() if pmu in ePMUs],
                latency_rpmus = [val["delay"] for pmu, val in pmu_paths.items() if pmu in rPMUs],
                edge_jain_index = edge_jain_index,
                node_jain_index = node_jain_index,
            )
        else:
            append_to_metrics(
                size = size,
                algorithm = name,
                
                drop = True
            )
            skipped.append((size, seed, name, "Incomplete"))

    def crash_and_eval(pmu_paths: dict, name: str, output_path: str = f"{dir}/metrics.txt"):
        essentialPMUs = all_params["essentialPMUs"]
        R = all_params["R"]

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
        node_res = res
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
            if crashing_edge in path_edges or tuple(reversed(crashing_edge)) in path_edges:
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
        edge_res = res
        lines.append(f"% of independent data which arrives in 1 copies: {round(100*res/len(ePMUs), 2)}%")
        res = sum(
            1
            for group in groups_dict.values()
            if sum(edge_crash_results.get(pmu, 0) for pmu in group) > 1
        )
        lines.append(f"% of independent data which arrives in 2+ copies: {round(100*res/len(ePMUs), 2)}%")
        
        # 3) Quanti failure prima di perdere osservabilità (un gruppo di resilienza)? (MTTF)
        lines.append(f"\nCrashing sequence: {crashing_seq}")
        seq_crash_results = {pmu: 1 for pmu in pmu_paths.keys()} # 0: dead, 1: alive
        ttf = None
        for i, fail in enumerate(crashing_seq):
            if type(fail) == str:
                for pmu, path in [(pmu, val["path"]) for pmu, val in pmu_paths.items()]:
                    if fail in path:
                        seq_crash_results[pmu] = 0   # Data from 'pmu' cannot reach CC anymore
            else:
                for pmu, path in [(pmu, val["path"]) for pmu, val in pmu_paths.items()]:
                    path_edges = zip(path[:-1], path[1:])
                    if fail in path_edges or tuple(reversed(fail)) in path_edges:
                        seq_crash_results[pmu] = 0   # Data from 'pmu' cannot reach CC anymore
            
            res = sum(
                1
                for group in groups_dict.values()
                if sum(seq_crash_results.get(pmu, 0) for pmu in group) > 0
            )
            if res < len(groups_dict):
                ttf = i + 1   # Se fallisce all'indice 0, il numero di failure necessari è 1
                break
        
        lines.append(f"Time (no. of node/edge faults) To Failure (of Observability): {ttf if not None else f">{len(crashing_seq)}"}")
        
        # 3b) Quanti node fails prima di perdere osservabilità?
        seq_crash_results = {pmu: 1 for pmu in pmu_paths.keys()} # 0: dead, 1: alive
        nftf = None
        for i, fail in enumerate(crashing_seq_nodes):
            for pmu, path in [(pmu, val["path"]) for pmu, val in pmu_paths.items()]:
                if fail in path:
                    seq_crash_results[pmu] = 0   # Data from 'pmu' cannot reach CC anymore
            
            res = sum(
                1
                for group in groups_dict.values()
                if sum(seq_crash_results.get(pmu, 0) for pmu in group) > 0
            )
            if res < len(groups_dict):
                nftf = i + 1   # Se fallisce all'indice 0, il numero di fault necessari è 1
                break
        lines.append(f"Time (no. of node faults) To Failure (of Observability): {nftf if not None else f">{len(crashing_seq_nodes)}"}")

        # 3c) Quanti edge fails prima di perdere osservabilità?
        seq_crash_results = {pmu: 1 for pmu in pmu_paths.keys()} # 0: dead, 1: alive
        eftf = None
        for i, fail in enumerate(crashing_seq_nodes):
            for pmu, path in [(pmu, val["path"]) for pmu, val in pmu_paths.items()]:
                path_edges = zip(path[:-1], path[1:])
                if fail in path_edges or tuple(reversed(fail)) in path_edges:
                    seq_crash_results[pmu] = 0   # Data from 'pmu' cannot reach CC anymore
            
            res = sum(
                1
                for group in groups_dict.values()
                if sum(seq_crash_results.get(pmu, 0) for pmu in group) > 0
            )
            if res < len(groups_dict):
                eftf = i + 1   # Se fallisce all'indice 0, il numero di fault necessari è 1
                break
        lines.append(f"Time (no. of edge faults) To Failure (of Observability): {eftf if not None else f">{len(crashing_seq_edges)}"}")


        lines.append("----------------------------\n\n")
        with open(output_path, mode='+a') as f:
            f.writelines([f"{l}\n" for l in lines])
            
        if sum(1 for v in pmu_paths.values() if v["path"]) == R.shape[0]:
            append_to_metrics(
                size = size,
                algorithm = name,
                
                ttf = ttf,
                flows_after_node = node_res,
                flows_after_edge = edge_res,
            )
            
    def append_to_metrics(**info):
        size = info["size"]
        algo = info["algorithm"]
        if not size or not algo:
            raise ValueError("metrics key not provided")
        
        if info.get("drop", False) == True:
            return

        if (size, algo) not in metrics:
            metrics[(size, algo)] = {
                "size": size,
                "algorithm": algo,
                
                "seeds": [],
                "pdcs_nums": [],
                "execution_times": [],
                "latency_distribution": {
                    "ePMUs": [],
                    "rPMUs": []
                },
                "edge_jain_indexes": [],
                "node_jain_indexes": [],
                "times_to_failure": [],
                "node_faults_to_failure": [],
                "edge_faults_to_failure": [],
                "independent_data_flows_after_1_node_fail": [],
                "independent_data_flows_after_1_edge_fail": [],
            }
            
        unitary_fields = {
            "seed": "seeds",
            "pdcs_num": "pdcs_nums",
            "execution_time": "execution_times",
            "edge_jain_index": "edge_jain_indexes",
            "node_jain_index": "node_jain_indexes",
            "ttf": "times_to_failure",
            "nftf": "node_faults_to_failure",
            "eftf": "edge_faults_to_failure",
            "flows_after_node": "independent_data_flows_after_1_node_fail",
            "flows_after_edge": "independent_data_flows_after_1_edge_fail",
        }
        
        for src, dst in unitary_fields.items():
            metrics[(size, algo)][dst].extend([info[src]] if src in info else [])
            
        metrics[(size, algo)]["latency_distribution"]["ePMUs"].extend(info.get("latency_epmus", []))
        metrics[(size, algo)]["latency_distribution"]["rPMUs"].extend(info.get("latency_rpmus", []))

    
    # ==== Execution ====
    run_and_save(None, "0_graph")
    
    params_greedy = {
        "max_latency": all_params["max_latency"],
        "flag_splitting": all_params["flag_splitting"],
    }
    run_and_save(place_pdcs_greedy, "00_greedy", **params_greedy)
    run_and_save(place_pdcs_random, "000_random", **params_greedy)
    
    params_resilient = {
        "max_latency": all_params["max_latency"],
        "essentialPMUs": all_params["essentialPMUs"],
        "v": all_params["v"],
        "R": all_params["R"],
        "parchi_constraint": all_params["parchi_constraint"],
        "cc_successors_constraint": all_params["cc_successors_constraint"],
        "pdc_prio": all_params["pdc_prio"],
    }
    run_and_save(place_pdcs_resiliently, "1_resilient", **params_resilient)

    params_others = {
        "max_latency": all_params["max_latency"],
        "essential_pmus": all_params["essentialPMUs"],
        "R": all_params["R"],
    }
    run_and_save(place_pdcs_greedy_edge_penalty, "2_greedy-edge-penalty", **params_others)
    run_and_save(place_pdcs_min_cost_flow_overlap, "3_tiered-min-cost", **params_others)
    run_and_save(place_pdcs_k_shortest_candidates, f"4_k({all_params["K"]})-shortest-candidates", **params_others, K=all_params["K"])


def main():
    STARTING_TIME = datetime.datetime.now()
    STARTING_SEED = 42
    OUT_DIR = f"runtime_results/{str(STARTING_TIME)}".replace(" ", "/").replace(":", "-")
    
    # Graph
    EDGE_LAT_MIN = 1
    EDGE_LAT_MAX = 10
    
    # Algo
    MAX_LAT = 500
    SPLITTING = False
    CONSIDER_NEIGH = True
    PAR_CHI = False
    PDC_PRIO = PDC_PRIO_UNCHANGED       # (Non cambia davvero nulla, bisognerebbe fare a ~~parità di latenza allora prio)
                                        # Se risultati mostrano troppi PDC magari aggiustiamo
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
        "out_dir": OUT_DIR,
    }
    
    random.seed(STARTING_SEED)

    # Random or specific seeds
    # seeds = [random.randrange(0, 1000) for _ in range(50)]
    seeds = [654, 114, 25, 759, 281, 250, 228, 142, 754, 104, 692, 758, 913, 558, 89, 604, 432, 32, 30, 95, 223, 238, 517, 616, 27, 574, 203, 733, 665, 718, 558, 429, 225, 459, 603, 284, 828, 890, 6, 777, 825, 163, 714, 432, 348, 284, 159, 220, 980, 781]
    seeds = seeds[:20]
    
    # Graph sizes (num_candidates) to check
    sizes = [10, 20, 30, 40, 50]
    # sizes = [50]
    
    metrics_dict = {}
    skipped = []
    
    for size in sizes:
        params["num_candidates"] = size
        params["cc_max_links"] = math.floor(size/2)
        params["K"] = math.floor(size/2) + 1
        params["flag_splitting"] = params["flag_splitting"] if size < 25 else True
        for seed in seeds:
            params["seed"] = seed
            test_case(params, metrics_dict, skipped)
            
    ENDING_TIME = datetime.datetime.now()
            
    with open(f"{OUT_DIR}/metrics.json", mode='w') as f:
        s = json.dumps(list(metrics_dict.values()), indent=2)
        f.write(s)

    with open(f"{OUT_DIR}/skipped.json", mode='w') as f:
        s = json.dumps(skipped, indent=2)
        f.write(s)

    with open(f"{OUT_DIR}/execution.txt", mode='w') as f:
        f.write(f"STARTING_TIME: {STARTING_TIME}\n")
        f.write(f"ENDING_TIME: {ENDING_TIME}\n")
        f.write(f"ELAPSED_TIME: {ENDING_TIME - STARTING_TIME}\n")
        
    
    

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