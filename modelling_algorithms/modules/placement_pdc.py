import networkx as nx
import random
import numpy as np
from itertools import islice
from itertools import combinations
from itertools import product
import signal
from functools import wraps

#-----------------TIMEOUT DECORATOR------------------#

class _TimeoutException(Exception):
    pass

def timeout_return_empty(seconds: int = 120):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def _handler(signum, frame):
                raise _TimeoutException()

            old_handler = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(int(seconds))
            try:
                return func(*args, **kwargs)
            except _TimeoutException:
                max_latency = kwargs.get("max_latency", None)
                if max_latency is None and len(args) >= 2:
                    max_latency = args[1]
                raise _TimeoutException()
                # return ([], {}, max_latency)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator

#-----------------PLACEMENT ALGORITHMS------------------#

@timeout_return_empty()
def place_pdcs_greedy(G, max_latency, flag_splitting=False):

    # -------------------- helpers --------------------

    def edge_key(u, v):
        return tuple(sorted((u, v)))

    def path_delay(H, path):
        edge_latency = sum(
            float(H[a][b].get("latency", 0.0))
            for a, b in zip(path[:-1], path[1:])
        )
        node_processing = sum(
            float(H.nodes[n].get("processing", 0.0))
            for n in path
            if H.nodes[n].get("role") not in {"PMU", "CC"}
        )
        return edge_latency + node_processing

    def build_active_subgraph(G):
        H = nx.Graph()
        for n, data in G.nodes(data=True):
            if data.get("status", "online") == "online":
                H.add_node(n, **data)
        for u, v, data in G.edges(data=True):
            if u in H and v in H and data.get("status", "up") == "up":
                H.add_edge(u, v, **data)
        return H

    # --------- UPDATED FUNCTION WITH STRUCTURAL PRUNING ---------

    def k_best_paths_within_latency(H, src, dst, K, max_lat):

        if src not in H or dst not in H:
            return []
        if not nx.has_path(H, src, dst):
            return []

        PROCESSING_COST = 21.5

        # Maximum number of candidate nodes allowed in a path
        max_candidates = int(max_lat // PROCESSING_COST) + 2 # to exclude CC and PMU

        paths_list = []

        try:
            gen = nx.shortest_simple_paths(H, src, dst, weight="latency")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

        for path in gen:

            num_candidates = sum(
                1 for n in path
                if H.nodes[n].get("role") == "candidate"
            )

            if num_candidates > max_candidates:
                continue

            d = float(path_delay(H, path))

            if d <= max_lat:
                paths_list.append((path, d))
                if len(paths_list) >= K:
                    break

        return paths_list

    # -------------------- build filtered graph --------------------

    H = build_active_subgraph(G)
    cc = "CC"
    if cc not in H:
        raise ValueError("CC node not present or not online (after filtering).")

    pmus = [n for n, d in H.nodes(data=True) if d.get("role") == "PMU"]
    pmu_rate = {pmu: float(H.nodes[pmu].get("data_rate", 0.0)) for pmu in pmus}

    if not flag_splitting:
        attached_demand = {}
        for pmu in pmus:
            for nbr in H.neighbors(pmu):
                if H.nodes[nbr].get("role") == "candidate":
                    attached_demand[nbr] = attached_demand.get(nbr, 0.0) + pmu_rate[pmu]

        for x, dem in attached_demand.items():
            if H.has_edge(x, cc):
                cap = float(H[x][cc].get("bandwidth", float("inf")))
                if cap < dem:
                    H.remove_edge(x, cc)

    # -------------------- MODE: splitting allowed --------------------

    if flag_splitting:

        used_bw = {}
        pmu_paths = {}
        pdcs = set()

        for pmu in pmus:
            demand = pmu_rate[pmu]
            K = 3

            opts = k_best_paths_within_latency(H, pmu, cc, K=K, max_lat=max_latency)

            chosen = None
            chosen_d = None

            for path, d in opts:
                ok = True
                for a, b in zip(path[:-1], path[1:]):
                    cap = float(H[a][b].get("bandwidth", float("inf")))
                    k = edge_key(a, b)
                    if used_bw.get(k, 0.0) + demand > cap:
                        ok = False
                        break
                if not ok:
                    continue

                chosen = path
                chosen_d = float(d)
                break

            if chosen is None:
                continue

            for a, b in zip(chosen[:-1], chosen[1:]):
                k = edge_key(a, b)
                used_bw[k] = used_bw.get(k, 0.0) + demand

            pmu_paths[pmu] = {"path": chosen, "delay": chosen_d}

            for node in chosen:
                if H.nodes[node].get("role") == "candidate":
                    pdcs.add(node)

        return pdcs, pmu_paths, max_latency

    # -------------------- MODE: splitting NOT allowed --------------------

    K = 3 # number of candidate paths to consider per PMU (after structural pruning)
    BACKTRACK_DEPTH = 3

    candidates = {}
    for pmu in pmus:
        candidates[pmu] = k_best_paths_within_latency(
            H, pmu, cc, K=K, max_lat=max_latency
        )

    order = sorted(pmus, key=lambda p: len(candidates.get(p, [])))

    used_bw = {}
    next_hop_to_cc = {}
    assignment = {}

    def can_place(pmu, path):
        demand = pmu_rate[pmu]

        for i in range(1, len(path) - 1):
            node = path[i]
            if H.nodes[node].get("role") != "candidate":
                continue
            nxt = path[i + 1]
            if node in next_hop_to_cc and next_hop_to_cc[node] != nxt:
                return False

        for a, b in zip(path[:-1], path[1:]):
            cap = float(H[a][b].get("bandwidth", float("inf")))
            k = edge_key(a, b)
            if used_bw.get(k, 0.0) + demand > cap:
                return False

        return True

    def apply_place(pmu, path):
        demand = pmu_rate[pmu]

        for a, b in zip(path[:-1], path[1:]):
            k = edge_key(a, b)
            used_bw[k] = used_bw.get(k, 0.0) + demand

        for i in range(1, len(path) - 1):
            node = path[i]
            if H.nodes[node].get("role") == "candidate":
                next_hop_to_cc.setdefault(node, path[i + 1])

    def rebuild_constraints_from_assignment():
        used_bw.clear()
        next_hop_to_cc.clear()

        for pmu2, (p2, _) in assignment.items():
            dem2 = pmu_rate[pmu2]
            for a, b in zip(p2[:-1], p2[1:]):
                k = edge_key(a, b)
                used_bw[k] = used_bw.get(k, 0.0) + dem2
            for i in range(1, len(p2) - 1):
                node = p2[i]
                if H.nodes[node].get("role") == "candidate":
                    next_hop_to_cc.setdefault(node, p2[i + 1])

    def greedy_limited_backtracking():

        decision_stack = []
        i = 0
        backtrack_streak = 0

        while i < len(order):

            pmu = order[i]
            opts = candidates.get(pmu, [])

            start_choice = 0
            if decision_stack and decision_stack[-1][0] == i and decision_stack[-1][1] == pmu:
                start_choice = decision_stack[-1][2] + 1
                decision_stack.pop()

            placed = False
            for choice_idx in range(start_choice, len(opts)):
                path, d = opts[choice_idx]
                if can_place(pmu, path):
                    assignment[pmu] = (path, d)
                    apply_place(pmu, path)
                    decision_stack.append((i, pmu, choice_idx))
                    placed = True
                    break

            if placed:
                backtrack_streak = 0
                i += 1
                continue

            if decision_stack:
                last_i, last_pmu, last_choice = decision_stack.pop()
                assignment.pop(last_pmu, None)
                rebuild_constraints_from_assignment()
                i = last_i
                decision_stack.append((i, last_pmu, last_choice))
                backtrack_streak += 1

                if backtrack_streak > BACKTRACK_DEPTH:
                    backtrack_streak = 0
                    i += 1
                continue

            backtrack_streak = 0
            i += 1

    greedy_limited_backtracking()

    pmu_paths = {}
    pdcs = set()

    for pmu, (path, d) in assignment.items():
        pmu_paths[pmu] = {"path": path, "delay": float(d)}
        for node in path:
            if H.nodes[node].get("role") == "candidate":
                pdcs.add(node)

    return pdcs, pmu_paths, max_latency



@timeout_return_empty(1*60*60)
def place_pdcs_greedy_no_backtracking(G, max_latency, flag_splitting=False):

    def edge_key(u, v):
        return tuple(sorted((u, v)))

    def path_delay(H, path):
        return sum(
            float(H[a][b].get("latency", 0.0))
            for a, b in zip(path[:-1], path[1:])
        )

    def build_active_subgraph(G):
        H = nx.Graph()
        for n, data in G.nodes(data=True):
            if data.get("status", "online") == "online":
                H.add_node(n, **data)

        for u, v, data in G.edges(data=True):
            if u in H and v in H and data.get("status", "up") == "up":
                H.add_edge(u, v, **data)
        return H

    H = build_active_subgraph(G)

    cc = "CC"
    if cc not in H:
        raise ValueError("Nodo 'CC' non presente o non online (dopo filtering).")

    pmus = [n for n, d in H.nodes(data=True) if d.get("role") == "PMU"]

    used_bw = {}        # (u,v) -> used bandwidth
    next_hop_to_cc = {} # PDC(candidate) -> next hop toward CC (no-splitting)

    pmu_paths = {}      # pmu -> {"path": [...], "delay": float}   
    pdcs = set()
    accepted_delays = []

    for pmu in pmus:
        demand = float(H.nodes[pmu].get("data_rate", 0.0))

        try:
            paths_iter = nx.shortest_simple_paths(H, pmu, cc, weight="latency")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue  

        chosen_path = None
        chosen_delay = None

        for path in paths_iter:
            delay = path_delay(H, path)
            if delay > max_latency:
                continue

            if not flag_splitting:
                ok = True
                for i in range(1, len(path) - 1):
                    node = path[i]
                    if H.nodes[node].get("role") != "candidate":
                        continue
                    nxt = path[i + 1]
                    if node in next_hop_to_cc and next_hop_to_cc[node] != nxt:
                        ok = False
                        break
                if not ok:
                    continue

            ok = True
            for a, b in zip(path[:-1], path[1:]):
                cap = float(H[a][b].get("bandwidth", float("inf")))
                k = edge_key(a, b)
                if used_bw.get(k, 0.0) + demand > cap:
                    ok = False
                    break
            if not ok:
                continue

            chosen_path = path
            chosen_delay = float(delay)
            break

        if chosen_path is None:
            continue  

        for a, b in zip(chosen_path[:-1], chosen_path[1:]):
            k = edge_key(a, b)
            used_bw[k] = used_bw.get(k, 0.0) + demand

        if not flag_splitting:
            for i in range(1, len(chosen_path) - 1):
                node = chosen_path[i]
                if H.nodes[node].get("role") == "candidate":
                    next_hop_to_cc.setdefault(node, chosen_path[i + 1])

        pmu_paths[pmu] = {"path": chosen_path, "delay": chosen_delay}
        accepted_delays.append(chosen_delay)

        for node in chosen_path:
            if H.nodes[node].get("role") == "candidate":
                pdcs.add(node)

    max_delay_out = max(accepted_delays) if accepted_delays else 0.0
    return pdcs, pmu_paths, max_delay_out

import random

@timeout_return_empty()
def place_pdcs_random(
    G,
    max_latency,
    seed=None,
    flag_splitting=True,       
    max_tries_per_pmu=80,
    sample_paths_per_pmu=10,
    debug=False
):

    if seed is not None:
        random.seed(seed)
        
    def get_cc_node(G):
        for n, data in G.nodes(data=True):
            if data.get("role") == "CC":
                return n
        raise ValueError("No CC node found in graph")

    CC = get_cc_node(G)

    def edge_key(u, v):
        return (u, v) if (u, v) in G.edges else (v, u)

    def edge_is_up(u, v):
        k = edge_key(u, v)
        return G.edges[k].get("status", "up") == "up"

    def node_is_online(n):
        if "status" in G.nodes[n]:
            return G.nodes[n].get("status") == "online"
        return G.nodes[n].get("online", True)

    def can_step(u, v):
        if G.nodes[v].get("role") == "PMU":
            return False
        if v != CC and not node_is_online(v):
            return False
        if not edge_is_up(u, v):
            return False
        return True

    def path_delay(path):
        total = 0.0
        for u, v in zip(path, path[1:]):
            k = edge_key(u, v)
            total += float(G.edges[k].get("latency", 0.0))
            if G.nodes[u].get("role") == "candidate":
                total += float(G.nodes[u].get("processing", 0.0))
        return total

    def bw_ok_on_edge(u, v, data_rate, committed_bw, local_bw):
        k = edge_key(u, v)
        cap = float(G.edges[k].get("bandwidth", float("inf")))
        used = float(committed_bw.get(k, 0.0)) + float(local_bw.get(k, 0.0))
        return (used + data_rate) <= cap

    def commit_bandwidth(path, data_rate, committed_bw):
        for u, v in zip(path, path[1:]):
            k = edge_key(u, v)
            committed_bw[k] = float(committed_bw.get(k, 0.0)) + float(data_rate)

    # --- state ---
    pdcs = set()
    pmu_paths = {}
    bandwidth_usage = {}    # edge_key -> cumulative traffic

    # only used when no_splitting is enforced
    forced_suffix = {}      # candidate_node -> suffix path [candidate_node, ..., CC]

    pmu_nodes = [n for n in G.nodes if G.nodes[n].get("role") == "PMU"]

    def suffix_feasible(suffix, visited, data_rate, committed_bw, local_bw):
        for x in suffix[1:]:
            if x in visited:
                return False
        for u, v in zip(suffix, suffix[1:]):
            if not (G.has_edge(u, v) or G.has_edge(v, u)):
                return False
            if v != CC and not can_step(u, v):
                return False
            if not bw_ok_on_edge(u, v, data_rate, committed_bw, local_bw):
                return False
        return True

    def update_forced_suffixes_for_path(path):
        for i, n in enumerate(path[:-1]):  # exclude CC
            if G.nodes[n].get("role") == "candidate":
                suffix = path[i:]
                if n in forced_suffix:
                    if forced_suffix[n] != suffix:
                        return False
                else:
                    forced_suffix[n] = suffix
        return True

    def dfs_random(current, target, visited, data_rate, committed_bw, local_bw):
        if current == target:
            return [current]

        visited.add(current)

        # Enforce no-splitting ONLY when no_splitting == True (i.e., flag_splitting == False)
        if not flag_splitting and G.nodes[current].get("role") == "candidate" and current in forced_suffix:
            suffix = forced_suffix[current]
            if suffix and suffix[0] == current and suffix_feasible(suffix, visited, data_rate, committed_bw, local_bw):
                for u, v in zip(suffix, suffix[1:]):
                    k = edge_key(u, v)
                    local_bw[k] = float(local_bw.get(k, 0.0)) + float(data_rate)
                visited.remove(current)
                return suffix[:]
            visited.remove(current)
            return None

        neighbors = list(G.neighbors(current))
        random.shuffle(neighbors)

        for nxt in neighbors:
            if nxt in visited:
                continue

            if nxt != target:
                if not can_step(current, nxt):
                    continue
            else:
                if not edge_is_up(current, nxt):
                    continue

            if not bw_ok_on_edge(current, nxt, data_rate, committed_bw, local_bw):
                continue

            k = edge_key(current, nxt)
            local_bw[k] = float(local_bw.get(k, 0.0)) + float(data_rate)

            sub = dfs_random(nxt, target, visited, data_rate, committed_bw, local_bw)
            if sub:
                visited.remove(current)
                return [current] + sub

            # backtrack
            local_bw[k] = float(local_bw.get(k, 0.0)) - float(data_rate)
            if local_bw[k] <= 0:
                local_bw.pop(k, None)

        visited.remove(current)
        return None

    # ---- main loop ----
    for pmu in pmu_nodes:
        data_rate = float(G.nodes[pmu].get("data_rate", 0.0))
        found_candidates = []

        for _ in range(max_tries_per_pmu):
            local_bw = {}
            path = dfs_random(pmu, CC, set(), data_rate, bandwidth_usage, local_bw)
            if not path:
                continue

            delay = path_delay(path)
            if delay > max_latency:
                continue

            # If no-splitting enforced, ensure this path is consistent with existing suffixes
            if not flag_splitting:
                if not update_forced_suffixes_for_path(path):
                    continue

            found_candidates.append((path, delay))
            if len(found_candidates) >= sample_paths_per_pmu:
                break

        if not found_candidates:
            if debug:
                print(f"⚠️ Not valid paths for {pmu} → CC (constraints max_latency/bandwidth/status/no_splitting).")
            continue

        # choose randomly among found candidates
        chosen_path, chosen_delay = random.choice(found_candidates)

        # commit bandwidth globally
        commit_bandwidth(chosen_path, data_rate, bandwidth_usage)

        pmu_paths[pmu] = {"path": chosen_path, "delay": chosen_delay}
        for node in chosen_path[1:-1]:
            if G.nodes[node].get("role") == "candidate":
                pdcs.add(node)

    # ---- report ----
    if pmu_paths:
        if debug:
            print(f"\n📍 Best configuration covers {len(pmu_paths)}/{len(pmu_nodes)} PMUs (max_latency={max_latency}).")
        for pmu, data in pmu_paths.items():
            path = data["path"]
            delay = data["delay"]
            if debug:
                print(f"{pmu} → CC: {' → '.join(path)}, Delay = {delay:.2f} ms")
    else:
        if debug:
            print("❌ No valid configuration found (covers 0 PMUs).")

    return (pdcs, pmu_paths, max_latency)
       
@timeout_return_empty(1*60*60)       
def place_pdcs_bruteforce(G, max_latency, flag_splitting=True, max_paths_per_pmu=None, cutoff=5):

    def is_valid_chain(path, pdc_nodes, G):
        if not path:
            return False
        roles = [G.nodes[n].get("role") for n in path]
        if roles[0] != "PMU" or roles[-1] != "CC":
            return False

        # intermediate nodes must be PDCs and online
        for i in range(1, len(path) - 1):
            if path[i] not in pdc_nodes:
                return False
            if not G.nodes[path[i]].get("online", True):
                return False

        # edges must exist and be up
        for u, v in zip(path, path[1:]):
            if not (G.has_edge(u, v) or G.has_edge(v, u)):
                return False
            e = (u, v) if (u, v) in G.edges else (v, u)
            if G[e[0]][e[1]].get("status", "up") != "up":
                return False

        return True

    def compute_path_latency(path, G, pdc_nodes):
        latency = 0.0
        for u, v in zip(path, path[1:]):
            e = (u, v) if (u, v) in G.edges else (v, u)
            latency += float(G[e[0]][e[1]].get("latency", 0.0))
        for n in path:
            if n in pdc_nodes:
                latency += float(G.nodes[n].get("processing", 0.0))
        return latency

    def check_splitting(pdc_nodes, pmu_paths):
        pdc_to_pmus = {}
        for pmu, data in pmu_paths.items():
            path = data["path"]
            for n in path[1:-1]:
                if n in pdc_nodes:
                    pdc_to_pmus.setdefault(n, []).append(pmu)

        for pdc, pmus in pdc_to_pmus.items():
            if len(pmus) > 1:
                ref_suffix = None
                for pmu in pmus:
                    path = pmu_paths[pmu]["path"]
                    idx = path.index(pdc)
                    suffix = path[idx:]  # from pdc to CC
                    if ref_suffix is None:
                        ref_suffix = suffix
                    elif ref_suffix != suffix:
                        return True
        return False

    # --- collect nodes ---
    pmu_nodes = [n for n, d in G.nodes(data=True) if d.get("role") == "PMU"]
    candidate_nodes = [n for n, d in G.nodes(data=True) if d.get("role") == "candidate"]
    cc_nodes = [n for n, d in G.nodes(data=True) if d.get("role") == "CC"]
    if not cc_nodes:
        raise ValueError("No CC node found in G")
    cc_node = cc_nodes[0]

    # --- best tracking ---
    best_config = None
    best_paths = {}

    best_covered = -1
    best_total_latency = float("inf")
    best_k = float("inf")

    # --- brute force over PDC sets ---
    for k in range(1, len(candidate_nodes) + 1):
        for pdc_tuple in combinations(candidate_nodes, k):
            pdc_nodes = set(pdc_tuple)

            # For each PMU, compute feasible paths (filtered by max_latency)
            pmu_to_paths = {}
            covered_pmus = []

            for pmu in pmu_nodes:
                allowed_nodes = set(pdc_nodes) | {pmu, cc_node}

                # build subgraph with only allowed nodes and up edges
                subgraph = nx.Graph()
                for u in allowed_nodes:
                    subgraph.add_node(u, **G.nodes[u])

                for u, v in G.edges():
                    if u in allowed_nodes and v in allowed_nodes:
                        if G[u][v].get("status", "up") == "up":
                            subgraph.add_edge(u, v, **G[u][v])

                # enumerate paths
                try:
                    all_paths = list(nx.all_simple_paths(subgraph, source=pmu, target=cc_node, cutoff=cutoff))
                except nx.NetworkXNoPath:
                    all_paths = []

                if not all_paths:
                    continue

                # keep only valid chains + latency <= max_latency
                feasible = []
                for p in all_paths:
                    if not is_valid_chain(p, pdc_nodes, G):
                        continue
                    delay = compute_path_latency(p, G, pdc_nodes)
                    if delay <= max_latency:
                        feasible.append((p, delay))

                if not feasible:
                    continue

                # sort by delay, optionally limit
                feasible.sort(key=lambda x: x[1])
                if max_paths_per_pmu is not None:
                    feasible = feasible[:max_paths_per_pmu]

                pmu_to_paths[pmu] = [p for (p, _d) in feasible]
                covered_pmus.append(pmu)

            # if covers nothing, skip
            if not covered_pmus:
                continue

            # Cartesian product: one path choice per covered PMU
            pmu_list = covered_pmus
            paths_product_iter = product(*(pmu_to_paths[pmu] for pmu in pmu_list))

            for paths_choice in paths_product_iter:
                current_paths = {pmu: {"path": path} for pmu, path in zip(pmu_list, paths_choice)}

                # splitting constraint
                if not flag_splitting and check_splitting(pdc_nodes, current_paths):
                    continue

                # bandwidth accumulation
                bandwidth_usage = {}
                bandwidth_ok = True

                for pmu in pmu_list:
                    data_rate = float(G.nodes[pmu].get("data_rate", 0.0))
                    path = current_paths[pmu]["path"]

                    # check without committing
                    for u, v in zip(path, path[1:]):
                        edge = (u, v) if (u, v) in G.edges else (v, u)
                        cap = float(G.edges[edge].get("bandwidth", float("inf")))
                        used = float(bandwidth_usage.get(edge, 0.0))
                        if used + data_rate > cap:
                            bandwidth_ok = False
                            break
                    if not bandwidth_ok:
                        break

                    # commit usage
                    for u, v in zip(path, path[1:]):
                        edge = (u, v) if (u, v) in G.edges else (v, u)
                        bandwidth_usage[edge] = float(bandwidth_usage.get(edge, 0.0)) + data_rate

                if not bandwidth_ok:
                    continue

                # compute total latency (sum over covered PMUs), and store per-PMU delay
                total_latency = 0.0
                for pmu in pmu_list:
                    path = current_paths[pmu]["path"]
                    delay = compute_path_latency(path, G, pdc_nodes)

                    # This should already be <= max_latency due to filtering, but keep safe:
                    if delay > max_latency:
                        total_latency = None
                        break

                    current_paths[pmu]["delay"] = delay
                    total_latency += delay

                if total_latency is None:
                    continue

                covered_count = len(pmu_list)

                # ranking: maximize coverage, then minimize total latency, then minimize #PDC
                better = False
                if covered_count > best_covered:
                    better = True
                elif covered_count == best_covered:
                    if total_latency < best_total_latency:
                        better = True
                    elif total_latency == best_total_latency and k < best_k:
                        better = True

                if better:
                    best_config = list(pdc_nodes)
                    best_paths = current_paths.copy()
                    best_covered = covered_count
                    best_total_latency = total_latency
                    best_k = k

    # --- output ---
    if best_config:
        print(f"\n📍 Best configuration covers {best_covered}/{len(pmu_nodes)} PMUs (max_latency={max_latency}).")
        for pmu, data in best_paths.items():
            path = data["path"]
            delay = data["delay"]
            print(f"{pmu} → CC: {' → '.join(path)}, Delay = {delay:.2f} ms")
    else:
        print("❌ No valid configuration found (covers 0 PMUs).")

    return best_config if best_config else [], best_paths, max_latency


#*********************************************************#
#                    WORK IN PROGRESS                     #
#*********************************************************#


def q_learning_placement(G, max_latency, episodes=25000, alpha=0.1, gamma=0.9, epsilon=0.8, seed=None, verbose=False):
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    nodes_pdc = [n for n, data in G.nodes(data=True)
                 if data.get("role") == "candidate" and data.get("status") == "online"]
    pmu_nodes = [n for n, data in G.nodes(data=True) if data.get("role") == "PMU"]

    num_actions = len(nodes_pdc)
    Q = np.zeros((2 ** num_actions, num_actions)) # inizializzazione Q-table

    def compute_total_delay(path): # calcola il ritardo totale di un path
        delay = 0.0
        for u, v in zip(path, path[1:]):
            delay += G[u][v]["latency"]
        for node in path[1:-1]:
            delay += G.nodes[node].get("processing", 0)
        return delay
    
    # def valid_path(path, state):# controlla che ci siano solo PDC online e archi up
    #     for node in path[1:-1]:
    #         if node not in nodes_pdc:
    #             return False
    #         idx = nodes_pdc.index(node)
    #         if state[idx] != 1:  # se nodo è candidato ma non attivo ( da state[idx] )
    #             return False
    #     for u, v in zip(path, path[1:]): # se qualche arco nel path non è attivo
    #         if G[u][v].get("status") != "up":
    #             return False
    #     if any(G.nodes[n].get("status") != "online" for n in path):
    #         return False
    #     return True
    
    def valid_path(path, state, bandwidth_usage, data_rate):
        # Verifica nodi intermedi PDC attivi
        for node in path[1:-1]:
            if node not in nodes_pdc:
                return False
            idx = nodes_pdc.index(node)
            if state[idx] != 1:
                return False
        # Verifica stato archi e nodi + saturazione banda
        for u, v in zip(path, path[1:]):
            if G[u][v].get("status") != "up":
                return False
            edge = (u, v) if (u, v) in G.edges else (v, u)
            capacity = G.edges[edge].get("bandwidth", float("inf"))
            usage = bandwidth_usage.get(edge, 0)
            if usage + data_rate > capacity:
                return False
        if any(G.nodes[n].get("status") != "online" for n in path):
            return False
        return True


    
    def find_best_paths(state):
        pmu_to_best = {}
        bandwidth_usage = {}
        for pmu in pmu_nodes:
            data_rate = G.nodes[pmu].get("data_rate", 0)
            best_path = None
            best_delay = float("inf")
            try:
                paths = islice(nx.all_simple_paths(G, pmu, "CC", cutoff=15), 500)
                for path in paths:
                    if not valid_path(path, state, bandwidth_usage, data_rate):
                        continue
                    delay = compute_total_delay(path)
                    if delay < best_delay:
                        best_delay = delay
                        best_path = path
            except nx.NetworkXNoPath:
                continue
            if best_path:
                pmu_to_best[pmu] = {"path": best_path, "delay": best_delay}
                for u, v in zip(best_path, best_path[1:]):
                    edge = (u, v) if (u, v) in G.edges else (v, u)
                    bandwidth_usage[edge] = bandwidth_usage.get(edge, 0) + data_rate
        return pmu_to_best


    def state_to_index(state):
        return int("".join(str(b) for b in state), 2)

    best_state = None
    best_score = -float("inf")

    for ep in range(episodes):
        if ep % 1000 == 0: print(f"🔄 Episodio {ep}/{episodes}...")
        state = [0] * num_actions
        pmu_covered = set()

        for step in range(num_actions): # ad ogni step, si attiva un PDC
            s_idx = state_to_index(state) # converte lo stato in indice per Q-table
            available = [i for i in range(num_actions) if state[i] == 0] # seleziona i PDC ancora spenti
            if not available:
                break

            if random.random() < epsilon:
                action = random.choice(available)
            else:
                q_values = Q[s_idx]
                q_masked = [q if i in available else -np.inf for i, q in enumerate(q_values)]
                action = int(np.argmax(q_masked))

            state[action] = 1

            best_paths = find_best_paths(state)
            new_covered = set(best_paths.keys()) # nuove PMU coperte
            delta_covered = len(new_covered - pmu_covered)
            pmu_covered = new_covered

            total_delay = sum(data["delay"] for data in best_paths.values())

            pdc_count = sum(state)

            reward = (
                +20 * delta_covered
                - 1.0 * total_delay / 100
                - 3 * pdc_count
            )


            next_s_idx = state_to_index(state)
            Q[s_idx, action] += alpha * (reward + gamma * np.max(Q[next_s_idx]) - Q[s_idx, action])

            if verbose:
                print(f"[Ep{ep:4d}][St{step}] ΔPMU={delta_covered}, delay={total_delay:.1f}, PDCs={pdc_count}, R={reward:.2f}")

            # if len(pmu_covered) == len(pmu_nodes): # tutte le PMU sono coperte
            #     break

        # finito l'episodio, calcola il punteggio per quell'episodio ( il Q-value valuta l'action effettuato )
        final_paths = find_best_paths(state)
        total_delay = sum(data["delay"] for data in final_paths.values())
        pdc_count = sum(state)
        latency_avg = total_delay / len(final_paths) if final_paths else float('inf')
        # Conta i PDC attivi ma non usati nei path
        used_pdcs = set()
        for data in final_paths.values():
            path = data["path"]
            for node in path[1:-1]:
                used_pdcs.add(node)

        useless_pdc_count = sum(
            1 for i, b in enumerate(state)
            if b == 1 and nodes_pdc[i] not in used_pdcs
        )

        final_score = (
            1000 * len(final_paths) 
            - 5 * latency_avg 
            - 20 * pdc_count
            - 50 * useless_pdc_count  # penalità forte per sprechi
        )


        #final_score = 1000 * len(final_paths) - total_delay - 20 * pdc_count

        if final_score > best_score:
            best_score = final_score
            best_state = state.copy()

    # Esito finale
    # selected_pdc = {nodes_pdc[i] for i, b in enumerate(best_state) if b == 1}
    # print("\n Politica appresa:")
    # print(f" Stato binario: {''.join(str(b) for b in best_state)}")
    # print(f" Nodi PDC selezionati: {selected_pdc}")
    # 🔎 Ricava i PDC usati nei cammini finali
    used_pdcs = set()
    for data in final_paths.values():
        path = data["path"]
        for node in path[1:-1]:  # solo nodi intermedi tra PMU e CC
            used_pdcs.add(node)


    # 🧼 Pulisci best_state per mantenere solo i PDC usati
    clean_best_state = [0] * len(nodes_pdc)
    for i, n in enumerate(nodes_pdc):
        if best_state[i] == 1 and n in used_pdcs:
            clean_best_state[i] = 1
    best_state = clean_best_state
    
    selected_pdc = {nodes_pdc[i] for i, b in enumerate(best_state) if b == 1}


    print("\n📡 Cammini PMU → CC (validi con PDC selezionati):")
    final_paths = find_best_paths(best_state)
    for pmu in pmu_nodes:
        if pmu in final_paths:
            path = final_paths[pmu]["path"]
            delay = final_paths[pmu]["delay"]

            status = " OK" if delay <= max_latency else f" Ritardo {delay:.2f} ms > soglia"
            print(f"  {pmu} → CC: {path} | Ritardo = {delay:.2f} ms {status}")
        else:
            print(f"  {pmu} → CC:  Nessun path valido")

    return selected_pdc, final_paths, max_latency if selected_pdc else None





