"""
This file is organized into three layers:

  1. Shared helpers (redundant-PMU grouping, super-source augmentation,
     weight functions, result aggregation).
  2. Disjoint-path ALGORITHMS, one function per technique, renamed so the
     name reflects what the algorithm actually guarantees.
  3. One `place_pdcs_*` PIPELINE per algorithm: parse PMUs -> build
     redundant groups -> run exactly one algorithm -> return
     (pdcs, pmu_paths).
"""

from operator import itemgetter
from typing import Callable, Optional

import networkx as nx

from resiliency import calc_path_cost, parse_pmus, vec_idx_from_pmu_name
from graph_model import (
    EDGE_LATENCY,
    LABEL_CC,
    LABEL_PMU,
    NODE_LATENCY,
    NODE_ROLE,
    ROLE_CANDIDATE,
)

LABEL_SUPERSOURCE = "temp0"


# ============================================================
# Shared helpers
# ============================================================

def make_path_weight_function(G: nx.Graph, super_source: str) -> Callable[[str, str, dict], float]:
    """Edge-weight function: edge latency + latency of the node being
    entered, except for edges leaving `super_source` (a synthetic node,
    not a real one, so it has no latency of its own).
    """
    def weight_fn(n1: str, _n2: str, edge_data: dict) -> float:
        if n1 == super_source:
            return edge_data.get(EDGE_LATENCY, 0)
        return G.nodes[n1].get(NODE_LATENCY, 0) + edge_data.get(EDGE_LATENCY, 0)

    return weight_fn


def build_redundant_pmu_groups(essential_pmus: list[str], R) -> dict[str, list[str]]:
    """For each essential PMU, find the PMUs correlated with it (redundant
    measurement sources) via the correlation matrix R.

    Returns {epmu: [epmu, redundant_pmu_1, redundant_pmu_2, ...]}.
    """
    groups: dict[str, list[str]] = {}
    for epmu in essential_pmus:
        group = [epmu]
        idx = vec_idx_from_pmu_name(epmu)
        row = R[idx]
        for j, val in enumerate(row):
            if val > 0:
                rpmu = LABEL_PMU(j + 1)
                group.append(rpmu)
                if rpmu in essential_pmus:
                    print(f"warning: essential PMUs should be uncorrelated ({epmu}, {rpmu})")
        groups[epmu] = group
    return groups


def with_super_source(G: nx.Graph, group: list[str], super_source: str = LABEL_SUPERSOURCE, weight: float = 0) -> nx.Graph:
    """Return a copy of G with a synthetic super-source connected (zero-cost
    by default) to every PMU in `group`, so a multi-source disjoint-path
    search can treat the whole group as a single origin."""
    Gcopy = G.copy()
    Gcopy.add_node(super_source)
    for pmu in group:
        Gcopy.add_edge(super_source, pmu, **{EDGE_LATENCY: weight})
    return Gcopy


def filter_paths_by_max_latency(G: nx.Graph, paths: list[list[str]], max_latency: float) -> list[list[str]]:
    return [p for p in paths if calc_path_cost(G, p) < max_latency]


def collect_pdcs_and_pmu_paths(G: nx.Graph, paths: list[list[str]]):
    """Turn a list of paths (each already starting at a real PMU node, not
    a super-source) into (pdcs, pmu_paths).

    (Renamed from `collect_pmu_paths_and_pdcs`; tuple order inverted to
    (pdcs, pmu_paths) to match the return order the pipelines use.)
    """
    pmu_paths = {}
    pdcs = set()
    for path in paths:
        pmu_paths[path[0]] = {"path": path, "delay": calc_path_cost(G, path)}
        for node in path:
            if G.nodes[node].get(NODE_ROLE, ROLE_CANDIDATE) == ROLE_CANDIDATE:
                pdcs.add(node)
    return pdcs, pmu_paths


# ============================================================
# 1) Node-disjoint paths (networkx builtin, max-flow/Menger based)
# ============================================================

def node_disjoint_paths_from_group(G: nx.Graph, group: list[str], target: str, cutoff: int = 10) -> list[list[str]]:
    """Node-disjoint paths from any PMU in `group` to `target`, via a
    synthetic super-source, using nx.node_disjoint_paths.
    Does not minimize latency."""
    Gcopy = with_super_source(G, group)
    paths: list[list[str]] = []
    try:
        for path in nx.node_disjoint_paths(Gcopy, s=LABEL_SUPERSOURCE, t=target, cutoff=cutoff):
            path.pop(0)
            paths.append(path)
    except nx.NetworkXNoPath:
        pass
    return paths


# ============================================================
# 2) Edge-disjoint paths (networkx builtin, max-flow based)
# ============================================================

def edge_disjoint_paths_from_group(G: nx.Graph, group: list[str], target: str, cutoff: int = 10) -> list[list[str]]:
    """Edge-disjoint paths from any PMU in `group` to `target`, via a
    synthetic super-source, using nx.edge_disjoint_paths.
    Does not minimize latency."""
    Gcopy = with_super_source(G, group)
    paths: list[list[str]] = []
    try:
        for path in nx.edge_disjoint_paths(Gcopy, s=LABEL_SUPERSOURCE, t=target, cutoff=cutoff):
            path.pop(0)
            paths.append(path)
    except nx.NetworkXNoPath:
        pass
    return paths


# ============================================================
# 3) Greedy edge-disjoint paths by HARD edge removal
# ============================================================

def greedy_edge_disjoint_paths_by_removal(
    G: nx.Graph,
    source: str,
    target: str,
    weight: Optional[Callable[[str, str, dict], float]] = None,
    k: int = 2,
    max_latency: float = None,
) -> list[list[str]]:
    """Repeatedly takes the current shortest path and deletes its edges before 
    searching again.
    No optimality guarantee for latency minimization.
    `max_latency` is only used to check if is possible to find paths."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if source == target:
        return [[source]]
    if not nx.has_path(G, source, target):
        raise nx.NetworkXNoPath(f"No path from {source} to {target}")

    working_graph = G.copy()
    paths: list[list[str]] = []

    for i in range(k):
        try:
            path = nx.shortest_path(working_graph, source=source, target=target, weight=weight)
        except nx.NetworkXNoPath:
            break
        if len(path) < 2:
            break
        if i == 0 and calc_path_cost(G, path) > max_latency:
            break   # Non esistono percorsi con la latenza richiesta

        paths.append(path)
        for u, v in zip(path[:-1], path[1:]):
            if working_graph.has_edge(u, v):
                working_graph.remove_edge(u, v)
            elif not working_graph.is_directed() and working_graph.has_edge(v, u):
                working_graph.remove_edge(v, u)

    return paths


# ============================================================
# 4) Greedy edge-disjoint paths by SOFT weight penalty
# ============================================================

def greedy_edge_disjoint_paths_by_penalty(
    G: nx.Graph,
    source: str,
    target: str,
    weight: Optional[Callable[[str, str, dict], float]] = None,
    k: int = 2,
    scale_factor: float = 10,
    max_latency: float = None,
) -> list[list[str]]:
    """Repeatedly takes the current shortest path and multiplies edge and node latency 
    before searching again, discouraging reuse on the next iteration.
    No optimality guarantee for latency minimization.
    `max_latency` is only used to check if is possible to find paths."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if source == target:
        return [[source]]
    if not nx.has_path(G, source, target):
        raise nx.NetworkXNoPath(f"No path from {source} to {target}")

    working_graph = G.copy()
    paths: list[list[str]] = []

    for i in range(k):
        try:
            path = nx.shortest_path(working_graph, source=source, target=target, weight=weight)
        except nx.NetworkXNoPath:
            break
        if len(path) < 2:
            break
        if i == 0 and calc_path_cost(G, path) > max_latency:
            break   # Non esistono percorsi con la latenza richiesta

        paths.append(path)

        for u, v in zip(path[:-1], path[1:]):
            working_graph.nodes[u][NODE_LATENCY] = scale_factor * working_graph.nodes[u].get(NODE_LATENCY, 0)
            if working_graph.has_edge(u, v):
                working_graph.edges[(u, v)][EDGE_LATENCY] = scale_factor * working_graph.edges[(u, v)].get(EDGE_LATENCY, 0)
            elif not working_graph.is_directed() and working_graph.has_edge(v, u):
                working_graph.edges[(v, u)][EDGE_LATENCY] = scale_factor * working_graph.edges[(v, u)].get(EDGE_LATENCY, 0)

    return paths


# ============================================================
# 5) Suurballe's algorithm (rigorous)
# ============================================================

def _dijkstra_shortest_tree(matrix: list[list[float]], initial: int, destination: int):
    size = len(matrix)
    dist = {initial: 0}
    parent: dict[int, int] = {}
    remaining = set(range(size))

    while remaining:
        current = None
        for node in remaining:
            if node in dist and (current is None or dist[node] < dist[current]):
                current = node
        if current is None:
            break
        remaining.remove(current)

        for neighbor in range(size):
            if matrix[current][neighbor] >= 0:
                new_dist = dist[current] + matrix[current][neighbor]
                if neighbor not in dist or new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    parent[neighbor] = current

    if destination not in dist:
        return dist, {}

    tree_edges: dict[int, int] = {}
    node = destination
    while node != initial:
        if node not in parent:
            break
        tree_edges[node] = parent[node]
        node = parent[node]

    return dist, tree_edges


def _suurballe_residual_matrix(matrix: list[list[float]], initial: int, destination: int):
    dist, path1 = _dijkstra_shortest_tree(matrix, initial, destination)
    size = len(matrix)
    original = [row[:] for row in matrix]

    # Reduce costs using node potentials from the first Dijkstra run.
    for i in range(size):
        for j in range(size):
            if matrix[i][j] >= 0:
                matrix[i][j] += dist[i] - dist[j]

    # Reverse the (now zero-cost) edges along path1 and forbid re-entering them forward.
    for j, i in path1.items():
        matrix[j][i] = matrix[i][j]
        matrix[i][j] = -1

    _, path2 = _dijkstra_shortest_tree(matrix, initial, destination)

    # Cancel edges traversed in opposite directions by the two paths ("interlacing").
    for j in list(path2.keys()):
        i = path2[j]
        if i in path1 and path1[i] == j:
            del path1[i]
            del path2[j]

    result = [[-1.0 for _ in range(size)] for _ in range(size)]
    for j, i in path1.items():
        result[i][j] = original[i][j]
    for j, i in path2.items():
        result[i][j] = original[i][j]
    return result


def suurballe_edge_disjoint_paths(
    G: nx.Graph,
    source: str,
    target: str,
    weight: Optional[Callable[[str, str, dict], float]] = None,
    k: int = 2,
) -> list[list[str]]:
    """Implementation of Suurballe's algorithm. It finds a k-uple of totally edge-disjoint
    paths minimizing the sum of their costs."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if source == target:
        return [[source]]
    if not nx.has_path(G, source, target):
        raise nx.NetworkXNoPath(f"No path from {source} to {target}")

    def default_weight(u, v, edge_data):
        return float(edge_data.get(EDGE_LATENCY, 0))

    weight_fn = weight or default_weight

    nodes = list(G.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    size = len(nodes)

    adjacency = [[-1.0 for _ in range(size)] for _ in range(size)]
    for u, v, edge_data in G.edges(data=True):
        cost = float(weight_fn(u, v, edge_data))
        adjacency[index[u]][index[v]] = cost
        if not G.is_directed():
            adjacency[index[v]][index[u]] = cost

    matrix = _suurballe_residual_matrix(adjacency, index[source], index[target])

    paths: list[list[str]] = []
    for _ in range(k):
        current = index[source]
        path = [nodes[current]]
        visited = {current}

        while current != index[target]:
            candidates = [n for n in range(size) if matrix[current][n] >= 0 and n not in visited]
            if not candidates:
                break
            nxt = candidates[0]
            path.append(nodes[nxt])
            visited.add(nxt)
            current = nxt

        if len(path) < 2 or path[-1] != target:
            break

        paths.append(path)
        for u_node, v_node in zip(path[:-1], path[1:]):
            matrix[index[u_node]][index[v_node]] = -1
            matrix[index[v_node]][index[u_node]] = -1

    return paths


# ============================================================
# 6) Tiered min-cost-flow least-overlap pair
# ============================================================
#
# Priority order enforced, regardless of the weights passed in:
#     shared NODE  >>  shared EDGE  >>  real latency.
#
# `weight_fn(n1, n2, edge_data)` follows the same convention as
# `nx.shortest_path(G, weight=weight_fn)`. If you don't pass one, the
# default is `edge_data.get("weight", 1)`.
#
# Latency is now part of the objective, not ignored: every edge copy in
# the flow network (the "free" one AND the "shared/penalized" one) carries
# the real weight_fn(u, v, edge_data) cost, because you pay that latency
# either way -- sharing an edge doesn't make it free to traverse, it just
# additionally costs the overlap penalty on top.
#
# edge_penalty and node_penalty are no longer parameters: they're derived
# from the graph so the priority order above always holds however large or
# small weight_fn's values are:
#   - edge_penalty > 2 * (sum of all edge latencies in the graph), which
#     is an upper bound on how much latency could possibly be saved,
#     across both paths combined, by choosing to share an edge instead of
#     avoiding it. So sharing is never "worth it" just to save latency.
#   - node_penalty > (num_edges + 1) * edge_penalty, which safely dominates
#     any combination of shared edges (at most num_edges of them) plus any
#     latency difference (already dominated by a single edge_penalty).

def _decompose_unit_flow(flow_dict, source, sink, n_paths: int):
    """Decompose an integer flow of value n_paths into n_paths simple
    paths from source to sink (assumes unit-capacity edges)."""
    remaining = {u: {v: f for v, f in d.items() if f > 0} for u, d in flow_dict.items()}
    paths = []
    for _ in range(n_paths):
        path = [source]
        current = source
        while current != sink:
            candidates = remaining.get(current, {})
            nxt = next(v for v, f in candidates.items() if f > 0)
            remaining[current][nxt] -= 1
            if remaining[current][nxt] == 0:
                del remaining[current][nxt]
            path.append(nxt)
            current = nxt
        paths.append(path)
    return paths


def _simplify_path(raw_path):
    """Strip auxiliary nodes (in/out split, dummy passthrough) down to the
    real graph nodes."""
    simplified = []
    for node in raw_path:
        if isinstance(node, tuple):
            if len(node) == 2 and node[1] in ("in", "out"):
                base = node[0]
            else:
                continue  # dummy node, not a real position
        else:
            base = node  # super-source or target
        if not simplified or simplified[-1] != base:
            simplified.append(base)
    return simplified


def min_cost_flow_least_overlap_pair(
    G: nx.Graph,
    node_a: str,
    node_b: str,
    target: str,
    weight_fn: Optional[Callable[[str, str, dict], float]] = None,
    latency_scale: int = 10**3,
):
    """
    NOTE: `nx.min_cost_flow` calls `network_simplex` internally, which is
    only guaranteed to terminate with INTEGER weights/capacities/demands
    (networkx's own docs warn that floating-point weights can make it
    cycle forever due to rounding). Real-valued latencies broke that
    guarantee -- this is what caused the infinite loop. Fixed by scaling
    every weight to an integer via `latency_scale` (default 1e6) before
    building the flow network. Raise `latency_scale` if your latencies
    need more decimal precision than that.
    """
    if node_a not in G or node_b not in G or target not in G:
        raise ValueError("node_a, node_b, target must all be nodes of the graph")
    if len({node_a, node_b, target}) < 3:
        raise ValueError("node_a, node_b, target must be three distinct nodes")

    if weight_fn is None:
        def weight_fn(n1, n2, edge_data):
            return edge_data.get(EDGE_LATENCY, 0)

    def integer_latency(edge_data):
        return round(edge_data.get(EDGE_LATENCY, 0) * latency_scale)

    # Auto-scaled penalties (integer arithmetic throughout, so the flow
    # solver is guaranteed to terminate):
    #   edge_penalty > 2 * (sum of all edge latencies), so sharing an edge
    #   is never "worth it" just to save latency.
    #   node_penalty > (num_edges + 1) * edge_penalty, so sharing a node is
    #   never "worth it" over sharing edges + any latency difference.
    total_latency = sum(
        integer_latency(edge_data)
        for u, v, edge_data in G.edges(data=True)
    )
    edge_penalty = 2 * total_latency + 1
    node_penalty = (G.number_of_edges() + 1) * edge_penalty + 1

    H = nx.DiGraph()
    super_source = LABEL_SUPERSOURCE

    def vin(v):
        return (v, "in")

    def vout(v):
        return (v, "out")

    for v in G.nodes():
        if v == target:
            continue
        H.add_edge(vin(v), vout(v), capacity=1, weight=0)
        dummy = (v, "dummy_node")
        H.add_edge(vin(v), dummy, capacity=1, weight=node_penalty)
        H.add_edge(dummy, vout(v), capacity=1, weight=0)

    for u, v, edge_data in G.edges(data=True):
        for x, y in ((u, v), (v, u)):
            if x == target:
                continue
            x_out = vout(x)
            y_in = target if y == target else vin(y)
            latency = integer_latency(edge_data)

            H.add_edge(x_out, y_in, capacity=1, weight=latency)
            dummy_e = (x, y, "dummy_edge")
            H.add_edge(x_out, dummy_e, capacity=1, weight=edge_penalty + latency)
            H.add_edge(dummy_e, y_in, capacity=1, weight=0)

    H.add_edge(super_source, vin(node_a), capacity=1, weight=0)
    H.add_edge(super_source, vin(node_b), capacity=1, weight=0)

    if target not in H:
        H.add_node(target)

    for n in H.nodes():
        H.nodes[n]["demand"] = 0
    H.nodes[super_source]["demand"] = -2
    H.nodes[target]["demand"] = 2

    try:
        flow_dict = nx.min_cost_flow(H)
    except nx.NetworkXUnfeasible:
        raise RuntimeError(
            f"No two paths {node_a}->{target} and {node_b}->{target} exist "
            "(graph not connected enough)."
        )

    raw_paths = _decompose_unit_flow(flow_dict, super_source, target, n_paths=2)
    paths = [_simplify_path(p) for p in raw_paths]

    path_a = paths[0] if paths[0][1] == node_a else paths[1]
    path_b = paths[0] if paths[0][1] == node_b else paths[1]
    path_a = path_a[1:]
    path_b = path_b[1:]

    return [path_a, path_b]


# ============================================================
# 7) Bhandari's maximally edge-disjoint paths
# ============================================================
#
# Generalized to return AT MOST `k` pairwise edge-disjoint paths. 
# This is the natural extension of the k=2 case: a "successive shortest 
# augmenting paths" loop, same family as min-cost flow with unit capacities. 
# At each iteration:
#   1. Build a working graph where edges already part of the accumulated
#      solution are heavily penalized in their original direction and made
#      available "in reverse" at negative cost (so a later path can
#      partially undo an earlier, suboptimal choice).
#   2. Find a shortest path in that working graph (Dijkstra on iteration 0,
#      since there are no negative edges yet; Bellman-Ford afterwards).
#   3. XOR its edges into the accumulated edge set: an edge used in the
#      opposite direction of one already in the set cancels out
#      ("interlacing"), otherwise it's added.
# After up to k iterations (fewer if no further augmenting path exists),
# the accumulated edge set is decomposed into up to k simple paths.

def bhandari_maximally_edge_disjoint_paths(
    G: nx.Graph,
    source: str,
    target: str,
    weight: str = EDGE_LATENCY,
    k: int = 2,
) -> list[list[str]]:
    if k < 1:
        raise ValueError("k must be at least 1")
    if source == target:
        return [[source]]

    G_dir = G.to_directed() if not G.is_directed() else G.copy()
    if not nx.has_path(G_dir, source, target):
        return []

    total_weight = sum(data.get(weight, 1) for _u, _v, data in G_dir.edges(data=True))
    big_penalty = total_weight * 10 + 1

    current_edges: set[tuple[str, str]] = set()

    for iteration in range(k):
        G_mod = G_dir.copy()
        for u, v in current_edges:
            G_mod[u][v][weight] += big_penalty
            original_w = G_dir[u][v][weight]
            G_mod[v][u][weight] = -original_w

        try:
            if iteration == 0:
                new_path = nx.shortest_path(G_mod, source=source, target=target, weight=weight)
            else:
                new_path = nx.bellman_ford_path(G_mod, source=source, target=target, weight=weight)
        except nx.NetworkXNoPath:
            break

        for u, v in zip(new_path[:-1], new_path[1:]):
            if (v, u) in current_edges:
                current_edges.remove((v, u))  # cancel interlacing
            else:
                current_edges.add((u, v))

    adj: dict[str, list[str]] = {}
    for u, v in current_edges:
        adj.setdefault(u, []).append(v)

    paths: list[list[str]] = []
    for _ in range(k):
        curr = source
        path = [curr]
        while curr != target:
            if curr in adj and adj[curr]:
                nxt = adj[curr].pop(0)
                path.append(nxt)
                curr = nxt
            else:
                break
        if len(path) >= 2 and path[-1] == target:
            paths.append(path)

    return paths


def bhandari_paths_from_group(G: nx.Graph, group: list[str], target: str) -> list[list[str]]:
    """Group-aware wrapper, mirroring node_disjoint/edge_disjoint/suurballe
    above: search from any PMU in `group` at once via a super-source, now
    that bhandari supports k > 2 this no longer needs to be pairwise."""
    Gcopy = with_super_source(G, group)
    paths = bhandari_maximally_edge_disjoint_paths(Gcopy, LABEL_SUPERSOURCE, target, weight=EDGE_LATENCY, k=len(group))
    for path in paths:
        path.remove(LABEL_SUPERSOURCE)
    return paths


# ============================================================
# 8) K-shortest-candidate best-pair search
# ============================================================

NODE_OVERLAP_PENALTY = 10
EDGE_OVERLAP_PENALTY = 1


def path_overlap_rank(path1: list[str], path2: list[str]) -> float:
    """Higher (less negative) is better = less overlap. Non-mutating."""
    trimmed1 = [n for n in path1 if n != LABEL_CC][1:]
    trimmed2 = [n for n in path2 if n != LABEL_CC][1:]

    common_nodes = [n for n in trimmed1 if n in trimmed2]
    edges1 = set(zip(trimmed1, trimmed1[1:]))
    edges2 = set(zip(trimmed2, trimmed2[1:])) | set(zip(trimmed2[1:], trimmed2))
    common_edges = [e for e in edges1 if e in edges2]

    return 0 - NODE_OVERLAP_PENALTY * len(common_nodes) - EDGE_OVERLAP_PENALTY * len(common_edges)


def k_shortest_candidate_paths(G: nx.Graph, pmu: str, target: str, k: int, max_latency: float) -> list[list[str]]:
    weight_fn = make_path_weight_function(G, pmu)
    candidates = []
    for i, path in enumerate(nx.shortest_simple_paths(G, pmu, target, weight=weight_fn)):
        if i >= k:
            break
        if max_latency is not None and calc_path_cost(G, path) > max_latency:
            break   # Da questo in poi saremmo fuori dal constraint di latenza
        
        candidates.append(path)
    return candidates


def best_candidate_pair(paths_a: list[list[str]], paths_b: list[list[str]]):
    """Pick the (path_a, path_b) combination with the least overlap."""
    scored = [
        ((i, j), path_overlap_rank(path_a, path_b))
        for i, path_a in enumerate(paths_a)
        for j, path_b in enumerate(paths_b)
    ]
    (best_i, best_j), _ = sorted(scored, key=itemgetter(1), reverse=True)[0]
    return paths_a[best_i], paths_b[best_j]


# ============================================================
# ============================================================
# place_pdcs_* pipelines
# ============================================================
# ============================================================
# Each follows the same shape: parse essential PMUs -> build redundant-PMU
# groups from R -> run exactly ONE algorithm per group -> return
# (pdcs, pmu_paths).

def place_pdcs_node_disjoint(G: nx.Graph, max_latency: float, essential_pmus, R):
    _, ePMUs, _ = parse_pmus(G, essential_pmus)
    groups = build_redundant_pmu_groups(ePMUs, R)

    all_paths = []
    for _, group in groups.items():
        all_paths.extend(node_disjoint_paths_from_group(G, group, LABEL_CC))

    all_paths = filter_paths_by_max_latency(G, all_paths, max_latency)
    return collect_pdcs_and_pmu_paths(G, all_paths)


def place_pdcs_edge_disjoint(G: nx.Graph, max_latency: float, essential_pmus, R):
    _, ePMUs, _ = parse_pmus(G, essential_pmus)
    groups = build_redundant_pmu_groups(ePMUs, R)

    all_paths = []
    for _, group in groups.items():
        all_paths.extend(edge_disjoint_paths_from_group(G, group, LABEL_CC))

    all_paths = filter_paths_by_max_latency(G, all_paths, max_latency)
    return collect_pdcs_and_pmu_paths(G, all_paths)


def place_pdcs_greedy_edge_removal(G: nx.Graph, max_latency: float, essential_pmus, R):
    _, ePMUs, _ = parse_pmus(G, essential_pmus)
    groups = build_redundant_pmu_groups(ePMUs, R)

    all_paths = []
    for _, group in groups.items():
        Gcopy = with_super_source(G, group)
        weight_fn = make_path_weight_function(Gcopy, LABEL_SUPERSOURCE)
        for path in greedy_edge_disjoint_paths_by_removal(Gcopy, LABEL_SUPERSOURCE, LABEL_CC, weight=weight_fn, k=2, max_latency=max_latency):
            path.pop(0)
            all_paths.append(path)

    all_paths = filter_paths_by_max_latency(G, all_paths, max_latency)
    return collect_pdcs_and_pmu_paths(G, all_paths)


def place_pdcs_greedy_edge_penalty(G: nx.Graph, max_latency: float, essential_pmus, R):
    _, ePMUs, _ = parse_pmus(G, essential_pmus)
    groups = build_redundant_pmu_groups(ePMUs, R)

    all_paths = []
    for _, group in groups.items():
        Gcopy = with_super_source(G, group)
        weight_fn = make_path_weight_function(Gcopy, LABEL_SUPERSOURCE)
        for path in greedy_edge_disjoint_paths_by_penalty(Gcopy, LABEL_SUPERSOURCE, LABEL_CC, weight=weight_fn, k=2, max_latency=max_latency):
            path.pop(0)
            all_paths.append(path)

    all_paths = filter_paths_by_max_latency(G, all_paths, max_latency)
    return collect_pdcs_and_pmu_paths(G, all_paths)


def place_pdcs_suurballe(G: nx.Graph, max_latency: float, essential_pmus, R):
    _, ePMUs, _ = parse_pmus(G, essential_pmus)
    groups = build_redundant_pmu_groups(ePMUs, R)

    all_paths = []
    for _, group in groups.items():
        Gcopy = with_super_source(G, group)
        weight_fn = make_path_weight_function(Gcopy, LABEL_SUPERSOURCE)
        for path in suurballe_edge_disjoint_paths(Gcopy, LABEL_SUPERSOURCE, LABEL_CC, weight=weight_fn, k=len(group)):
            path.pop(0)
            all_paths.append(path)

    all_paths = filter_paths_by_max_latency(G, all_paths, max_latency)
    return collect_pdcs_and_pmu_paths(G, all_paths)


def place_pdcs_min_cost_flow_overlap(G: nx.Graph, max_latency: float, essential_pmus, R):
    """Inherently pairwise (one epmu + one redundant pmu -> two paths to
    the CC). If an essential PMU has more than one redundant PMU, only the
    first is paired (same limitation as before)."""
    _, ePMUs, _ = parse_pmus(G, essential_pmus)
    groups = build_redundant_pmu_groups(ePMUs, R)
    weight_fn = make_path_weight_function(G, LABEL_SUPERSOURCE)

    all_paths = []
    for _, group in groups.items():
        if len(group) < 2:
            continue  # no redundant PMU to pair with
        if len(group) > 2:
            print(f"Ignoring some PMUs... {group[2:]}")
        all_paths.extend(min_cost_flow_least_overlap_pair(G, group[0], group[1], LABEL_CC))

    all_paths = filter_paths_by_max_latency(G, all_paths, max_latency)
    return collect_pdcs_and_pmu_paths(G, all_paths)


def place_pdcs_bhandari(G: nx.Graph, max_latency: float, essential_pmus, R, k: int = 2):
    _, ePMUs, _ = parse_pmus(G, essential_pmus)
    groups = build_redundant_pmu_groups(ePMUs, R)

    all_paths = []
    for _, group in groups.items():
        all_paths.extend(bhandari_paths_from_group(G, group, LABEL_CC))

    all_paths = filter_paths_by_max_latency(G, all_paths, max_latency)
    return collect_pdcs_and_pmu_paths(G, all_paths)


def place_pdcs_k_shortest_candidates(G: nx.Graph, max_latency: float, essential_pmus, R, K: int = 5):
    """Same known limitation as before: with multiple redundant PMUs per
    essential PMU, only the first pairing is fixed (would need a
    recursive/joint assignment to fix properly)."""
    PMUs, ePMUs, _ = parse_pmus(G, essential_pmus)
    candidate_paths = {pmu: k_shortest_candidate_paths(G, pmu, LABEL_CC, K, max_latency) for pmu in PMUs}

    groups = build_redundant_pmu_groups(ePMUs, R)
    fixed_paths: dict[str, list[str]] = {}

    for epmu, group in groups.items():
        if epmu in fixed_paths:
            continue
        for rpmu in group[1:]:
            if rpmu in fixed_paths:
                continue
            best_a, best_b = best_candidate_pair(candidate_paths[epmu], candidate_paths[rpmu])
            fixed_paths[epmu] = best_a
            fixed_paths[rpmu] = best_b

    return collect_pdcs_and_pmu_paths(G, list(fixed_paths.values()))