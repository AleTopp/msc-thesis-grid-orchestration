from graph_model import EDGE_LATENCY, LABEL_CC, NODE_LATENCY, NODE_ROLE, ROLE_CANDIDATE, ROLE_PMU
from placement_pdc import place_pdcs_greedy
from operator import itemgetter
import networkx as nx
import numpy as np
import math, re, types

LATENCY_THRESHOLD = 200

PDC_PRIO_UNCHANGED = 0
PDC_PRIO_FALSE = 1
PDC_PRIO_TRUE = 2


def place_pdcs_resiliently(
  G: nx.Graph, 
  max_latency: float, 
  essentialPMUs: list[str] | int, 
  v, 
  R, 
  parchi_constraint: bool = True,
  cc_successors_constraint: bool = True,
  pdc_prio: int = PDC_PRIO_UNCHANGED,
  debug: bool = False
):
  PMUs, _, rPMUs = parse_pmus(G, essentialPMUs)
    
  greedyG = G.copy()
  greedyG.remove_nodes_from(rPMUs)

  (pdcs, pmu_paths, _) = place_pdcs_greedy(greedyG, max_latency, False)
  T = prefix_tree_from_pmu_paths(pmu_paths)

  for rpmu in rPMUs:
    shortest_e2e = nx.shortest_path(G, source=rpmu, target=LABEL_CC, weight=setup_calc_edge_weight(G, src=rpmu))
    
    try:
      if debug:
        print(f"--- Starting recursion for {rpmu}")
      path = choose(
        G, 
        T, 
        PMUs, 
        rpmu=rpmu, 
        parent=LABEL_CC,
        v=v, 
        R=R, 
        max_latency=max_latency,
        parent_latency=calc_path_cost(G, shortest_e2e),
        parchi_constraint=parchi_constraint,
        cc_successors_constraint=cc_successors_constraint,
        pdc_prio=pdc_prio,
        debug=debug
      )
      path = list(reversed([LABEL_CC, *path]))
    except ValueError as err:
      if debug:
        print(f"Error returned to {LABEL_CC}: {err}")
        print(f"No path found in Tree for {rpmu}, using shortest {LABEL_CC}-{rpmu}")
      path = list(shortest_e2e)
      
    if debug:
      print(f"{rpmu}: {path}")
      
    pmu_paths[rpmu] = {"path": path, "delay": calc_path_cost(G, path)}
    for node in path:
      if G.nodes[node].get(NODE_ROLE, ROLE_CANDIDATE) == ROLE_CANDIDATE:
        pdcs.add(node)

  return pdcs, pmu_paths

def choose(
  G: nx.Graph, 
  T: nx.DiGraph, 
  PMUs: list[str], 
  rpmu: str, 
  parent: str,
  v,
  R,
  max_latency: float = LATENCY_THRESHOLD,
  parent_latency: float = math.inf,
  best_node_above: list[tuple[float, str, float, bool]] = [],
  parchi_constraint: bool = True,
  cc_successors_constraint: bool = True,
  pdc_prio: int = PDC_PRIO_UNCHANGED,
  debug: bool = False,
):
  neighbors: list[str] = list(G.neighbors(parent))
  try:
    nodes: list[str] = list(T.successors(parent))
  except:
    nodes: list[str] = []
  
  if not cc_successors_constraint and parent == LABEL_CC:
    nodes = neighbors.copy()
  
  # Se è direttamente collegato al padre (il nodo chiamante) ho trovato percorso
  # (Caso base della ricorsione)
  if rpmu in nodes or rpmu in neighbors:
    return [rpmu]

  # Caso limite (percorso non trovato)
  if len(nodes) == 0:
    raise ValueError(f"Path not found for {rpmu}.")
  
  # Calcolo percorsi con Dijkstra dall'rPMU a tutti gli altri nodi
  paths = nx.shortest_path(G, source=rpmu, weight=setup_calc_edge_weight(G, src=rpmu))

  # Ogni tupla ha (delta_rel, nome_nodo, costo, pdc_già_presente)
  valid_nodes: list[tuple[float, str, float, bool]] = []

  # Valuto le latenze fino a ognuno dei nodi del livello attuale
  for node in nodes:
    # Se non è un candidato o un PDC, non va bene
    if G.nodes[node].get(NODE_ROLE, ROLE_CANDIDATE) != ROLE_CANDIDATE:
      continue

    # -- Latenza end-to-end CC-rPMU passando per Node --
    if T.has_node(node):
      path_from_cc = all_predecessors(T, node)
    elif node in neighbors: 
      path_from_cc = [LABEL_CC]
    else:
      continue
    # ^ Path CC - (Node)

    path_to_rpmu = list(reversed(paths[node]))   # Path Node - rPMU
    path_e2e = [*path_from_cc, *path_to_rpmu]
    latency_e2e = calc_path_cost(G, path_e2e)
    # -----
    
    if latency_e2e > max_latency:
      continue

    if parchi_constraint and latency_e2e > parent_latency:
      # Così consideriamo solo i figli che hanno latenza minore rispetto al padre
      # Passare per il padre (o attraverso un altro figlio) sarebbe meglio
      continue
    
    # Se è ok passare per esso, controlliamo il suo coefficiente di osservabilità
    # e calcoliamo la sua variazione tra prima e dopo aver aggiunto questo rPMU
    coeff_con_pmu = calc_coeff_v2(get_x_from_node(node, T, PMUs, rpmu), v, R)
    coeff_senza = calc_coeff_v2(get_x_from_node(node, T, PMUs), v, R)
    if coeff_senza == 0:
      delta_rel = coeff_con_pmu
    else:
      delta_rel = (coeff_con_pmu - coeff_senza) / coeff_senza
      
    if debug:
      print(f"rho({node}): {coeff_senza} -> {coeff_con_pmu} ({delta_rel})")

    # Salviamo il risultato tra i nodi validi
    valid_nodes.append((delta_rel, node, latency_e2e, coeff_senza > 0))

  if debug:
    print(f"all valid_nodes ({len(valid_nodes)}): {valid_nodes}")

  # Consideriamo solo i figli (`nodes`) che hanno la migliore variazione (relativa) del coefficiente di osservabilità,
  # E li ordiniamo per latenza crescente rPMU-nodo-CC (pos 2 nella tupla).
  valid_nodes = top_tied(valid_nodes)
  valid_nodes = sorted(valid_nodes, key=itemgetter(2))
  
  # Poi li ordiniamo in base a chi ha priorità (PDC o no PDC)
  if pdc_prio == PDC_PRIO_TRUE:
    valid_nodes = sorted(valid_nodes, key=itemgetter(3), reverse=True)  # Prima i True (PDC già presente)
  elif pdc_prio == PDC_PRIO_FALSE:
    valid_nodes = sorted(valid_nodes, key=itemgetter(3), reverse=False) # Prima i False (no PDC)
    
  if debug:
    print(f"remaining valid_nodes ({len(valid_nodes)}): {valid_nodes}")
  
  best_path: list[str] = None

  for _, node, cost, _ in valid_nodes:
    try:
      # Cerchiamo ricorsivamente il percorso tra i figli del nodo (nell'albero)
      if debug:
        print(f"Provo {node} sulla via per {rpmu}.")
      best_path = choose(
        G, 
        T, 
        PMUs, 
        rpmu=rpmu, 
        parent=node,
        v=v, 
        R=R, 
        max_latency=max_latency,
        parent_latency=cost,
        best_node_above=[*best_node_above, valid_nodes[0]],
        parchi_constraint=parchi_constraint,
        cc_successors_constraint=cc_successors_constraint,
        pdc_prio=pdc_prio,
        debug=debug)
      break
    except ValueError:
      # Se ha fallito la ricorsione
      if debug:
        print(f"Escludo {node} sulla via per {rpmu}.")
      continue

  if best_path is None:
    if len(valid_nodes) == 0:
      # Non ci sono nodi utili, e non c'è un path tra i figli
      raise ValueError(f"Path not found for {rpmu}.")
    
    # Best candidate at this layer
    _, node, cost, _ = valid_nodes[0]
    
    # Check if other layers have better candidates
    if best_node_above:
      best_nodes_for_layer = [*best_node_above, valid_nodes[0]]
      best_nodes_for_layer = top_tied(best_nodes_for_layer)
      best_nodes_for_layer = sorted(best_nodes_for_layer, key=itemgetter(2))
      
      if best_nodes_for_layer[0][1] != node:
        if debug:
          print(f"Best path is not here ({node}) for {rpmu}. It's {best_nodes_for_layer[0][1]}.")
        raise ValueError(f"Best path is not here ({node}) for {rpmu}.")
    
    best_path = list(reversed(paths[node]))
    best_path.pop(0)
    
    # In caso di parchi_constraint=False è possibile che il percorso migliore del figlio
    # passi per il parent, creando dei loop.
    if not parchi_constraint and parent in best_path:
      if debug:
        print(f"Best path for {rpmu} loops on parent {parent}.")
      raise ValueError(f"Best path for {rpmu} loops on parent {parent}.")
    
    if debug:
      print(f"Sono arrivato in fondo a {node} sulla via per {rpmu}, scelgo Dijkstra (path: {paths[node]}).")

  # Quando ritorna dalla ricorsione avremo il sottopath verso il rPMU
  # a cui anteporre il nodo scelto corrente
  return [node, *best_path]

## TODO: Path fanno splitting! Ci sta bene istanziare più PDC sullo stesso nodo?
# Oppure dobbiamo valutare quale tra i nodi padri possibili scegliere

def pmu_by_node(node: str, T: nx.DiGraph, PMUs: list[str]) -> list[str]:
  reachable = []

  for pmu in PMUs:
    try:
      # has_path looks for a path considering directional edges in T
      if nx.has_path(T, node, pmu):
        reachable.append(pmu)
    except nx.NodeNotFound:
      # rPMUs should not be counted until fixed in the Tree
      continue
  
  return reachable

def vec_idx_from_pmu_name(pmu_name: str) -> int:
  m = re.match(r"[r]?PMU(\d+)", pmu_name)
  if m:
    return int(m.group(1)) - 1
  return -1

def get_x_from_pmus(num, *pmus):
  x = np.zeros((num, 1), dtype=int)
  for pmu in pmus:
    x[vec_idx_from_pmu_name(pmu)] = 1

  return x

def get_x_from_node(node: str, T: nx.DiGraph, PMUs: list[str], added_pmu: str | None = None):
  pmus = pmu_by_node(node, T, PMUs)
  if added_pmu is None:
    return get_x_from_pmus(len(PMUs), *pmus)
  else:
    return get_x_from_pmus(len(PMUs), *pmus, added_pmu)

def calc_coeff_v2(x, v, R) -> float:
  sz = R.shape[0]
  selected = np.zeros_like(x)
  
  for i in range(sz):
    if x[i] == 0:
      continue
    
    useful = True
    for j in range(sz):
      if i != j and selected[j] and R[i, j] > 0:
        if v[j] < v[i]:
          useful = True
          selected[j] = 0
        else:
          useful = False
    
    if useful:
      selected[i] = 1
  
  rho = float((selected.transpose() @ v)[0,0])
  return rho

def prefix_tree_from_pmu_paths(pmu_paths: dict[str, dict[str, list[str] | float]]):
  rev_paths = [list(reversed(val["path"])) for val in pmu_paths.values()]
  return my_prefix_tree(rev_paths)

def my_prefix_tree(paths: list, root = LABEL_CC) -> nx.DiGraph:
  def get_children(paths):
    children: dict[types.Any, list] = {}
    # Populate dictionary with key(s) as the child/children of the root and
    # value(s) as the remaining paths of the corresponding child/children
    for path in paths:
      # If path is empty, we add an edge to the NIL node.
      if not path:
        # Nothing to do, no children here.
        continue
      child, *rest = path
      # `child` may exist as the head of more than one path in `paths`.
      children.setdefault(child, []).append(rest)
    return children

  # Initialize the prefix tree with the root node.
  tree = nx.DiGraph()
  tree.add_node(root, role=root)
  children = get_children(paths)
  stack = [(root, iter(children.items()))]
  while stack:
    parent, remaining_children = stack[-1]
    try:
      child, remaining_paths = next(remaining_children)
    # Pop item off stack if there are no remaining children
    except StopIteration:
      stack.pop()
      continue
    tree.add_node(child)
    tree.add_edge(parent, child)
    children = get_children(remaining_paths)
    stack.append((child, iter(children.items())))

  # Remove edges CC-CC created, since it appears in the paths
  tree.remove_edge(root, root)

  return tree

def calc_path_cost(G: nx.Graph, path: list[str]) -> float:
  total_cost = 0
  
  if not nx.is_path(G, path):
    raise nx.NetworkXNoPath("path does not exist")
  
  edge_weight = setup_calc_edge_weight(G, src=path[0])
  
  for u, v in zip(path, path[1:]):
    total_cost += edge_weight(u, v, G.edges[u,v])
  
  return total_cost

def setup_calc_edge_weight(G: nx.Graph, src: str):
  def calc_edge_weight(n1: str, _: str, edge_data) -> float:
    if n1 == src:
      return edge_data.get(EDGE_LATENCY, 1)
    return G.nodes[n1].get(NODE_LATENCY, 0) + edge_data.get(EDGE_LATENCY, 1)
  
  return calc_edge_weight

def all_predecessors(T: nx.DiGraph, node: str, root: str = LABEL_CC) -> list[str]:
  if node == root:
    return []
  
  predecessors = T.predecessors(node)
  for pred in predecessors:
    try:
      return [*all_predecessors(T, pred), pred]
    except:
      continue
  
  raise ValueError("Root not found.")

def top_tied(lst: list[tuple[float]], rel_tol: float = 1e-3):
  if not lst:
    return []
  top = max(t[0] for t in lst)
  return [t for t in lst if math.isclose(t[0], top, rel_tol=rel_tol)]

def parse_pmus(G: nx.Graph, essentialPMUs: list[str] | int) -> tuple[list[str], list[str], list[str]]:
  PMUs = []
  ePMUs = []
  rPMUs = []
  
  for n, data in G.nodes(data=True):
    if data.get(NODE_ROLE, ROLE_CANDIDATE) == ROLE_PMU:
      PMUs.append(n)
      
  if len(PMUs) == 0:
      raise ValueError("No PMUs found.")

  if isinstance(essentialPMUs, list):
    ePMUs = [pmu for pmu in PMUs if pmu in essentialPMUs]
    rPMUs = [pmu for pmu in PMUs if pmu not in essentialPMUs]
    if len(ePMUs) == 0:
      raise ValueError("No valid essential PMU found.")
    
  elif isinstance(essentialPMUs, int):
    if essentialPMUs <= 0:
      raise ValueError("essentialPMUs, as int, must be positive and non-zero.")
    
    ePMUs = [pmu for i, pmu in enumerate(PMUs) if i < essentialPMUs]
    rPMUs = [pmu for i, pmu in enumerate(PMUs) if i >= essentialPMUs]
    
    if len(ePMUs) == 0:
      raise ValueError("No valid essential PMU found.")
  else:
    raise ValueError("essentialPMUs must be of type list[str] or int")
  
  return (PMUs, ePMUs, rPMUs)