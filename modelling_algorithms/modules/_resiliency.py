from graph_model import EDGE_BANDWIDTH, EDGE_LATENCY, EDGE_STATUS_ONLINE, LABEL_CC, NODE_LATENCY, NODE_ROLE, NODE_STATUS, ROLE_CANDIDATE, ROLE_PMU, create_graph
from visualizer import draw_graph
from placement_pdc import place_pdcs_greedy
from operator import itemgetter
import networkx as nx
import numpy as np
import math, random, re, types

LATENCY_THRESHOLD = 200
RPMU_LINKS = 1
NUM_ePMU = 4
R = None

def main():
  # -- HELPERS --
  def add_edge(u, v):
    if u == v or G.has_edge(u, v):
      return
    G.add_edge(
      u, v,
      **{
        EDGE_LATENCY: round(random.uniform(1, 3), 2),
        EDGE_BANDWIDTH: 400,
        NODE_STATUS: EDGE_STATUS_ONLINE,
      }
    )

  # ---- SETUP ----
  # Crea grafo casuale G con M PMU (che saranno ePMU), CC e nodi liberi
  G = create_graph(seed=42, num_pmus=NUM_ePMU)
  # Eseguiamo algo greedy di Dario -> Tree
  (pdcs, pmu_paths, _) = place_pdcs_greedy(G, max_latency=LATENCY_THRESHOLD)
  rev_paths = [list(reversed(val["path"])) for val in pmu_paths.values()]
  T = my_prefix_tree(rev_paths)
  # Inizializza matrice di ridondanza
  global R
  R = np.zeros((2*NUM_ePMU, 2*NUM_ePMU), dtype=float)

  old = {
    "T": T.copy(),
    "G": G.copy(),
    "pdcs": pdcs.copy(), 
    "pmu_paths": pmu_paths.copy()
  }

  # ---- NEW ----
  PMUs = [f"PMU{i}" for i in range(1, NUM_ePMU + 1)]
  rPMUs = [f"rPMU{NUM_ePMU + j + 1}-({j + 1})" for j in range(0, NUM_ePMU)]
  PMUs.extend(rPMUs)
  
  for j in range(0, NUM_ePMU): #(nuovi rPMU)
    # Posiziona casualmente il PMU (ovvero collega casualmente a un nodo)
    rPMU = rPMUs[j]
    G.add_node(rPMU, **{ NODE_ROLE: ROLE_PMU })
    for n in random.sample([n for n, d in G.nodes(data=True) if d.get(NODE_ROLE, ROLE_CANDIDATE) == ROLE_CANDIDATE], RPMU_LINKS):
      add_edge(rPMU, n)
    
    # Riempiamo la matrice di ridondanza
    R[j][NUM_ePMU+j] = 1
    R[NUM_ePMU+j][j] = 1
    
    # Scegliamo il percorso migliore dal punto di vista dell'osservabilità
    # scendendo lungo l'albero creato dall'algoritmo di Dario
    try:
      path = choose(
        G, 
        T, 
        PMUs, 
        rpmu=rPMU, 
        nodes=list(T.successors(LABEL_CC)), 
        neighbors=list(G.neighbors(LABEL_CC)), 
        v=np.ones((2*NUM_ePMU, 1), dtype=int),
        R=R,
        max_latency=LATENCY_THRESHOLD,
        parent_latency=nx.shortest_path_length(G, source=rPMU, target=LABEL_CC, weight=setup_calc_edge_weight(G, src=rPMU)),
      )
      path = [LABEL_CC, *path]
      pmu_paths[rPMU] = {"path": path, "delay": calc_path_cost(G, path)} # TODO: Set delay?
      print(f"{rPMU}: {path}")
    except ValueError as e:
      print(e)
  
  pos = None
  try:
    pos = nx.nx_pydot.pydot_layout(G, prog="dot")
  except:
    pos = nx.spring_layout(G, seed=42)

  # Grafici iniziali (posizionamento di Dario)
  draw_graph(old["G"], None, {}, max_latency=None, output_path="output/0-graph-empty.png", pos=pos)
  draw_graph(old["G"], old["pdcs"], old["pmu_paths"], output_path="output/1-graph-dario.png", pos=pos)
  draw_graph(old["T"], old["pdcs"], old["pmu_paths"], output_path="output/2-tree-dario.png", pos=pos)

  # Singoli percorsi per PMU
  for j, (pmu, data) in enumerate(pmu_paths.items()):
    only_path = {pmu: data}
    draw_graph(G, pdcs, only_path, output_path=f"output/3.{j}-graph-resilient.png", pos=pos)
  
  # Tutti percorsi insieme
  draw_graph(G, pdcs, pmu_paths, output_path="output/3.z-graph-resilient.png", pos=pos)

  # Coppie ridondanti
  items = list(pmu_paths.items())
  for j in range(len(items) // 2):
    epmu, data_e = items[j]
    rpmu, data_r = items[NUM_ePMU + j]
    only_paths = {epmu: data_e, rpmu: data_r}
    draw_graph(G, pdcs, only_paths, output_path=f"output/4.{j}-graph-resilient.png", pos=pos)

def place_pdcs_resiliently(
  G: nx.Graph, 
  max_latency: float, 
  essentialPMUs: list[str] | int, 
  v, 
  R, 
  debug: bool = False
):
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
  
  greedyG = G.copy()
  greedyG.remove_nodes_from(rPMUs)

  (pdcs, pmu_paths, _) = place_pdcs_greedy(greedyG, max_latency, False)
  T = prefix_tree_from_pmu_paths(pmu_paths)

  for rpmu in rPMUs:
    try:
      path = choose(
        G, 
        T, 
        PMUs, 
        rpmu=rpmu, 
        nodes=list(T.successors(LABEL_CC)), 
        neighbors=list(G.neighbors(LABEL_CC)), 
        v=v, 
        R=R, 
        max_latency=max_latency,
        parent_latency=nx.shortest_path_length(G, source=rpmu, target=LABEL_CC, weight=setup_calc_edge_weight(G, src=rpmu)),
        debug=debug
      )
      path = list(reversed([LABEL_CC, *path]))
      pmu_paths[rpmu] = {"path": path, "delay": calc_path_cost(G, path)}
      if debug:
        print(f"{rpmu}: {path}")
      
      for node in path:
        if G.nodes[node].get(NODE_ROLE, ROLE_CANDIDATE) == ROLE_CANDIDATE:
          pdcs.add(node)
    except ValueError as e:
      print(e)

  return pdcs, pmu_paths

def choose(
  G: nx.Graph, 
  T: nx.DiGraph, 
  PMUs: list[str], 
  rpmu: str, 
  nodes: list[str], 
  neighbors: list[str], 
  v,
  R,
  max_latency: float = LATENCY_THRESHOLD,
  parent_latency: float = math.inf,
  debug: bool = False,
):
  # Se è direttamente collegato al padre (il nodo chiamante) ho trovato percorso
  # (Caso base della ricorsione)
  if rpmu in nodes or rpmu in neighbors:
    return [rpmu]
  
  # Calcolo percorsi con Dijkstra da PMUs[i] a tutti gli altri nodi
  costs, paths = nx.multi_source_dijkstra(
    G, 
    sources={rpmu}, 
    weight=setup_calc_edge_weight(G, src=rpmu), 
    cutoff=max_latency
  )

  # Caso limite (percorso non trovato)
  if len(nodes) == 0:
    raise ValueError(f"Path not found for {rpmu}.")

  valid_nodes: list[tuple[float, str, float]] = []

  # Valuto le latenze fino a ognuno dei nodi del livello attuale
  for node in nodes:
    # Se non è tra i path, vuol dire che è stato tagliato fuori per il constraint sulla latenza
    if node not in paths.keys():
      continue

    # Se non è un candidato o un PDC, non va bene
    if G.nodes[node].get(NODE_ROLE, ROLE_CANDIDATE) != ROLE_CANDIDATE:
      continue

    # Latenza da PMUs[i] al node
    latency = costs[node]

    # Consideriamo solo i figli che hanno latenza minore rispetto al padre
    if latency > parent_latency:
      # Passare per il padre (o attraverso un altro figlio) sarebbe meglio
      continue
    
    # Se è ok passare per esso, controlliamo il suo coefficiente di osservabilità
    # e calcoliamo la sua variazione tra prima e dopo aver aggiunto questo rPMU
    coeff_con_pmu = calc_coeff_v2(get_x_from_node(node, T, PMUs, rpmu), v, R)
    coeff_senza = calc_coeff_v2(get_x_from_node(node, T, PMUs), v, R)
    if coeff_senza == 0:
      delta_rel = math.inf
    else:
      delta_rel = (coeff_con_pmu - coeff_senza) / coeff_senza
      
    if debug:
      print(f"rho({node}): {coeff_senza} -> {coeff_con_pmu} ({delta_rel})")

    # Salviamo il risultato tra i nodi validi
    valid_nodes.append((delta_rel, node, latency))

  # Consideriamo in ordine le migliori variazioni (relative) del coefficiente di osservabilità,
  # e la PMU sarà associata a quel nodo.
  # A parità di coefficiente (posizione 0 della tupla), si ordina per costo dal PMU[i] (pos 2).
  valid_nodes = sorted(valid_nodes, key=itemgetter(0,2), reverse=True)
  best_path: list[str] = None

  for _, node, cost in valid_nodes:
    try:
      # Cerchiamo ricorsivamente il percorso tra i figli del nodo (nell'albero)
      if debug:
        print(f"Provo {node} sulla via per {rpmu}.")
      best_path = choose(
        G, 
        T, 
        PMUs, 
        rpmu=rpmu, 
        nodes=list(T.successors(node)), 
        neighbors=list(G.neighbors(node)), 
        v=v, 
        R=R, 
        parent_latency=cost, 
        debug=debug)
      break
    except:
      # Se ha fallito la ricorsione
      if debug:
        print(f"Escludo {node} sulla via per {rpmu}.")
      continue

  if best_path is None:
    _, node, _ = valid_nodes[0]
    best_path = list(reversed(paths[node]))
    best_path.pop(0)
    if debug:
      print(f"Sono arrivato in fondo a {node} sulla via per {rpmu}, scelgo Dijkstra (path: {paths[node]}).")

  # Quando ritorna dalla ricorsione avremo il sottopath verso il rPMU
  # a cui anteporre il nodo scelto corrente
  return [node, *best_path]

## TODO: Path fanno splitting! Ci sta bene istanziare più PDC sullo stesso nodo?
# Oppure dobbiamo valutare quale tra i nodi padri possibili scegliere

def pmu_by_node(node: str, T: nx.DiGraph, PMUs: list[str]) -> list[str]:
  T2 = T.copy()
  T2.remove_node(LABEL_CC)

  reachable = []
  for pmu in PMUs:
    try:
      if nx.has_path(T2, node, pmu):
        reachable.append(pmu)
        # TODO: Controlla che non possa tornare indietro salendo sull'albero
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

if __name__ == "__main__":
  main()