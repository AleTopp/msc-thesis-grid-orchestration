import networkx as nx
import random

LABEL_CC = "CC"
LABEL_PMU = lambda i : f"PMU{i}"
LABEL_CANDIDATE = lambda i : f"N{i}"
NODE_ROLE = "role"
ROLE_CC = "CC"
ROLE_PMU = "PMU"
ROLE_CANDIDATE = "candidate"
NODE_LATENCY = "processing"
NODE_STATUS = "status"
NODE_STATUS_ONLINE = "online"
NODE_STATUS_OFFLINE = "offline"
NODE_DATARATE = "data_rate"
EDGE_LATENCY = "latency"
EDGE_STATUS_ONLINE = "up"
EDGE_STATUS_OFFLINE = "down"
EDGE_BANDWIDTH = "bandwidth"

DEFAULT_NODE_LATENCY = 21.5

def create_graph(
    num_candidates=8,
    num_pmus=4,
    seed=None,
    p_extra=0.35,          # probability to add extra edge between candidates (beyond the spanning tree)
    cc_min_links=2,        # minimum links from CC to candidates (at least 1)
    cc_max_links=None,     # maximum links from CC to candidates (None = up to num_candidates)
    pmu_links=1,           # links from each PMU to candidates
    edge_latency_min=1,
    edge_latency_max=3,
):
    if seed is not None:
        random.seed(seed)

    G = nx.Graph()

    # --- nodes ---
    G.add_node(
        LABEL_CC,
        **{
            NODE_ROLE: ROLE_CC,
            NODE_LATENCY: DEFAULT_NODE_LATENCY,
            NODE_STATUS: NODE_STATUS_ONLINE
        }
    )

    candidates = [LABEL_CANDIDATE(i) for i in range(1, num_candidates + 1)]
    for n in candidates:
        G.add_node(
            n,
            **{
                NODE_ROLE: ROLE_CANDIDATE,
                NODE_LATENCY: DEFAULT_NODE_LATENCY,
                NODE_STATUS: NODE_STATUS_ONLINE
            }
        )

    pmus = []
    for i in range(1, num_pmus + 1):
        pmus.append(LABEL_PMU(i))
    for p in pmus:
        G.add_node(
            p,
            **{
                NODE_ROLE: ROLE_PMU,
                NODE_DATARATE: 100, 
                NODE_STATUS: NODE_STATUS_ONLINE
            }
        )

    # helper to add an edge with random latency and default bandwidth/status
    def add_edge(u, v):
        if u == v or G.has_edge(u, v):
            return
        G.add_edge(
            u, v,
            **{
                EDGE_LATENCY: round(random.uniform(edge_latency_min, edge_latency_max), 2),
                EDGE_BANDWIDTH: 400,
                NODE_STATUS: EDGE_STATUS_ONLINE,
            }
        )

    # --- 1) Subgraph ---
    # (a) Before, create a random spanning tree among candidates to ensure connectivity
    shuffled = candidates[:]
    random.shuffle(shuffled)
    for i in range(1, len(shuffled)):
        u = shuffled[i]
        v = random.choice(shuffled[:i])  
        add_edge(u, v)

    # (b) Add extra edges between candidates with probability p_extra
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if random.random() < p_extra:
                add_edge(candidates[i], candidates[j])

    # --- 2) Link CC to candidates ---
    if cc_max_links is None:
        cc_max_links = num_candidates
    cc_min_links = max(1, min(cc_min_links, num_candidates))
    cc_max_links = max(cc_min_links, min(cc_max_links, num_candidates))

    k = random.randint(cc_min_links, cc_max_links)
    for n in random.sample(candidates, k):
        add_edge(LABEL_CC, n)

    # --- 3) Link PMUs to candidates ---
    pmu_links = max(1, min(pmu_links, num_candidates))
    for p in pmus:
        for n in random.sample(candidates, pmu_links):
            add_edge(p, n)
    

    return G

def _edge_key(u: str, v: str) -> tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def modify_latency(G):
    ops_applied = []

    while True:
        print("\n🔗 Actually latency:")
        for u, v, data in G.edges(data=True):
            print(f"{u} – {v}: {data[EDGE_LATENCY]} ms")

        risposta = input("\nDo you want to modify a latency? (y/n): ").lower()
        if risposta != "y":
            break

        u = input("Node 1: ").strip()
        v = input("Node 2: ").strip()

        if G.has_edge(u, v):
            try:
                before = float(G[u][v].get(EDGE_LATENCY))
                nuova_latenza = float(input(f"Enter new latency for edge {u}–{v}: "))
                G[u][v][EDGE_LATENCY] = nuova_latenza
                print(f"✔️ Latency updated for {u}–{v} to {nuova_latenza} ms.")

                uu, vv = _edge_key(u, v)
                ops_applied.append({
                    "type": "latency",
                    "u": uu,
                    "v": vv,
                    "before": before,
                    "after": nuova_latenza,
                })

            except ValueError:
                print("❌ Invalid value.")
        else:
            print("❌ The specified edge does not exist.")

    return ops_applied


def modify_edge_status(G):
    """
    Interactive: optionally modifies edge status.
    Returns a list of applied operations:
      {"type":"status","u":..., "v":..., "before":..., "after":...}
    """
    ops_applied = []

    while True:
        print("\n🔗 Current edge statuses:")
        for u, v, data in G.edges(data=True):
            print(f"{u} – {v}: {data[NODE_STATUS]}")

        risposta = input("\nDo you want to modify the status of an edge? (y/n): ").lower()
        if risposta != "y":
            break

        u = input("Node 1: ").strip()
        v = input("Node 2: ").strip()

        if G.has_edge(u, v):
            before = str(G[u][v].get(NODE_STATUS))
            nuovo_stato = input(f"Enter new status for edge {u}–{v} (up/down): ").strip().lower()
            if nuovo_stato in [EDGE_STATUS_ONLINE, EDGE_STATUS_OFFLINE]:
                G[u][v][NODE_STATUS] = nuovo_stato
                print(f"✔️ Status updated for {u}–{v} to {nuovo_stato}.")

                uu, vv = _edge_key(u, v)
                ops_applied.append({
                    "type": "status",
                    "u": uu,
                    "v": vv,
                    "before": before,
                    "after": nuovo_stato,
                })
            else:
                print("❌ Invalid status. Use 'up' or 'down'.")
        else:
            print("❌ The specified edge does not exist.")

    return ops_applied


def modify_bandwidth(G):
    
    ops_applied = []

    while True:
        print("\n🔗 Current bandwidths:")
        for u, v, data in G.edges(data=True):
            print(f"{u} – {v}: {data[EDGE_BANDWIDTH]} kbps")

        risposta = input("\nDo you want to modify a bandwidth? (y/n): ").lower()
        if risposta != "y":
            break

        u = input("Node 1: ").strip()
        v = input("Node 2: ").strip()

        if G.has_edge(u, v):
            try:
                before = float(G[u][v].get(EDGE_BANDWIDTH))
                nuova_bandwidth = float(input(f"Enter new bandwidth for edge {u}–{v}: "))
                G[u][v][EDGE_BANDWIDTH] = nuova_bandwidth
                print(f"✔️ Bandwidth updated for {u}–{v} to {nuova_bandwidth} kbps.")

                uu, vv = _edge_key(u, v)
                ops_applied.append({
                    "type": "bandwidth",
                    "u": uu,
                    "v": vv,
                    "before": before,
                    "after": nuova_bandwidth,
                })

            except ValueError:
                print("❌ Invalid value.")
        else:
            print("❌ The specified edge does not exist.")

    return ops_applied
