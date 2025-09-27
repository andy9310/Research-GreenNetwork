import numpy as np
import networkx as nx
from typing import Dict, Tuple, List
from dataclasses import dataclass
import random

from cluster import dynamic_clustering
from algorithm import DeterministicLinkManager  # New import

@dataclass
class Flow:
    s: int
    t: int
    size: float
    prio: int
    ttl: int  # steps remaining

class SDNEnv:
    """
    Simplified SDN environment for hierarchical MARL with dynamic clustering.
    - Graph with capacities and base delays.
    - Stochastic flows; routing via k-shortest paths with congestion delay.
    - Actions: per-cluster utilization thresholds + inter-cluster min edges to keep.
    - Deterministic refinement: protect high-priority flows by force-keeping links on their paths.
    """
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.rng = random.Random(cfg.get("seed", 42))
        np.random.seed(cfg.get("seed", 42))
        self.n = cfg["num_nodes"]
        self.num_edges = cfg["num_edges"]
        self.num_regions = cfg["num_regions"]
        self.num_hosts = cfg["num_hosts"]
        self.energy_on = cfg["energy_per_link_on"]
        self.energy_sleep = cfg["energy_per_link_sleep"]
        self.k_paths = cfg["routing_k_paths"]
        self.cong_factor = cfg["congestion_delay_factor"]
        self.recluster_every = cfg["recluster_every_steps"]
        self.cluster_bins = np.array(cfg["cluster_threshold_bins"], dtype=float)
        self.inter_keep_opts = cfg["inter_cluster_keep_min"]
        # self.prio_weights = {int(k): v for k, v in cfg["priority_weights"].items()}
        self.sla_latency_ms = {int(k): v for k, v in cfg["sla_latency_ms"].items()}

        # Initialize deterministic link manager
        algorithm_type = cfg.get("deactivation_algorithm", "greedy")
        self.link_manager = DeterministicLinkManager(algorithm_type)

        self._build_topology()
        self._assign_regions_and_hosts()
        self._time = 0
        self._flows: List[Flow] = []
        self._cluster_map: Dict[int, int] = {}
        self._last_action = None

        # Observation / action sizes
        self.num_clusters = self.num_regions  # we keep k=regions for clarity
        self.action_n = self.num_clusters * len(self.cluster_bins) + len(self.inter_keep_opts)  # thresholds for each cluster + one global inter-keep choice
        self.obs_dim = self.num_clusters * 6 + 3  # [traffic_in, traffic_out, active_links, mean_util, svc_hi_share, svc_lo_share]*K + inter_summary(3)

    def _build_topology(self):
        # Random geometric graph for spatial flavor
        pos = {i: (np.random.rand(), np.random.rand()) for i in range(self.n)}
        G = nx.gnm_random_graph(self.n, self.num_edges, seed=self.cfg.get("seed", 42))
        # Ensure connectivity
        if not nx.is_connected(G):
            comps = list(nx.connected_components(G))
            for i in range(len(comps)-1):
                a = random.choice(list(comps[i]))
                b = random.choice(list(comps[i+1]))
                G.add_edge(a, b)
        for u, v in G.edges():
            cap = max(0.5, np.random.normal(self.cfg["edge_capacity_mean"], self.cfg["edge_capacity_std"]))
            delay = self.cfg["edge_base_delay_ms"] * (1.0 + 0.5*np.random.rand())
            G[u][v]["capacity"] = cap
            G[u][v]["delay_ms"] = delay
            G[u][v]["active"] = 1
        self.G_full = G

    def _assign_regions_and_hosts(self):
        # Partition nodes into regions by node index for reproducibility
        per = self.n // self.num_regions
        self.region_of = {}
        for i in range(self.n):
            self.region_of[i] = min(i // per, self.num_regions-1)

        # choose hosts from nodes with degree>=1
        candidates = [i for i in self.G_full.nodes() if self.G_full.degree(i) > 0]
        self.hosts = random.sample(candidates, k=min(self.num_hosts, len(candidates)))
        # map hosts to edge nodes (themselves for this abstraction)
        self.host_nodes = self.hosts

    def reset(self) -> np.ndarray:
        # Reset edge states active=1
        for u, v in self.G_full.edges():
            self.G_full[u][v]["active"] = 1
        self._time = 0
        self._flows = []
        self._clusterize()
        return self._observe()

    def _clusterize(self):
        # Build a quick traffic matrix + svc share snapshot for clustering
        n = self.n
        tm = np.zeros((n, n), dtype=float)
        svcC = 6
        svc_share = np.ones((n, svcC), dtype=float)
        svc_share /= svc_share.sum(axis=1, keepdims=True)
        self._cluster_map = dynamic_clustering(self.G_full, tm, svc_share, k=self.num_clusters, seed=self.cfg.get("seed", 42))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Action is a flat index -> thresholds per cluster + one inter-keep option"""
        thresholds, inter_keep = self._decode_action(action)
        
        # Use the new deterministic algorithm system
        self._apply_enhanced_action(thresholds, inter_keep)

        # Generate flows for this step
        self._generate_new_flows()

        # Route flows and compute latency/energy
        latency_ms, sla_viol, utilization = self._route_and_measure()

        energy = self._energy_cost()
        # Reward: energy saving positive, latency & SLA viol negative (weighted)
        base_all_on = self.energy_on * self.G_full.number_of_edges()
        energy_saving = (base_all_on - energy) / base_all_on
        penalty = 0.002 * latency_ms + 0.02 * sla_viol  # scaled down
        reward = energy_saving - penalty

        self._time += 1
        if self._time % self.recluster_every == 0:
            self._clusterize()

        obs = self._observe(utilization=utilization)
        done = self._time >= self.cfg["max_steps_per_episode"]
        info = {"energy": energy, "latency_ms": latency_ms, "sla_viol": sla_viol, "energy_saving": energy_saving}
        self._last_action = action
        return obs, reward, done, info

    def _decode_action(self, a: int):
        binsK = len(self.cluster_bins)
        # thresholds per cluster chosen by taking a in base-binsK
        thresholds_idx = []
        for _ in range(self.num_clusters):
            thresholds_idx.append(a % binsK)
            a //= binsK
        inter_idx = a % len(self.inter_keep_opts)
        thresholds = [float(self.cluster_bins[i]) for i in thresholds_idx]
        return thresholds, self.inter_keep_opts[inter_idx]

    def _apply_enhanced_action(self, thresholds: List[float], inter_keep_min: int):
        """Use the new deterministic algorithm system"""
        # Apply the complete deterministic deactivation strategy
        self.link_manager.apply_complete_deactivation_strategy(
            graph=self.G_full,
            region_of=self.region_of,
            thresholds=thresholds,
            inter_keep_min=inter_keep_min,
            flows=self._flows
        )

    def _generate_new_flows(self):
        # Determine which region is in peak based on time
        current_step = self._time
        peaks = self.cfg["peaks"]
        host_by_region = {r: [] for r in range(self.num_regions)}
        for h in self.hosts:
            host_by_region[self.region_of[h]].append(h)
        for r in range(self.num_regions):
            lo, hi = peaks[f"region_{r}"]
            in_peak = (current_step % self.cfg["max_steps_per_episode"]) in range(lo, hi)
            if in_peak:
                interval = self.cfg["peak_flow_interval_s"]
                size_rng = self.cfg["peak_size_range"]
            else:
                interval = self.cfg["offpeak_flow_interval_s"]
                size_rng = self.cfg["offpeak_size_range"]

            # Probability to spawn a flow this step ~ 1/avg_interval
            p = 1.0 / float(sum(interval)/2.0)
            if random.random() < p and len(host_by_region[r]) >= 2:
                s, t = random.sample(host_by_region[r], 2)
                size = random.uniform(size_rng[0], size_rng[1])
                prio = random.randint(1,6)
                ttl = random.randint(3, 8)  # flow lasts a few steps
                self._flows.append(Flow(s=s, t=t, size=size, prio=prio, ttl=ttl))

        # decrement TTL & remove expired
        alive = []
        for f in self._flows:
            f.ttl -= 1
            if f.ttl > 0:
                alive.append(f)
        self._flows = alive

    def _shortest_path_active(self, s: int, t: int):
        # Use only active edges
        H = nx.Graph()
        for u, v, data in self.G_full.edges(data=True):
            if data["active"] == 1:
                H.add_edge(u, v, delay_ms=data["delay_ms"], capacity=data["capacity"])
        if not H.has_node(s) or not H.has_node(t):
            return None
        try:
            return nx.shortest_path(H, s, t, weight="delay_ms")
        except nx.NetworkXNoPath:
            return None

    def _route_and_measure(self) -> Tuple[float, int, Dict[Tuple[int,int], float]]:
        # Simple routing: route each flow along current shortest path (active edges only)
        # accumulate per-edge utilization and compute per-flow latency
        util = {(u, v): 0.0 for u, v in self.G_full.edges()}
        total_latency_ms = 0.0
        sla_viol = 0

        # prebuild active subgraph
        H = nx.Graph()
        for u, v, data in self.G_full.edges(data=True):
            if data["active"] == 1:
                H.add_edge(u, v, delay_ms=data["delay_ms"], capacity=data["capacity"])

        for f in self._flows:
            try:
                path = nx.shortest_path(H, f.s, f.t, weight="delay_ms")
            except Exception:
                path = None
            if path is None:
                # penalize unserved flow: large latency + SLA violation
                total_latency_ms += 50.0
                sla_viol += 1
                continue

            # base latency
            base = 0.0
            min_cap = float("inf")
            edges = []
            for u, v in zip(path, path[1:]):
                if u > v: u, v = v, u
                d = self.G_full[u][v]
                base += d["delay_ms"]
                min_cap = min(min_cap, d["capacity"])
                edges.append((u, v))

            # add congestion penalty based on temp utilization
            # we approximate utilization by incrementing edge loads by flow "size"
            # size is in bytes; convert to unitless load via /100
            flow_load = f.size / 100.0
            edge_delay = 0.0
            for (u, v) in edges:
                util[(u, v)] += flow_load
                cap = self.G_full[u][v]["capacity"]
                ufrac = util[(u, v)] / max(1e-6, cap)
                if ufrac <= 1.0:
                    extra = 0.0
                else:
                    extra = self.cong_factor * (ufrac - 1.0) * self.G_full[u][v]["delay_ms"]
                edge_delay += extra

            flow_latency = base + edge_delay
            total_latency_ms += flow_latency
            # SLA check
            if flow_latency > self.sla_latency_ms[f.prio]:
                sla_viol += 1

        # normalize util to fraction
        util_frac = {}
        for (u, v), load in util.items():
            cap = self.G_full[u][v]["capacity"]
            util_frac[(u, v)] = load / max(cap, 1e-6)
        return total_latency_ms, sla_viol, util_frac

    def _energy_cost(self) -> float:
        on = 0
        off = 0
        for _, _, d in self.G_full.edges(data=True):
            if d["active"] == 1:
                on += 1
            else:
                off += 1
        return on * self.energy_on + off * self.energy_sleep

    def _observe(self, utilization: Dict[Tuple[int,int], float] = None) -> np.ndarray:
        # Aggregate features per cluster
        K = self.num_clusters
        feats = []
        G = self.G_full
        for c in range(K):
            nodes = [i for i in range(self.n) if self.region_of[i] == c]
            sub = G.subgraph(nodes).copy()
            active_links = sum(1 for _,_,d in sub.edges(data=True) if d["active"] == 1)
            mean_util = 0.0
            m = 0
            if utilization is not None:
                for u, v in sub.edges():
                    u_, v_ = (u, v) if u < v else (v, u)
                    if (u_, v_) in utilization:
                        mean_util += utilization[(u_, v_)]
                        m += 1
            mean_util = (mean_util / m) if m > 0 else 0.0
            # traffic summaries (approximate by number of active flows involving nodes in c)
            tin = sum(1 for f in self._flows if f.t in nodes)
            tout = sum(1 for f in self._flows if f.s in nodes)
            # svc shares: hi (prio<=2) vs lo
            hi = sum(1 for f in self._flows if ((f.s in nodes or f.t in nodes) and f.prio <= 2))
            lo = sum(1 for f in self._flows if ((f.s in nodes or f.t in nodes) and f.prio >= 5))
            feats.extend([float(tin), float(tout), float(active_links), float(mean_util), float(hi), float(lo)])

        # Inter-cluster summary: number of active edges between each pair averaged
        inter_active = 0
        inter_total = 0
        for u, v, d in G.edges(data=True):
            if self.region_of[u] != self.region_of[v]:
                inter_total += 1
                if d["active"] == 1:
                    inter_active += 1
        inter_ratio = (inter_active / inter_total) if inter_total > 0 else 1.0
        # time-of-day enc (phase)
        phase = (self._time % self.cfg["max_steps_per_episode"]) / max(1, self.cfg["max_steps_per_episode"]-1)
        last_a = float(self._last_action if self._last_action is not None else 0) / float(max(1, self.action_n-1))
        feats.extend([inter_ratio, phase, last_a])
        return np.array(feats, dtype=np.float32)

    # Convenience for baselines
    def set_all_active(self):
        for u, v in self.G_full.edges():
            self.G_full[u][v]["active"] = 1

    def sleep_by_threshold(self, thr: float):
        G = self.G_full
        deg = {i: max(1, G.degree(i)) for i in G.nodes()}
        for u, v in G.edges():
            proxy_util = 0.5*(1.0/deg[u] + 1.0/deg[v])
            G[u][v]["active"] = 1 if proxy_util >= thr else 0