import json, numpy as np, time
from env import SDNEnv
from algorithm import HierarchicalDQN, GlobalThresholdDQN

# ILP via pulp (CBC) — simplified: choose edges to keep to minimize energy while ensuring connectivity among active host pairs
try:
    import pulp
    HAVE_PULP = True
except Exception:
    HAVE_PULP = False

def eval_agent(env: SDNEnv, agent, episodes: int):
    metrics = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        tot_energy = 0.0
        tot_latency = 0.0
        tot_viol = 0
        steps = 0
        while not done:
            if isinstance(agent, GlobalThresholdDQN):
                a = agent.act(obs)
                # decode to threshold and apply via env helper (override step mechanics)
                # but to keep fair, we still call step with hierarchical action close to that thr.
                obs, r, done, info = env.step(a)  # same action space in this toy setup
            else:
                a = agent.act(obs)
                obs, r, done, info = env.step(a)
            tot_energy += info["energy"]
            tot_latency += info["latency_ms"]
            tot_viol += info["sla_viol"]
            steps += 1
        metrics.append((tot_energy/steps, tot_latency/steps, tot_viol/steps, info["energy_saving"]))
    arr = np.array(metrics)
    return {"energy": float(arr[:,0].mean()), "latency": float(arr[:,1].mean()), "sla_viol": float(arr[:,2].mean()), "energy_saving": float(arr[:,3].mean())}

def heuristic_baseline(env: SDNEnv, thr: float=0.4, episodes:int=3):
    metrics = []
    for _ in range(episodes):
        env.reset()
        done = False
        tot_energy=tot_latency=0.0
        tot_viol=0
        steps=0
        while not done:
            env.sleep_by_threshold(thr)
            # call a no-op action index 0 for accounting (won't matter since we already set links)
            obs, r, done, info = env.step(0)
            tot_energy += info["energy"]
            tot_latency += info["latency_ms"]
            tot_viol += info["sla_viol"]
            steps += 1
        metrics.append((tot_energy/steps, tot_latency/steps, tot_viol/steps, info["energy_saving"]))
    arr = np.array(metrics)
    return {"energy": float(arr[:,0].mean()), "latency": float(arr[:,1].mean()), "sla_viol": float(arr[:,2].mean()), "energy_saving": float(arr[:,3].mean())}

def ilp_baseline_once(env: SDNEnv):
    if not HAVE_PULP:
        return {"energy": None, "latency": None, "sla_viol": None, "energy_saving": None, "note": "PuLP not available"}
    # Build a single-step ILP: min sum(active_e) s.t. graph remains connected
    # (We don't model full multi-commodity flows here to keep it quick in Colab.)
    G = env.G_full
    E = list(G.edges())
    # Variables x_e in {0,1} whether edge e is active
    x = pulp.LpVariable.dicts("x", (range(len(E))), lowBound=0, upBound=1, cat="Binary")
    prob = pulp.LpProblem("MinActiveEdges", pulp.LpMinimize)
    # Objective: energy on cost
    prob += pulp.lpSum([x[i] * env.energy_on + (1-x[i])*env.energy_sleep for i in range(len(E))])
    # Connectivity constraints using cut formulation for a few random cuts
    nodes = list(G.nodes())
    rng = np.random.default_rng(0)
    for _ in range(min(20, len(nodes))):
        S = set(rng.choice(nodes, size=max(2, len(nodes)//4), replace=False))
        T = set(nodes) - S
        cut_edges = [i for i,(u,v) in enumerate(E) if (u in S and v in T) or (u in T and v in S)]
        if cut_edges:
            prob += pulp.lpSum([x[i] for i in cut_edges]) >= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10))
    # Apply solution and evaluate one episode quickly
    for i,(u,v) in enumerate(E):
        G[u][v]["active"] = int(pulp.value(x[i]) >= 0.5)
    # Evaluate one episode with fixed links (no changes)
    obs = env.reset()
    done=False
    tot_energy=tot_latency=0.0; tot_viol=0; steps=0
    while not done:
        # keep current link statuses; use step with action 0 (decoded inside)
        obs, r, done, info = env.step(0)
        tot_energy += info["energy"]; tot_latency += info["latency_ms"]; tot_viol += info["sla_viol"]; steps += 1
    return {"energy": tot_energy/steps, "latency": tot_latency/steps, "sla_viol": tot_viol/steps, "energy_saving": info["energy_saving"]}

def main(cfg_path="config.json"):
    cfg = json.load(open(cfg_path))
    env = SDNEnv(cfg)
    obs = env.reset()
    # HMARL
    h_agent = HierarchicalDQN(obs_dim=obs.shape[0], action_n=env.action_n, cfg=cfg, device=cfg.get("device","cpu"))
    # quick warmstart with few episodes
    for ep in range(5):
        obs = env.reset()
        done=False
        while not done:
            a = h_agent.act(obs)
            obs2, r, done, info = env.step(a)
            h_agent.push(obs, a, r, obs2, float(done))
            h_agent.train_step()
            obs = obs2
        h_agent.update_target()

    res_h = eval_agent(env, h_agent, episodes=cfg.get("eval_episodes",3))
    res_heur = heuristic_baseline(env, thr=0.4, episodes=cfg.get("eval_episodes",3))
    res_ilp = ilp_baseline_once(env)

    print("\n=== Comparison (avg per-step) ===")
    def fmt(x): return "n/a" if x is None else f"{x:.4f}"
    print(f"HMARL   | energy={fmt(res_h['energy'])} latency(ms)={fmt(res_h['latency'])} SLAviol={fmt(res_h['sla_viol'])} saving={fmt(res_h['energy_saving'])}")
    print(f"Heuristic(thr=0.4) | energy={fmt(res_heur['energy'])} latency(ms)={fmt(res_heur['latency'])} SLAviol={fmt(res_heur['sla_viol'])} saving={fmt(res_heur['energy_saving'])}")
    print(f"ILP(CBC, connectivity) | energy={fmt(res_ilp['energy'])} latency(ms)={fmt(res_ilp['latency'])} SLAviol={fmt(res_ilp['sla_viol'])} saving={fmt(res_ilp['energy_saving'])}")

if __name__ == "__main__":
    main()