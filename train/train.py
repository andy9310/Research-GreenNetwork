import json, os, time, numpy as np

from env import SDNEnv
from algorithm import HierarchicalDQN

def main(cfg_path="config.json"):
    cfg = json.load(open(cfg_path))
    env = SDNEnv(cfg)
    obs = env.reset()
    agent = HierarchicalDQN(obs_dim=obs.shape[0], action_n=env.action_n, cfg=cfg, device=cfg.get("device","cpu"))

    ep_rewards = []
    t_global = 0
    for ep in range(cfg["episodes"]):
        obs = env.reset()
        total_r = 0.0
        for t in range(cfg["max_steps_per_episode"]):
            a = agent.act(obs)
            obs2, r, done, info = env.step(a)
            agent.push(obs, a, r, obs2, float(done))
            obs = obs2
            total_r += r
            t_global += 1
            if t_global % cfg["train_every"] == 0:
                agent.train_step()
            if t_global % cfg["target_update"] == 0:
                agent.update_target()
            if done:
                break
        ep_rewards.append(total_r)
        if (ep+1) % max(1,cfg["log_every"]) == 0 or ep==0:
            print(f"[Episode {ep+1}/{cfg['episodes']}] reward={total_r:.3f} eps={agent.eps:.2f}")
    save_path = "hmarl_q.pt"
    agent.save(save_path)
    print(f"Saved model to {save_path}")
    print(f"Average reward over {len(ep_rewards)} episodes: {np.mean(ep_rewards):.3f}")
    return save_path

if __name__ == "__main__":
    main()