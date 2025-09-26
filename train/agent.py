import json, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

class QNet(nn.Module):
    def __init__(self, obs_dim, action_n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_n),
        )
    def forward(self, x):
        return self.net(x)

Transition = namedtuple("Transition", ("s", "a", "r", "s2", "d"))

class ReplayBuffer:
    def __init__(self, capacity:int):
        self.buf = deque(maxlen=capacity)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, batch_size:int):
        batch = random.sample(self.buf, batch_size)
        s = torch.as_tensor(np.stack([b.s for b in batch]), dtype=torch.float32)
        a = torch.as_tensor([b.a for b in batch], dtype=torch.int64).unsqueeze(1)
        r = torch.as_tensor([b.r for b in batch], dtype=torch.float32).unsqueeze(1)
        s2 = torch.as_tensor(np.stack([b.s2 for b in batch]), dtype=torch.float32)
        d = torch.as_tensor([b.d for b in batch], dtype=torch.float32).unsqueeze(1)
        return s, a, r, s2, d
    def __len__(self):
        return len(self.buf)

class HierarchicalDQN:
    """
    Single shared Q-network controlling a factored action (clusters thresholds + inter-keep option)
    """
    def __init__(self, obs_dim:int, action_n:int, cfg:dict, device="cpu"):
        self.obs_dim = obs_dim
        self.action_n = action_n
        self.device = device
        self.gamma = cfg["gamma"]
        self.batch_size = cfg["batch_size"]
        self.use_double = cfg.get("use_double_dqn", True)
        self.grad_clip = cfg.get("grad_clip_norm", 0.0)

        self.q = QNet(obs_dim, action_n).to(device)
        self.qt = QNet(obs_dim, action_n).to(device)
        self.qt.load_state_dict(self.q.state_dict())
        self.optim = optim.Adam(self.q.parameters(), lr=cfg["lr"])
        self.rb = ReplayBuffer(cfg["buffer_size"])

        self.eps = cfg["epsilon_start"]
        self.eps_end = cfg["epsilon_end"]
        self.eps_decay = cfg["epsilon_decay_steps"]
        self.step_count = 0

    def act(self, obs: np.ndarray) -> int:
        self.step_count += 1
        self.eps = max(self.eps_end, self.eps - (self.eps - self.eps_end)/max(1,self.eps_decay))
        if random.random() < self.eps:
            return random.randrange(self.action_n)
        with torch.no_grad():
            q = self.q(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            return int(q.argmax(dim=1).item())

    def push(self, s, a, r, s2, d):
        self.rb.push(s, a, r, s2, d)

    def train_step(self):
        if len(self.rb) < self.batch_size:
            return None
        s, a, r, s2, d = self.rb.sample(self.batch_size)
        s = s.to(self.device); a = a.to(self.device); r = r.to(self.device); s2 = s2.to(self.device); d = d.to(self.device)
        qsa = self.q(s).gather(1, a)

        with torch.no_grad():
            if self.use_double:
                next_actions = self.q(s2).argmax(dim=1, keepdim=True)
                q_next = self.qt(s2).gather(1, next_actions)
            else:
                q_next = self.qt(s2).max(dim=1, keepdim=True).values
            target = r + (1.0 - d) * self.gamma * q_next

        loss = nn.SmoothL1Loss()(qsa, target)
        self.optim.zero_grad()
        loss.backward()
        if self.grad_clip and self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.optim.step()
        return float(loss.item())

    def update_target(self):
        self.qt.load_state_dict(self.q.state_dict())

    def save(self, path:str):
        torch.save(self.q.state_dict(), path)

    def load(self, path:str, map_location="cpu"):
        self.q.load_state_dict(torch.load(path, map_location=map_location))
        self.qt.load_state_dict(self.q.state_dict())

# Simple vanilla DQN baseline with single global threshold action
class GlobalThresholdDQN(HierarchicalDQN):
    def __init__(self, obs_dim:int, cfg:dict, device="cpu"):
        # Define small discrete action space of thresholds [0.1..0.9]
        self.thresholds = np.linspace(0.1, 0.9, 9).astype(np.float32)
        super().__init__(obs_dim, action_n=len(self.thresholds), cfg=cfg, device=device)

    def decode(self, a:int) -> float:
        return float(self.thresholds[a]) 