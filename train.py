import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 定義Q網路
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Replay Buffer (經驗回放)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# 定義超參數
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

lr = 1e-3
gamma = 0.99
batch_size = 64
buffer_size = 10000
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
target_update_freq = 10
episodes = 300

# 初始化網路
q_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=lr)
buffer = ReplayBuffer(buffer_size)
criterion = nn.MSELoss()

epsilon = epsilon_start

# 訓練主迴圈
for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        # ε-greedy 動作選擇
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = q_net(state_tensor)
            action = q_values.argmax().item()

        next_state, reward, done, _ = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # 訓練 Q 網路
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.LongTensor(actions).unsqueeze(1)
            rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
            next_states_tensor = torch.FloatTensor(next_states)
            dones_tensor = torch.FloatTensor(dones).unsqueeze(1)

            # 計算目標 Q 值
            current_q = q_net(states_tensor).gather(1, actions_tensor)
            next_q = target_net(next_states_tensor).max(1)[0].unsqueeze(1)
            target_q = rewards_tensor + gamma * next_q * (1 - dones_tensor)

            # 更新 Q 網路
            loss = criterion(current_q, target_q.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    # 更新epsilon
    epsilon = max(epsilon_end, epsilon_decay * epsilon)

    # 定期更新目標網路
    if episode % target_update_freq == 0:
        target_net.load_state_dict(q_net.state_dict())

    print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

env.close()
