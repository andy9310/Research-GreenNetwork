from network_simulation import NetworkEnvironment
from dqn_agent import DQNAgent
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

def set_seed(seed):
    random.seed(seed)  # Fix Python's random module seed
    np.random.seed(seed)  # Fix NumPy seed
    torch.manual_seed(seed)  # Fix PyTorch seed
    torch.cuda.manual_seed(seed)  # If using CUDA
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic
    torch.backends.cudnn.benchmark = False  # Avoid non-deterministic optimizations

switches = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
hosts = [f"h{h}" for h in range(1, 18)]
links = {
        (1, 2): 1500, (2, 1): 1500, (1, 3): 1500, (3, 1): 1500, (1, 4): 1500, (4, 1): 1500, (1, 9): 1500, (9, 1): 1500, (2, 4): 1500, (4, 2): 1500,
        (2, 9): 1500, (9, 2): 1500, (3, 4): 1500, (4, 3): 1500, (3, 9): 1500, (9, 3): 1500, (4, 5): 1500, (5, 4): 1500, (4, 6): 1500, (6, 4): 1500,
        (4, 7): 1500, (7, 4): 1500, (4, 8): 1500, (8, 4): 1500, (4, 9): 1500, (9, 4): 1500, (4, 10): 1500, (10, 4): 1500, (4, 11): 1500, (11, 4): 1500,
        (4, 15): 1500, (15, 4): 1500, (5, 9): 1500, (9, 5): 1500, (6, 15): 1500, (15, 6): 1500, (7, 9): 1500, (9, 7): 1500, (8, 9): 1500, (9, 8): 1500,
        (9, 10): 1500, (10, 9): 1500, (9, 15): 1500, (15, 9): 1500, (10, 12): 1500, (12, 10): 1500, (10, 13): 1500, (13, 10): 1500, (10, 14): 1500,
        (14, 10): 1500, (10, 15): 1500, (15, 10): 1500, (10, 16): 1500, (16, 10): 1500, (10, 17): 1500, (17, 10): 1500, (11, 15): 1500, (15, 11): 1500,
        (12, 15): 1500, (15, 12): 1500, (13, 15): 1500, (15, 13): 1500, (14, 15): 1500, (15, 14): 1500, (15, 16): 1500, (16, 15): 1500,
        (15, 17): 1500, (17, 15): 1500
    }
port_info = {
        (1, 2): 3, (2, 1): 2,(1, 3): 4, (3, 1): 2,(1, 4): 5, (4, 1): 2,(1, 9): 10, (9, 1): 2,(2, 4): 5, (4, 2): 3,(2, 9): 10, (9, 2): 3,(3, 4): 5, 
        (4, 3): 4,(3, 9): 10, (9, 3): 4,(4, 5): 6, (5, 4): 5,(4, 6): 7, (6, 4): 5,(4, 7): 8, (7, 4): 5,(4, 8): 9, (8, 4): 5,(4, 9): 10, (9, 4): 5,
        (4, 10): 11, (10, 4): 5,(4, 11): 12, (11, 4): 5,(4, 15): 16, (15, 4): 5,(5, 9): 10, (9, 5): 6,(6, 15): 16, (15, 6): 7,(7, 9): 10, (9, 7): 8,
        (8, 9): 10, (9, 8): 9,(9, 10): 11, (10, 9): 10,(9, 15): 16, (15, 9): 10,(10, 12): 13, (12, 10): 11,(10, 13): 14, (13, 10): 11,(10, 14): 15, 
        (14, 10): 11,(10, 15): 16, (15, 10): 11,(10, 16): 17, (16, 10): 11,(10, 17): 18, (17, 10): 11,(11, 15): 16, (15, 11): 12,(12, 15): 16, 
        (15, 12): 13,(13, 15): 16, (15, 13): 14,(14, 15): 16, (15, 14): 15,(15, 16): 17, (16, 15): 16,(15, 17): 18, (17, 15): 16,
    }
env = NetworkEnvironment(switches, hosts, links, port_info, max_ports_per_linecard=4, fixed_throughput=100, target_switches=[4, 9, 10, 15])

state_size = len(env.state)
action_size = len(env.linecard_status)
print("state_size", state_size)
print("action_size", action_size)

agent = DQNAgent(state_size=state_size, action_size=action_size)
episodes = 3000
batch_size = 128
set_seed(42)

# Ensure the directory exists
os.makedirs("saved_models", exist_ok=True)
os.makedirs("image", exist_ok=True)
episode_rewards = []
losses = []
gradients = []


# Training loop
for e in tqdm(range(episodes), desc="Training Progress"):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time in range(500):  # Limit the maximum steps per episode
        # Agent selects an action
        action = agent.act(state)

        # Environment applies the action
        # print("action", action)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        # Store the experience
        agent.remember(state, action, reward, next_state, done)

        # Update the state and reward
        state = next_state
        total_reward += reward

        if done:
            print(f"Episode {e+1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
            break
    
    episode_rewards.append(total_reward)




    # Train the agent with replay memory
    if len(agent.memory) > batch_size:
        
        loss, gradient_norm = agent.replay(batch_size)  # Replay returns loss and gradient norm
        losses.append(loss)
        gradients.append(gradient_norm)
    else:
        print("no")

    # Update target model periodically
    if e % 10 == 0:
        agent.update_target_model()

    # Save model periodically
    if (e + 1) % 100 == 0:  # Corrected condition
        torch.save(agent.model.state_dict(), f"saved_models/dqn_model_episode_{e+1}.pth")
        print(f"Model saved at episode {e+1}.")
        plt.figure()
        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Accumulated Reward")
        plt.title("Accumulated Rewards Over Episodes")
        plt.savefig(f"image/accumulated_rewards_{e+1}.png")
        # plt.show()
        plt.close()

        if losses:
            plt.figure()
            plt.plot(losses)
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.title("Training Loss Over Time")
            plt.savefig(f"image/training_loss_{e+1}.png")
            # plt.show()
            plt.close()

        # Plot gradient norms
        if gradients:
            plt.figure()
            plt.plot(gradients)
            plt.xlabel("Training Step")
            plt.ylabel("Gradient Norm")
            plt.title("Gradient Norm Over Time")
            plt.savefig(f"image/gradient_norms_{e+1}.png")
            # plt.show()
            plt.close()





# Save rewards, losses, and gradients to a file for further analysis
np.save("image/episode_rewards.npy", episode_rewards)
np.save("image/losses.npy", losses)
np.save("image/gradients.npy", gradients)

# Plot accumulated rewards over episodes
plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Accumulated Reward")
plt.title("Accumulated Rewards Over Episodes")
plt.savefig("image/accumulated_rewards.png")
# plt.show()
plt.close()

# Plot training loss
if losses:
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.savefig("image/training_loss.png")
    # plt.show()
    plt.close()

# Plot gradient norms
if gradients:
    plt.figure()
    plt.plot(gradients)
    plt.xlabel("Training Step")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm Over Time")
    plt.savefig("image/gradient_norms.png")
    # plt.show()
    plt.close()