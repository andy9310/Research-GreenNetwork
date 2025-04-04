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



def train(env, agent, episodes=3000, batch_size=128):
    """Train the DQN agent."""
    # Ensure directories exist
    os.makedirs("saved_models/3_switch_new", exist_ok=True)
    os.makedirs("image/3_switch_new", exist_ok=True)

    episode_rewards = []
    losses = []
    gradients = []

    for e in tqdm(range(episodes), desc="Training Progress"):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        for time in range(100):  # Limit the maximum steps per episode
            # print("state", state)
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                print(f"Episode {e+1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
                break

        episode_rewards.append(total_reward)

        # Train the agent with replay memory
        if len(agent.memory) > batch_size:
            loss, gradient_norm = agent.replay(batch_size)
            losses.append(loss)
            gradients.append(gradient_norm)

        # Update target model periodically
        if e % 50 == 0:
            agent.update_target_model()

        # Save model periodically
        if (e + 1) % 100 == 0:
            torch.save(agent.model.state_dict(), f"saved_models/3_switch_new/dqn_model_episode_{e+1}.pth")
            print(f"Model saved at episode {e+1}.")

            # Save plots
            plt.figure()
            plt.plot(episode_rewards)
            plt.xlabel("Episode")
            plt.ylabel("Accumulated Reward")
            plt.title("Accumulated Rewards Over Episodes")
            plt.savefig(f"image/3_switch_new/accumulated_rewards_{e+1}.png")
            plt.close()

            if losses:
                plt.figure()
                plt.plot(losses)
                plt.xlabel("Training Step")
                plt.ylabel("Loss")
                plt.title("Training Loss Over Time")
                plt.savefig(f"image/3_switch_new/training_loss_{e+1}.png")
                # plt.show()
                plt.close()

            # Plot gradient norms
            if gradients:
                plt.figure()
                plt.plot(gradients)
                plt.xlabel("Training Step")
                plt.ylabel("Gradient Norm")
                plt.title("Gradient Norm Over Time")
                plt.savefig(f"image/3_switch_new/gradient_norms_{e+1}.png")
                # plt.show()
                plt.close()

    # Save metrics
    np.save("image/3_switch_new/episode_rewards.npy", episode_rewards)
    np.save("image/3_switch_new/losses.npy", losses)
    np.save("image/3_switch_new/gradients.npy", gradients)

    # Final reward plot
    plt.figure()
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Accumulated Reward")
    plt.title("Accumulated Rewards Over Episodes")
    plt.savefig("image/3_switch_new/accumulated_rewards.png")
    plt.close()


def evaluate(env, agent, n_eval_episodes=50, max_steps=100, method="trained"):
    """Evaluate the DQN agent."""
    # agent.model.eval()  # Set the model to evaluation mode
    eval_rewards = []
    eval_energy_consumptions = []
    eval_connectivity_penalty = []
    eval_bandwidth_exceed = []
    eval_throughput = []

    for episode in range(n_eval_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        total_energy_consumption = 0
        total_connectivity_penalty = 0
        total_bandwidth_exceed_penalty = 0

        total_throughput = 0

        # Initialize variables for round-robin
        if method == "roundrobin":
            current_index = 0

        for step in range(max_steps):
            # Select action based on the method
            if method == "trained":
                action = agent.act_eval(state)
                
            elif method == "random":
                action = np.random.randint(agent.num_actions)
                # action = 63
            elif method == "roundrobin":
                action = current_index
                current_index = (current_index + 1) % agent.num_actions
            
            elif method == "greedy_off":
                action = 0
            
            elif method == "greedy_on":
                action = 2047

            # action = 2047
            # action = np.array([0,0,0,0,0,0,0,0,0,0,0])                    # -356              -2104000.00
            # action = np.array([0,1,0,0,0,0,0,0,0,0,0])    
            # action = np.array([1,0,1,1,1,1,1,1,1,1,1])                    # -300              -1879000.00
            # action = np.array([1,1,0,1,1,0,1,0,1,1,1])                    # -300              -1879000.00
            # action = np.array([1,1,1])
            # action = np.array([0,0,0])
            # action = np.array([1,0,1])

            # Apply action to the environment
            # action = np.array([1,1,0,1,1,0,1,0,1,1,1])
            # action = env.action_map[action]
            # print("action", action)
            # print(action)
            # action = 1372
            # print(action)
            # print(env.action_map[action])
            next_state, reward, done, info, info2 = env.step(action)
            state = np.reshape(next_state, [1, state_size])

            # Accumulate metrics
            total_reward += reward
            total_energy_consumption += info.get("energy", 0)
            total_connectivity_penalty += info.get("connectivity_penalty", 0)
            total_bandwidth_exceed_penalty += info.get("bandwidth_exceed_penalty", 0)
            total_throughput += info2

            if done:
                break



        eval_rewards.append(total_reward)
        eval_energy_consumptions.append(total_energy_consumption)
        eval_connectivity_penalty.append(total_connectivity_penalty)
        eval_bandwidth_exceed.append(total_bandwidth_exceed_penalty)

        total_energy_consumption = total_energy_consumption/100
        total_connectivity_penalty = total_connectivity_penalty/100
        total_bandwidth_exceed_penalty = total_bandwidth_exceed_penalty /100
        total_throughput = total_throughput/100
        eval_throughput.append(total_throughput)

        print(f"Evaluation Episode {episode+1}/{n_eval_episodes}, "
              f"Total Reward: {total_reward:.2f}, Energy: {total_energy_consumption:.2f}, Connectivity_penalty: {total_connectivity_penalty:.2f}, Bandwidth_exceed: {total_bandwidth_exceed_penalty:.2f}")
    
        # print(env.linecard_status)

    avg_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    avg_energy = np.mean(eval_energy_consumptions)/100
    std_energy = np.std(eval_energy_consumptions)/100
    max_energy = 3758
    save_energy = 1- (avg_energy/max_energy)
    avg_connectivity_penalty = np.mean(eval_connectivity_penalty)/100
    avg_bandwidth_exceed_penalty = np.mean(eval_bandwidth_exceed)/100

    average_throughpuy = np.mean(eval_throughput)

    print(f"\nEvaluation Summary:")
    print(f"Average Reward over {n_eval_episodes} episodes: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Energy Consumption over {n_eval_episodes} episodes: {avg_energy:.2f} ± {std_energy:.2f}")
    print(f"Average Saved Energy Consumption over {n_eval_episodes} episodes: {save_energy:.2f}")
    print(f"Average Connectivity_penalty over {n_eval_episodes} episodes: {avg_connectivity_penalty:.2f}")
    print(f"Average Bandwidth_exceed_penalty over {n_eval_episodes} episodes: {avg_bandwidth_exceed_penalty:.2f}")
    print(f"Average Throughput over {n_eval_episodes} episodes: {average_throughpuy:.2f}")

    # Save evaluation plot
    plt.figure()
    plt.plot(eval_rewards, marker='o', label="Reward")
    plt.plot(eval_energy_consumptions, marker='x', label="Energy")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Value")
    plt.title("Evaluation: Rewards and Energy Consumption")
    plt.legend()
    plt.savefig("image/3_switch_new/evaluation_rewards_energy.png")
    plt.close()
    print("evaluation done")


if __name__ == '__main__':
    set_seed(42)
    switches = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    target_switches_ = [4, 9, 10, 15]
    hosts = [f"h{h}" for h in range(1, 18)]
    links = {
            (1, 2): 7000, (2, 1): 7000, (1, 3): 7000, (3, 1): 7000, (1, 4): 7000, (4, 1): 7000, (1, 9): 7000, (9, 1): 7000, (2, 4): 7000, (4, 2): 7000,
            (2, 9): 7000, (9, 2): 7000, (3, 4): 7000, (4, 3): 7000, (3, 9): 7000, (9, 3): 7000, (4, 5): 7000, (5, 4): 7000, (4, 6): 7000, (6, 4): 7000,
            (4, 7): 7000, (7, 4): 7000, (4, 8): 7000, (8, 4): 7000, (4, 9): 7000, (9, 4): 7000, (4, 10): 7000, (10, 4): 7000, (4, 11): 7000, (11, 4): 7000,
            (4, 15): 7000, (15, 4): 7000, (5, 9): 7000, (9, 5): 7000, (6, 15): 7000, (15, 6): 7000, (7, 9): 7000, (9, 7): 7000, (8, 9): 7000, (9, 8): 7000,
            (9, 10): 7000, (10, 9): 7000, (9, 15): 7000, (15, 9): 7000, (10, 12): 7000, (12, 10): 7000, (10, 13): 7000, (13, 10): 7000, (10, 14): 7000,
            (14, 10): 7000, (10, 15): 7000, (15, 10): 7000, (10, 16): 7000, (16, 10): 7000, (10, 17): 7000, (17, 10): 7000, (11, 15): 7000, (15, 11): 7000,
            (12, 15): 7000, (15, 12): 7000, (13, 15): 7000, (15, 13): 7000, (14, 15): 7000, (15, 14): 7000, (15, 16): 7000, (16, 15): 7000,
            (15, 17): 7000, (17, 15): 7000
        }
    port_info = {
            (1, 2): 3, (2, 1): 2,(1, 3): 4, (3, 1): 2,(1, 4): 5, (4, 1): 2,(1, 9): 10, (9, 1): 2,(2, 4): 5, (4, 2): 3,(2, 9): 10, (9, 2): 3,(3, 4): 5, 
            (4, 3): 4,(3, 9): 10, (9, 3): 4,(4, 5): 6, (5, 4): 5,(4, 6): 7, (6, 4): 5,(4, 7): 8, (7, 4): 5,(4, 8): 9, (8, 4): 5,(4, 9): 10, (9, 4): 5,
            (4, 10): 11, (10, 4): 5,(4, 11): 12, (11, 4): 5,(4, 15): 16, (15, 4): 5,(5, 9): 10, (9, 5): 6,(6, 15): 16, (15, 6): 7,(7, 9): 10, (9, 7): 8,
            (8, 9): 10, (9, 8): 9,(9, 10): 11, (10, 9): 10,(9, 15): 16, (15, 9): 10,(10, 12): 13, (12, 10): 11,(10, 13): 14, (13, 10): 11,(10, 14): 15, 
            (14, 10): 11,(10, 15): 16, (15, 10): 11,(10, 16): 17, (16, 10): 11,(10, 17): 18, (17, 10): 11,(11, 15): 16, (15, 11): 12,(12, 15): 16, 
            (15, 12): 13,(13, 15): 16, (15, 13): 14,(14, 15): 16, (15, 14): 15,(15, 16): 17, (16, 15): 16,(15, 17): 18, (17, 15): 16,
        }
    env = NetworkEnvironment(switches, hosts, links, port_info, max_ports_per_linecard=4, fixed_throughput=100, target_switches=target_switches_)
    state_size = len(env.state)
    action_size = len(env.linecard_status)
    num_actions = 2**action_size #
    print("state_size", state_size)
    print("action_size", action_size)
    print("action_dimension", num_actions)

    # print("linecard_mapping", env.linecard_mapping)
    
    # Choose mode: "train" or "evaluate"
    run_mode = "evaluate"
    eva_method = "greedy_off"   # greedy_on, greedy_off, random, roundrobin
    agent = DQNAgent(state_size=state_size, num_actions=num_actions)
    


    
    if run_mode == "train":
        print("Trainning Begins")
        train(env, agent, episodes=2000, batch_size=512)

    elif run_mode == "evaluate":
        # Load a trained model
        print("Evaluation Begins")
        model_path = "saved_models/3_switch_new/dqn_model_episode_3000.pth"  # Update with your model path
        agent.model.load_state_dict(torch.load(model_path))
        evaluate(env, agent, method=eva_method)
    
    

    