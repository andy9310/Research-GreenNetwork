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


def evaluate_linecard_decision(linecard_mapping, decision):
    """
    Evaluate which linecards are on or off based on a given decision array.

    Args:
        linecard_mapping (dict): Mapping of switches to their linecards and ports.
        decision (list): Array where 1 indicates the linecard is on, and 0 indicates off.

    Returns:
        dict: A dictionary indicating the status of each switch's linecards and their ports.
    """
    result = {}
    decision_index = 0

    for switch, linecards in linecard_mapping.items():
        result[switch] = {}
        for linecard, ports in linecards.items():
            # Check if we have a corresponding decision value for this linecard
            if decision_index < len(decision):
                status = "on" if decision[decision_index] == 1 else "off"
                result[switch][linecard] = {
                    "status": status,
                    "ports": ports  # Always include ports regardless of status
                }
                decision_index += 1
            else:
                raise ValueError("Decision array does not match the number of linecards.")
    return result

def _create_action_map(num_actions):
        """Map indices (0-7) to their corresponding binary action vectors."""
        action_map = []
        for i in range(num_actions):
            action_map.append(np.array([int(x) for x in f"{i:08b}"]))  # Convert index to binary
        # for idx, action in enumerate(actions):
        #     print(f"Action Index: {idx}, Action Vector: {action}")  # Debugging print
        return action_map


def set_seed(seed):
    random.seed(seed)  # Fix Python's random module seed
    np.random.seed(seed)  # Fix NumPy seed
    torch.manual_seed(seed)  # Fix PyTorch seed
    torch.cuda.manual_seed(seed)  # If using CUDA
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic
    torch.backends.cudnn.benchmark = False  # Avoid non-deterministic optimizations


def evaluate(env, agent, n_eval_episodes=1, max_steps=100, method="trained"):
    """Evaluate the DQN agent."""
    # agent.model.eval()  # Set the model to evaluation mode
    eval_rewards = []
    eval_energy_consumptions = []
    eval_connectivity_penalty = []
    eval_bandwidth_exceed = []

    for episode in range(n_eval_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        total_energy_consumption = 0
        total_connectivity_penalty = 0
        total_bandwidth_exceed_penalty = 0

        # Initialize variables for round-robin
        if method == "roundrobin":
            current_index = 0

        for step in range(max_steps):
            # Select action based on the method
            if method == "trained":
                print("AI mode")
                print("current state is:", state)
                action = agent.act_eval(state)
                print("action index:", action)
                print("action is:", action_map[action])

                # helpful to see which card on/off
                result = evaluate_linecard_decision(env.linecard_mapping, action_map[action])
                for switch, linecards in result.items():
                    print(f"Switch {switch}:")
                    for linecard, details in linecards.items():
                        print(f"  Linecard {linecard} is {details['status']}, Ports: {details['ports']}")
                
            elif method == "random":
                print("random mode")
                print("current state is:", state)
                action = np.random.randint(agent.num_actions)
                print("action index:", action)
                print("action is:", action_map[action])
                # action = 63

                 # helpful to see which card on/off
                result = evaluate_linecard_decision(env.linecard_mapping, action_map[action])
                for switch, linecards in result.items():
                    print(f"Switch {switch}:")
                    for linecard, details in linecards.items():
                        print(f"  Linecard {linecard} is {details['status']}, Ports: {details['ports']}")
                

            elif method == "roundrobin":
                print("roundrobin mode")
                print("current state is:", state)
                action = current_index
                current_index = (current_index + 1) % agent.num_actions
                print("action is:", action_map[current_index])

                 # helpful to see which card on/off
                result = evaluate_linecard_decision(env.linecard_mapping, action_map[current_index])
                for switch, linecards in result.items():
                    print(f"Switch {switch}:")
                    for linecard, details in linecards.items():
                        print(f"  Linecard {linecard} is {details['status']}, Ports: {details['ports']}")
                

                
            
            elif method == "greedy_off":
                print("greedy_off mode: all off")
                action = 0
                print("action is:", action_map[action])
            
            elif method == "greedy_on":
                print("greedy_on mode: all on")
                action = 255
                print("action is:", action_map[action])
            
            print()
            print()
            next_state, reward, done, info = env.step(action)
            state = np.reshape(next_state, [1, state_size])

            # Accumulate metrics
            total_reward += reward
            total_energy_consumption += info.get("energy", 0)
            total_connectivity_penalty += info.get("connectivity_penalty", 0)
            total_bandwidth_exceed_penalty += info.get("bandwidth_exceed_penalty", 0)

            if done:
                break


        eval_rewards.append(total_reward)
        eval_energy_consumptions.append(total_energy_consumption)
        eval_connectivity_penalty.append(total_connectivity_penalty)
        eval_bandwidth_exceed.append(total_bandwidth_exceed_penalty)

        total_energy_consumption = total_energy_consumption/100
        total_connectivity_penalty = total_connectivity_penalty/100
        total_bandwidth_exceed_penalty = total_bandwidth_exceed_penalty /100

        print(f"Evaluation Episode {episode+1}/{n_eval_episodes}, "
              f"Total Reward: {total_reward:.2f}, Energy: {total_energy_consumption:.2f}, Connectivity_penalty: {total_connectivity_penalty:.2f}, Bandwidth_exceed: {total_bandwidth_exceed_penalty:.2f}")
    
    

    avg_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    avg_energy = np.mean(eval_energy_consumptions)/100
    std_energy = np.std(eval_energy_consumptions)/100
    max_energy = 2806
    save_energy = 1- (avg_energy/max_energy)
    avg_connectivity_penalty = np.mean(eval_connectivity_penalty)/100
    avg_bandwidth_exceed_penalty = np.mean(eval_bandwidth_exceed)/100

    print(f"\nEvaluation Summary:")
    print(f"Average Reward over {n_eval_episodes} episodes: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Energy Consumption over {n_eval_episodes} episodes: {avg_energy:.2f} ± {std_energy:.2f}")
    print(f"Average Saved Energy Consumption over {n_eval_episodes} episodes: {save_energy:.2f}")
    print(f"Average Connectivity_penalty over {n_eval_episodes} episodes: {avg_connectivity_penalty:.2f}")
    print(f"Average Bandwidth_exceed_penalty over {n_eval_episodes} episodes: {avg_bandwidth_exceed_penalty:.2f}")


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
    env = NetworkEnvironment(switches, hosts, links, port_info, max_ports_per_linecard=4, fixed_throughput=100, target_switches=[4, 9, 10])
    num_actions = 2**8  #

    action_map = _create_action_map(num_actions)
    # print(action_map)

    state_size = len(env.state)
    action_size = len(env.linecard_status)
    print("state_size", state_size)
    print("action_size", action_size)
    print("num_actions", num_actions)
    print("linecard_mapping:", env.linecard_mapping)
    
    # Choose mode: "train" or "evaluate"
    run_mode = "evaluate"
    eva_method = "trained"
    agent = DQNAgent(state_size=state_size, num_actions=num_actions)
    


    if run_mode == "evaluate":
        # Load a trained model
        print("Evaluation Begins")
        model_path = "saved_models/3_switch_new/dqn_model_episode_3000.pth"  # Update with your model path
        agent.model.load_state_dict(torch.load(model_path))
        evaluate(env, agent, method=eva_method)
    

    