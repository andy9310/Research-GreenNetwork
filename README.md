# Network Optimization with Reinforcement Learning

This project implements various reinforcement learning approaches for network optimization tasks. It supports Deep Q-Learning (DQN) and Monte Carlo methods with multiple neural network architectures and configuration files.

## Architecture Options

The training script supports three different neural network architectures:

| Architecture | Description | Recommended Hidden Dim |
|--------------|-------------|------------------------|
| `mlp` | Standard Multi-Layer Perceptron with 3 layers | 256 |
| `fat_mlp` | Wider and deeper MLP with 5 layers and layer normalization | 512 |
| `transformer` | Transformer-based architecture with self-attention | 256 |

## Usage

### Training

To train a model with a specific architecture:

```bash
python Qlearning/train.py --config configs/config_name.json --architecture [mlp|fat_mlp|transformer] --hidden-dim [256|512]
```

#### Important Parameters:

- `--architecture`: Choose between `mlp`, `fat_mlp`, or `transformer` (default: transformer)
- `--hidden-dim`: Size of hidden layers (recommended: 256 for mlp/transformer, 512 for fat_mlp)
- `--config`: Path to the configuration file
- `--episodes`: Number of episodes to train per traffic matrix (default: 3000)
- `--verbose`: Enable detailed output during training
- `--load-model`: Continue training from a previously saved model
- `--gpu`: Use GPU for training if available

## Quick Start Guide

This section provides practical commands for quickly getting started with training and evaluating models using the smaller 5-node topology.

### Training Q-Learning

```bash
# Activate GPU environment if available
conda activate rl_hw3_new

# Train using MLP architecture
python Qlearning/train.py --config configs/config_5node.json --architecture mlp --gpu

# Train using fat MLP architecture
python Qlearning/train.py --config configs/config_5node.json --architecture fat_mlp --gpu

# Train using Transformer architecture
python Qlearning/train.py --config configs/config_5node.json --architecture transformer --gpu
```

### Training Monte Carlo

```bash
# Train Monte Carlo with same configuration
python MonteCarlo/train.py --config configs/config_5node.json --architecture mlp --gpu
```

### Training with Traffic Matrix Representation Learning

```bash
# Train using Traffic Matrix Encoder for better generalization across traffic matrices
python train_tm_encoder.py --config configs/extended_config_5node.json --tm-rounds 5 --episodes 500 --tm-embedding-dim 64

# Evaluate model trained with Traffic Matrix Encoder
python evaluate_tm_encoder.py --config configs/extended_config_5node.json --model-path models/tm_dqn_mlp_extended_config_5node.pth
```

### Evaluating Models

```bash
# Evaluate Q-Learning on different traffic matrices
python Qlearning/evaluate.py --config configs/config_5node.json --architecture mlp --tm-index 0
python Qlearning/evaluate.py --config configs/config_5node.json --architecture mlp --tm-index 1
python Qlearning/evaluate.py --config configs/config_5node.json --architecture mlp --tm-index 2
python Qlearning/evaluate.py --config configs/config_5node.json --architecture mlp --tm-index 3

# Evaluate Monte Carlo method
python MonteCarlo/evaluate.py --config configs/config_5node.json --architecture mlp
```

### Comparing Both Methods

```bash
# Run comparison with specific parameters
python compare_methods.py --config configs/config_5node.json --architecture mlp --episodes-per-tm 2000 --epsilon-min 0.1 --epsilon-decay-steps 5000
```

### Running Bruteforce Optimization

```bash
# Find optimal configuration using bruteforce
python testing_configs/run_bruteforce_tests.py --config configs/config_5node.json
```

### Evaluation

To evaluate a trained model, use the respective evaluation script for each approach:

#### Q-Learning Evaluation

```bash
python Qlearning/evaluate.py --config configs/config_name.json --architecture [mlp|fat_mlp|transformer]
```

#### Monte Carlo Evaluation

```bash
python MonteCarlo/evaluate.py --config configs/config_name.json --model-path models/monte_carlo_transformer_config_name.pth
```

Make sure to use the same architecture and configuration file that was used during training.

### Bruteforce Optimization

The project includes bruteforce scripts to find optimal link configurations for smaller networks. This serves as a ground truth for evaluating the performance of learning-based approaches.

#### Running Bruteforce on a Configuration

```bash
python testing_configs/run_bruteforce_tests.py --config configs/config_name.json
```

This will exhaustively search all possible link configurations (2^n where n is the number of links) to find the optimal configuration that maximizes energy savings while satisfying network constraints.

#### Bruteforce Results

Results are saved to a JSON file with the format `bruteforce_results_config_name.json`. For each traffic matrix, it provides:
- The optimal link configuration (which links to close/open)
- Number of links that can be safely closed
- Maximum achievable score (energy saved)
- Number of valid configurations found

### Comparing Methods

To compare the learning performance of Monte Carlo and Q-Learning approaches side by side:

```bash
python compare_methods.py --config config_name.json --architecture [mlp|fat_mlp|transformer] --episodes-per-tm 1000 --epsilon-min 0.2 --epsilon-decay-steps 5000
```

#### Comparison Parameters

- `--architecture`: Neural network architecture to use (applies to both methods)
- `--hidden-dim`: Size of hidden layers (default: 256)
- `--episodes-per-tm`: Number of episodes to train per traffic matrix (default: 1000)
- `--epsilon-start`: Starting exploration rate (default: 1.0)
- `--epsilon-min`: Minimum exploration rate (default: 0.1)
- `--epsilon-decay-steps`: Number of steps to decay epsilon (default: 10000)
- `--learning-rate`: Learning rate for both methods (default: 1e-4)
- `--gamma`: Discount factor (default: 0.99)

#### Comparison Results

The script generates visualizations in the `visualizations/` directory:
- Reward comparison curves
- Success rate comparison (episodes without violations)
- Average links closed in successful episodes
- Early termination rate comparison
- Performance radar chart 

This provides clear empirical evidence of the differences in learning behavior between Monte Carlo and Q-learning methods, particularly in environments with early termination.

## Configuration Files

The project supports multiple configuration files for different network topologies:

- `config_5node.json`: A smaller 5-node topology for quicker testing
- `config_17node_25edges.json`: A larger 17-node topology with 25 edges

## Model Files

Trained models are saved in the `models/` directory with filenames that include both the architecture and configuration name:

```
models/dqn_[architecture]_[config_name].pth
```

For example: `dqn_fat_mlp_config_5node.pth` or `monte_carlo_transformer_config_5node.pth`

## Monte Carlo vs. Q-Learning for Network Optimization

This project implements both Monte Carlo and Q-Learning approaches for the network optimization task. Each has distinct advantages and challenges when dealing with network constraints and violations.

### Monte Carlo Learning Challenges

When implementing early termination for network violations (which is necessary as violation states are unacceptable in real networks), Monte Carlo methods face several challenges:

1. **Incomplete Exploration**: When episodes terminate early due to violations, the agent doesn't experience the consequences of its actions for the remaining links in the network, leading to a partial view of the state-action space.

2. **Credit Assignment Problem**: Monte Carlo methods update Q-values based on the total return from a complete episode. With early termination, the agent only observes penalties for violations and never experiences potential rewards from successfully closing links later in the sequence.

3. **Sparse Learning Signal**: Most episodes end with negative rewards (penalties), creating a risk-averse behavior where the agent may prefer to keep all links open rather than risk violations.

4. **Sequence Dependency**: Network optimization decisions about early links affect the feasibility of closing later links. With early termination, the agent doesn't learn these dependencies effectively.

5. **Limited Success Examples**: With immediate termination, few episodes reach completion (especially early in training), giving the agent limited examples of successful network configurations.

### Q-Learning Advantages

Q-learning offers several advantages in this environment:

1. **Bootstrap Learning**: Q-learning can update value estimates from partial episodes using the bootstrapping mechanism: `Q(s,a) = Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]`. This allows learning from episodes that terminate early with violations.

2. **Efficient Experience Use**: While Monte Carlo wastes most early-termination experiences, Q-learning extracts value from every step through TD updates, learning even from failed attempts.

3. **Targeted Credit Assignment**: TD error is calculated at each step rather than propagated from episode end, helping identify exactly which actions lead to violations.

4. **Off-Policy Learning**: Q-learning can learn the optimal policy while following an exploratory policy, which is crucial when successful episodes are rare.

5. **Value Propagation**: Q-learning can propagate good outcomes backward through the value function, gradually building a map of safe state-action pairs even with sparse rewards.

### Practical Implications

For network optimization tasks with hard constraints (where violations must trigger immediate termination):

- Q-learning approaches like DQN are generally more effective than Monte Carlo methods
- If using Monte Carlo, consider enhancing it with techniques like reward shaping, curriculum learning, or imitation learning
- The advantage of Q-learning becomes more pronounced as the network size and complexity increase

## Traffic Matrix Representation Learning

Traffic Matrix Representation Learning is an advanced approach implemented in this project to help the model generalize across different traffic patterns. This is particularly valuable in network optimization, where the space of possible traffic matrices is infinite and training on all possible patterns is infeasible.

### Key Concepts

1. **Neural Encoding of Traffic Matrices**: Instead of treating each traffic matrix as a unique input, a neural encoder extracts meaningful features and patterns from the traffic matrix.

2. **Embedding Vectors**: Traffic matrices are compressed into fixed-size embedding vectors that capture their essential characteristics and structure.

3. **Combined State Representation**: These traffic matrix embeddings are combined with the network state for richer input to the Q-network.

4. **Pattern Recognition**: The model learns to recognize underlying patterns in traffic distribution rather than memorizing specific traffic matrices.

### Implementation Details

The implementation consists of:

- **TrafficMatrixEncoder**: A neural network that transforms traffic matrices into embeddings.
- **EnhancedQNetwork**: A Q-network that operates on both state and traffic matrix embeddings.
- **TMEnhancedDQNAgent**: An agent that incorporates traffic matrix awareness into the training process.

### Training Process

During training, the model:

1. Cycles through available traffic matrices in shuffled order across multiple rounds
2. Learns to extract relevant features from each traffic matrix
3. Develops a policy that generalizes to traffic patterns rather than specific matrices

### Benefits

- **Better Generalization**: Models trained with this approach can handle new, unseen traffic matrices
- **Transfer Learning**: Knowledge learned from one traffic pattern transfers to similar patterns
- **Robust Performance**: Less vulnerability to specific traffic matrix characteristics
- **Adaptability**: Can better respond to changing traffic conditions in the network

This approach is particularly valuable for real-world deployments where traffic patterns evolve over time.
