# DQN Network Optimization

This project implements Deep Q-Learning for network optimization tasks. It supports multiple neural network architectures and configuration files.

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

### Evaluation

To evaluate a trained model:

```bash
python Qlearning/evaluate.py --config configs/config_name.json --architecture [mlp|fat_mlp|transformer]
```

Make sure to use the same architecture and configuration file that was used during training.

## Configuration Files

The project supports multiple configuration files for different network topologies:

- `config_5node.json`: A smaller 5-node topology for quicker testing
- `config_17node_25edges.json`: A larger 17-node topology with 25 edges

## Model Files

Trained models are saved in the `models/` directory with filenames that include both the architecture and configuration name:

```
models/dqn_[architecture]_[config_name].pth
```

For example: `dqn_fat_mlp_config_5node.pth`
