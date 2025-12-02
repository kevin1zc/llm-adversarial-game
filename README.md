# Multi-Turn Censorship Game

A game-theoretic framework for training robust LLM safety policies against multi-turn jailbreak attacks.

## Overview

This project implements a two-player game between:
- **Adversary**: Attempts multi-turn jailbreaks (gradual escalation, distraction, etc.)
- **Defender**: Learns to detect and refuse harmful requests while maintaining utility

Training uses self-play and equilibrium learning to find robust policies for both agents.

## Installation

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv venv && source .venv/bin/activate
uv sync
```

## Quick Start

```bash
# Train with default config
python train.py

# Train with specific mode
python train.py --mode equilibrium

# Train with custom experiment name
python train.py --name my_experiment

# Monitor training with TensorBoard
tensorboard --logdir logs
```

## Training

### Command Line Options

```bash
python train.py [OPTIONS]

Options:
  --config PATH       Config file (default: configs/default.yaml)
  --mode MODE         Training mode: self_play | equilibrium
  --name NAME         Experiment name for logging
  --no-tensorboard    Disable TensorBoard logging
```

### Training Modes

| Mode | Description |
|------|-------------|
| `self_play` | Train defender against fixed/heuristic adversary |
| `equilibrium` | Train both agents to find Nash equilibrium |

### TensorBoard

Training metrics are logged to TensorBoard automatically:

```bash
# Start TensorBoard (in separate terminal)
tensorboard --logdir logs

# Open http://localhost:6006 in browser
```

Logged metrics:
- `defender_reward`, `adversary_reward`
- `violation_rate`, `refusal_rate`
- `defender_loss`, `adversary_loss`
- `exploitability` (equilibrium mode)

## Configuration

Edit `configs/default.yaml`:

```yaml
training:
  mode: "equilibrium"      # or "self_play"
  num_episodes: 10000
  batch_size: 16           # Reduce if GPU OOM
  defender_lr: 0.00001
  adversary_lr: 0.00001
  eval_interval: 500

game:
  max_turns: 10
  safety_threshold: 0.5
```

## Project Structure

```
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── example.py            # Demo game
├── configs/
│   └── default.yaml      # Configuration
└── src/
    ├── config.py         # Typed config loader
    ├── game/
    │   ├── environment.py   # Game state & dynamics
    │   ├── adversary.py     # Attack agent
    │   ├── defender.py      # Defense agent
    │   └── rewards.py       # Reward functions
    ├── training/
    │   ├── self_play.py     # Defender-only training
    │   ├── equilibrium.py   # Two-player equilibrium learning
    │   └── logger.py        # TensorBoard & console logging
    ├── models/
    │   └── safety_classifier.py
    ├── data/
    │   └── dataset.py       # HuggingFace dataset loading
    └── evaluation/
        └── metrics.py
```

## Datasets

Supports loading from HuggingFace:
- `ScaleAI/mhj` - Multi-turn Human Jailbreaks
- `marslabucla/XGuard-Train` - XGuard training set (requires `huggingface-cli login`)

## Evaluation Metrics

- **Violation Rate**: % of episodes with safety violations
- **Refusal Rate**: % of refused requests
- **Exploitability**: Distance from Nash equilibrium
- **Episode Length**: Average conversation turns
