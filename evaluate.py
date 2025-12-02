"""Evaluation script for multi-turn censorship game."""

import argparse
import json
import random
from pathlib import Path
import torch

from src.config import load_config
from src.game.environment import CensorshipGameEnvironment
from src.game.adversary import AdversaryAgent
from src.game.defender import DefenderAgent
from src.game.rewards import RewardFunction
from src.data.dataset import load_datasets, load_xguard_splits, JailbreakDataset
from src.training.self_play import SelfPlayTrainer
from src.evaluation.metrics import EvaluationMetrics


def evaluate(defender, adversary, env, datasets, num_samples=500):
    """Evaluate the defender agent."""
    trainer = SelfPlayTrainer(
        defender=defender,
        adversary=adversary,
        reward_fn=RewardFunction(),
        env=env,
    )

    all_examples = []
    for dataset in datasets:
        all_examples.extend(dataset.get_examples())

    print(f"Evaluating on {num_samples} samples...")
    episode_results = [
        trainer.train_episode(random.choice(all_examples) if all_examples else None)
        for _ in range(num_samples)
    ]

    metrics = EvaluationMetrics().compute_metrics(episode_results)
    return metrics, episode_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-turn censorship game")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_path", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--dataset", type=str, default="mhj")
    parser.add_argument("--num_samples", type=int, default=500)
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Initialize components
    env = CensorshipGameEnvironment(
        max_turns=config.game.max_turns,
        safety_threshold=config.game.safety_threshold,
        min_benign_turns=config.game.min_benign_turns,
        terminate_on_violation=config.game.terminate_on_violation,
    )

    defender = DefenderAgent(
        model_name=config.defender.model_name,
        device=device,
        max_length=config.defender.max_length,
        temperature=config.defender.temperature,
        use_policy=True,
    )

    if args.model_path and Path(args.model_path).exists():
        defender.load_checkpoint(args.model_path)
        print(f"Loaded checkpoint: {args.model_path}")

    adversary = AdversaryAgent(
        model_name=config.adversary.model_name,
        device=device,
        max_length=config.adversary.max_length,
        temperature=config.adversary.temperature,
        use_dataset=True,
    )

    if args.dataset == "mhj":
        eval_datasets = [JailbreakDataset("mhj", data_dir=config.data.data_dir)]
    elif args.dataset == "xguard_test":
        _, _, xguard_test = load_xguard_splits(
            seed=config.data.split_seed, data_dir=config.data.data_dir
        )
        eval_datasets = [xguard_test]
    elif args.dataset == "xguard_val":
        _, xguard_val, _ = load_xguard_splits(
            seed=config.data.split_seed, data_dir=config.data.data_dir
        )
        eval_datasets = [xguard_val]
    else:
        eval_datasets = load_datasets([args.dataset], data_dir=config.data.data_dir)

    # Evaluate
    metrics, _ = evaluate(
        defender=defender,
        adversary=adversary,
        env=env,
        datasets=eval_datasets,
        num_samples=args.num_samples,
    )

    EvaluationMetrics().print_metrics(metrics)

    # Save results
    results_path = Path(config.training.checkpoint_dir) / "evaluation_results.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(
            {"metrics": metrics, "num_samples": args.num_samples, "dataset": args.dataset},
            f,
            indent=2,
        )
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
