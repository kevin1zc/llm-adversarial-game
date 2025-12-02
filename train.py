"""Main training script for multi-turn censorship game."""

import argparse
import json
from pathlib import Path
from datetime import datetime
import torch

from src.config import load_config
from src.game.environment import CensorshipGameEnvironment
from src.game.adversary import AdversaryAgent
from src.game.defender import DefenderAgent
from src.game.rewards import RewardFunction
from src.data.dataset import load_datasets, load_xguard_splits, JailbreakDataset
from src.training.self_play import SelfPlayTrainer
from src.training.equilibrium import EquilibriumTrainer
from src.training.logger import TrainingLogger


def main():
    parser = argparse.ArgumentParser(description="Train multi-turn censorship game")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--mode", type=str, choices=["self_play", "equilibrium"], default=None)
    parser.add_argument("--name", type=str, default=None, help="Experiment name for logging")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
    args = parser.parse_args()

    # Load typed configuration
    config = load_config(args.config)
    training_mode = args.mode or config.training.mode

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create experiment name
    experiment_name = args.name or f"{training_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize logger
    logger = TrainingLogger(
        log_dir=config.logging.log_dir,
        experiment_name=experiment_name,
        use_tensorboard=not args.no_tensorboard,
        log_interval=config.logging.log_interval,
    )

    # Log hyperparameters
    logger.log_hyperparameters(
        {
            "mode": training_mode,
            "device": device,
            "num_episodes": config.training.num_episodes,
            "batch_size": config.training.batch_size,
            "defender_lr": config.training.defender_lr,
            "adversary_lr": config.training.adversary_lr,
            "gamma": config.training.gamma,
            "update_mode": config.training.update_mode,
            "max_turns": config.game.max_turns,
            "safety_threshold": config.game.safety_threshold,
            "defender_model": config.defender.model_name,
            "adversary_model": config.adversary.model_name,
        }
    )

    print(f"\n🖥️  Device: {device}")
    print(f"🎯 Mode: {training_mode}")
    print(f"📁 Experiment: {experiment_name}\n")

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
        use_policy=config.defender.use_learned_policy,
    )

    adversary = AdversaryAgent(
        model_name=config.adversary.model_name,
        device=device,
        max_length=config.adversary.max_length,
        temperature=config.adversary.temperature,
        use_dataset=True,
        use_learned_policy=config.adversary.use_learned_policy,
    )

    reward_fn = RewardFunction(
        utility_weight=config.game.utility_weight,
        safety_weight=config.game.safety_weight,
    )

    # Load datasets with XGuard splits and MHJ for testing
    x_train, x_val, x_test = load_xguard_splits(
        seed=config.data.split_seed, data_dir=config.data.data_dir
    )
    mhj_dataset = JailbreakDataset("mhj", data_dir=config.data.data_dir)

    train_datasets = [x_train]
    val_datasets = [x_val]
    test_datasets = [x_test, mhj_dataset]

    # Optional extra datasets from config (excluding the default entries we already handled)
    extra_train = [name for name in config.data.train_datasets if name.lower() != "xguard"]
    if extra_train:
        train_datasets.extend(load_datasets(extra_train, data_dir=config.data.data_dir))

    # Create checkpoint directory
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # Initialize trainer
    if training_mode == "equilibrium":
        trainer = EquilibriumTrainer(
            defender=defender,
            adversary=adversary,
            reward_fn=reward_fn,
            env=env,
            defender_lr=config.training.defender_lr,
            adversary_lr=config.training.adversary_lr,
            gamma=config.training.gamma,
            update_mode=config.training.update_mode,
            batch_size=config.training.batch_size,
            logger=logger,
        )
    else:
        trainer = SelfPlayTrainer(
            defender=defender,
            adversary=adversary,
            reward_fn=reward_fn,
            env=env,
            learning_rate=config.training.defender_lr,
            gamma=config.training.gamma,
            logger=logger,
        )

    # Train
    training_stats = trainer.train(
        num_episodes=config.training.num_episodes,
        datasets=train_datasets,
        eval_interval=config.training.eval_interval,
        val_datasets=val_datasets if val_datasets else None,
    )

    # Save checkpoints
    if training_mode == "equilibrium":
        trainer.save_checkpoints(
            str(checkpoint_dir / "defender_final.pt"),
            str(checkpoint_dir / "adversary_final.pt"),
        )
    elif defender.use_policy and defender.policy is not None:
        torch.save(defender.policy.state_dict(), checkpoint_dir / "defender_final.pt")

    # Save training stats
    stats_path = checkpoint_dir / "training_stats.json"
    serializable_stats = {
        k: (
            [float(v) if isinstance(v, (int, float)) else str(v) for v in vals]
            if isinstance(vals, list)
            else vals
        )
        for k, vals in training_stats.items()
    }
    with open(stats_path, "w") as f:
        json.dump(serializable_stats, f, indent=2)

    # Close logger
    logger.close()

    print(f"\n✅ Training completed!")
    print(f"📁 Checkpoints saved to: {checkpoint_dir}")
    print(f"📊 Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
