import argparse
import json
from pathlib import Path
from datetime import datetime
import torch

from src.config import load_config
from src.game.environment import CensorshipGameEnvironment
from src.game.adversary import AdversaryAgent
from src.game.defender import DefenderAgent
from src.data.dataset import load_datasets, load_xguard_splits, JailbreakDataset
from src.training.equilibrium import EquilibriumTrainer
from src.training.logger import TrainingLogger
from src.reporting.generate import generate_run_report


def main():
    parser = argparse.ArgumentParser(description="Train multi-turn censorship game")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--mode", type=str, choices=["equilibrium"], default=None)
    parser.add_argument("--name", type=str, default=None, help="Experiment name for logging")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
    parser.add_argument("--num-episodes", type=int, default=None, help="Override num_episodes")
    parser.add_argument("--eval-interval", type=int, default=None, help="Override eval_interval")
    parser.add_argument("--eval-num-malicious", type=int, default=None, help="Num malicious eval episodes")
    parser.add_argument("--eval-num-benign", type=int, default=None, help="Num benign eval episodes")
    parser.add_argument("--defender-lr", type=float, default=None, help="Override defender_lr")
    parser.add_argument("--adversary-lr", type=float, default=None, help="Override adversary_lr")
    parser.add_argument("--refuse-threshold", type=float, default=None, help="Override game.refuse_threshold")
    args = parser.parse_args()

    config = load_config(args.config)
    training_mode = args.mode or config.training.mode
    if args.num_episodes is not None:
        config.training.num_episodes = int(args.num_episodes)
    if args.eval_interval is not None:
        config.training.eval_interval = int(args.eval_interval)
    if args.eval_num_malicious is not None:
        config.training.eval_num_malicious = int(args.eval_num_malicious)
    if args.eval_num_benign is not None:
        config.training.eval_num_benign = int(args.eval_num_benign)
    if args.defender_lr is not None:
        config.training.defender_lr = float(args.defender_lr)
    if args.adversary_lr is not None:
        config.training.adversary_lr = float(args.adversary_lr)
    if args.refuse_threshold is not None:
        config.game.refuse_threshold = float(args.refuse_threshold)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    experiment_name = args.name or f"{training_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger = TrainingLogger(
        log_dir=config.logging.log_dir,
        experiment_name=experiment_name,
        use_tensorboard=not args.no_tensorboard,
        log_interval=config.logging.log_interval,
    )

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

    env = CensorshipGameEnvironment(
        max_turns=config.game.max_turns,
        safety_threshold=config.game.safety_threshold,
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
        use_learned_policy=config.adversary.use_learned_policy,
    )

    x_train, x_val, x_test = load_xguard_splits(
        seed=config.data.split_seed, data_dir=config.data.data_dir
    )
    mhj_dataset = JailbreakDataset("mhj", data_dir=config.data.data_dir)

    train_datasets = [x_train]
    val_datasets = [x_val]
    test_datasets = [x_test, mhj_dataset]

    extra_train = [name for name in config.data.train_datasets if name.lower() != "xguard"]
    if extra_train:
        train_datasets.extend(load_datasets(extra_train, data_dir=config.data.data_dir))

    checkpoint_dir = Path(config.training.checkpoint_dir) / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = EquilibriumTrainer(
        defender=defender,
        adversary=adversary,
        env=env,
        defender_lr=config.training.defender_lr,
        adversary_lr=config.training.adversary_lr,
        gamma=config.training.gamma,
        update_mode=config.training.update_mode,
        batch_size=config.training.batch_size,
        logger=logger,
        benign_prob=config.game.benign_prob,
        lambda_violation=config.game.lambda_violation,
        mu_refusal=config.game.mu_refusal,
        gamma_turn=config.game.gamma_turn,
        warmstart_steps=config.training.warmstart_steps,
    )
    trainer.refuse_threshold = config.game.refuse_threshold

    training_stats = trainer.train(
        num_episodes=config.training.num_episodes,
        datasets=train_datasets,
        eval_interval=config.training.eval_interval,
        val_datasets=val_datasets if val_datasets else None,
        eval_num_malicious=config.training.eval_num_malicious,
        eval_num_benign=config.training.eval_num_benign,
    )

    trainer.save_checkpoints(
        str(checkpoint_dir / "defender_final.pt"),
        str(checkpoint_dir / "adversary_final.pt"),
    )

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

    log_metrics_path = logger.experiment_dir / "metrics.json"
    with open(log_metrics_path, "w") as f:
        json.dump(serializable_stats, f, indent=2)

    report_manifest = generate_run_report(
        experiment_dir=logger.experiment_dir,
        metrics=serializable_stats,
        hparams={
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
            "refuse_threshold": config.game.refuse_threshold,
            "defender_model": config.defender.model_name,
            "adversary_model": config.adversary.model_name,
        },
    )

    logger.close()

    print(f"\n✅ Training completed!")
    print(f"📁 Checkpoints saved to: {checkpoint_dir}")
    print(f"📊 Stats saved to: {stats_path}")
    print(f"🧾 Report saved to: {logger.experiment_dir}")
    if report_manifest.get("figures"):
        print(f"   Figures: {len(report_manifest['figures'])}")
    if report_manifest.get("tables"):
        print(f"   Tables: {len(report_manifest['tables'])}")


if __name__ == "__main__":
    main()
