import argparse
import json
from pathlib import Path
import torch

from src.config import load_config
from src.game.environment import CensorshipGameEnvironment
from src.game.adversary import AdversaryAgent
from src.game.defender import DefenderAgent
from src.data.dataset import load_datasets, load_xguard_splits, JailbreakDataset
from src.training.equilibrium import EquilibriumTrainer


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-turn censorship game")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_path", type=str, help="Path to trained model checkpoint (.pt) or run directory")
    parser.add_argument("--adversary_path", type=str, default=None, help="Path to adversary checkpoint (.pt) or run directory")
    parser.add_argument("--dataset", type=str, default="mhj")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--refuse-threshold", type=float, default=None, help="Override refuse_threshold")
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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
        use_policy=True,
    )

    loaded_defender_from = None
    if args.model_path:
        p = Path(args.model_path)
        if p.is_dir():
            candidate = p / "defender_final.pt"
            if candidate.exists():
                defender.load_checkpoint(str(candidate))
                loaded_defender_from = str(candidate)
        elif p.exists():
            defender.load_checkpoint(str(p))
            loaded_defender_from = str(p)
    if loaded_defender_from:
        print(f"Loaded defender checkpoint: {loaded_defender_from}")

    adversary = AdversaryAgent(
        model_name=config.adversary.model_name,
        device=device,
        max_length=config.adversary.max_length,
        temperature=config.adversary.temperature,
        use_learned_policy=True,
    )
    loaded_adversary_from = None
    if args.adversary_path:
        p = Path(args.adversary_path)
        if p.is_dir():
            candidate = p / "adversary_final.pt"
            if candidate.exists():
                adversary.load_policy(str(candidate))
                loaded_adversary_from = str(candidate)
        elif p.exists():
            adversary.load_policy(str(p))
            loaded_adversary_from = str(p)
    elif args.model_path:
        p = Path(args.model_path)
        if p.is_dir():
            candidate = p / "adversary_final.pt"
            if candidate.exists():
                adversary.load_policy(str(candidate))
                loaded_adversary_from = str(candidate)
    if loaded_adversary_from:
        print(f"Loaded adversary checkpoint: {loaded_adversary_from}")

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

    refuse_threshold = (
        float(args.refuse_threshold)
        if args.refuse_threshold is not None
        else float(config.game.refuse_threshold)
    )

    trainer = EquilibriumTrainer(
        defender=defender,
        adversary=adversary,
        env=env,
        defender_lr=0.0,
        adversary_lr=0.0,
        gamma=config.training.gamma,
        update_mode=config.training.update_mode,
        batch_size=config.training.batch_size,
        logger=None,
        benign_prob=config.game.benign_prob,
        lambda_violation=config.game.lambda_violation,
        mu_refusal=config.game.mu_refusal,
        gamma_turn=config.game.gamma_turn,
        warmstart_steps=0,
    )
    trainer.refuse_threshold = refuse_threshold

    n_mal = int(args.num_samples) // 2
    n_ben = int(args.num_samples) - n_mal
    metrics = trainer.evaluate_slides_metrics(
        num_malicious=n_mal,
        num_benign=n_ben,
        datasets=eval_datasets,
    )

    print("\n" + "=" * 50)
    print("EVALUATION METRICS")
    print("=" * 50)
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print("=" * 50 + "\n")

    if loaded_defender_from:
        out_dir = Path(loaded_defender_from).parent
    else:
        out_dir = Path(config.training.checkpoint_dir)
    results_path = out_dir / "evaluation_results.json"
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
