from dataclasses import dataclass, field
from typing import List
import yaml


@dataclass
class ModelConfig:
    model_name: str = "gpt2"
    max_length: int = 512
    temperature: float = 0.7
    use_learned_policy: bool = True


@dataclass
class GameConfig:
    max_turns: int = 10
    safety_threshold: float = 0.5
    refuse_threshold: float = 0.5
    utility_weight: float = 0.3
    safety_weight: float = 0.7
    terminate_on_violation: bool = False
    benign_prob: float = 0.2
    lambda_violation: float = 5.0
    mu_refusal: float = 0.2
    gamma_turn: float = 0.01


@dataclass
class TrainingConfig:
    mode: str = "equilibrium"
    num_episodes: int = 10000
    batch_size: int = 32
    defender_lr: float = 1e-5
    adversary_lr: float = 1e-5
    gamma: float = 0.99
    eval_interval: int = 500
    eval_num_malicious: int = 20
    eval_num_benign: int = 20
    checkpoint_dir: str = "checkpoints"
    update_mode: str = "alternating"
    warmstart_steps: int = 200


@dataclass
class DataConfig:
    train_datasets: List[str] = field(default_factory=lambda: ["xguard"])
    val_datasets: List[str] = field(default_factory=lambda: ["mhj"])
    data_dir: str = "data"
    split_seed: int = 42


@dataclass
class LoggingConfig:
    use_wandb: bool = False
    log_dir: str = "logs"
    log_interval: int = 100


@dataclass
class Config:
    defender: ModelConfig = field(default_factory=ModelConfig)
    adversary: ModelConfig = field(
        default_factory=lambda: ModelConfig(max_length=256, temperature=0.9)
    )
    game: GameConfig = field(default_factory=GameConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    config = Config()

    if "model" in raw:
        if "defender" in raw["model"]:
            d = raw["model"]["defender"]
            config.defender = ModelConfig(
                model_name=d.get("model_name", "gpt2"),
                max_length=int(d.get("max_length", 512)),
                temperature=float(d.get("temperature", 0.7)),
                use_learned_policy=bool(d.get("use_learned_policy", True)),
            )
        if "adversary" in raw["model"]:
            a = raw["model"]["adversary"]
            config.adversary = ModelConfig(
                model_name=a.get("model_name", "gpt2"),
                max_length=int(a.get("max_length", 256)),
                temperature=float(a.get("temperature", 0.9)),
                use_learned_policy=bool(a.get("use_learned_policy", False)),
            )
    if "game" in raw:
        g = raw["game"]
        config.game = GameConfig(
            max_turns=int(g.get("max_turns", 10)),
            safety_threshold=float(g.get("safety_threshold", 0.5)),
            refuse_threshold=float(g.get("refuse_threshold", 0.5)),
            utility_weight=float(g.get("utility_weight", 0.3)),
            safety_weight=float(g.get("safety_weight", 0.7)),
            terminate_on_violation=bool(g.get("terminate_on_violation", False)),
            benign_prob=float(g.get("benign_prob", 0.2)),
            lambda_violation=float(g.get("lambda_violation", 5.0)),
            mu_refusal=float(g.get("mu_refusal", 0.2)),
            gamma_turn=float(g.get("gamma_turn", 0.01)),
        )

    if "training" in raw:
        t = raw["training"]
        config.training = TrainingConfig(
            mode=str(t.get("mode", "equilibrium")),
            num_episodes=int(t.get("num_episodes", 10000)),
            batch_size=int(t.get("batch_size", 32)),
            defender_lr=float(t.get("defender_lr", 1e-5)),
            adversary_lr=float(t.get("adversary_lr", 1e-5)),
            gamma=float(t.get("gamma", 0.99)),
            eval_interval=int(t.get("eval_interval", 500)),
            eval_num_malicious=int(t.get("eval_num_malicious", 20)),
            eval_num_benign=int(t.get("eval_num_benign", 20)),
            checkpoint_dir=str(t.get("checkpoint_dir", "checkpoints")),
            update_mode=str(t.get("update_mode", "alternating")),
            warmstart_steps=int(t.get("warmstart_steps", 200)),
        )

    if "data" in raw:
        d = raw["data"]
        config.data = DataConfig(
            train_datasets=d.get("train_datasets", ["xguard"]),
            val_datasets=d.get("val_datasets", ["mhj"]),
            data_dir=str(d.get("data_dir", "data")),
            split_seed=int(d.get("split_seed", 42)),
        )

    if "logging" in raw:
        l = raw["logging"]
        config.logging = LoggingConfig(
            use_wandb=bool(l.get("use_wandb", False)),
            log_dir=str(l.get("log_dir", "logs")),
            log_interval=int(l.get("log_interval", 100)),
        )

    return config
