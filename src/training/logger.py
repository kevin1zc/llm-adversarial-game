"""Training logger with TensorBoard and console output."""

from typing import Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None


class TrainingLogger:
    """Logger for training metrics with TensorBoard support."""

    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True,
        log_interval: int = 100,
    ):
        self.log_interval = log_interval
        self.use_tensorboard = use_tensorboard and HAS_TENSORBOARD

        # Create experiment directory
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path(log_dir) / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer: Optional[SummaryWriter] = None
        if self.use_tensorboard:
            self.writer = SummaryWriter(str(self.experiment_dir))
            print(f"📊 TensorBoard logging to: {self.experiment_dir}")
            print(f"   Run: tensorboard --logdir {log_dir}")

        # Running stats for smoothed logging
        self._running_stats: Dict[str, list] = {}

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, values: Dict[str, float], step: int):
        """Log multiple scalars under the same tag."""
        if self.writer:
            self.writer.add_scalars(main_tag, values, step)

    def log_episode(
        self,
        episode: int,
        stats: Dict[str, float],
        prefix: str = "train",
    ):
        """Log episode statistics."""
        # Add to running stats
        for key, value in stats.items():
            if key not in self._running_stats:
                self._running_stats[key] = []
            self._running_stats[key].append(value)

        # Log to TensorBoard
        for key, value in stats.items():
            self.log_scalar(f"{prefix}/{key}", value, episode)

        # Console output at intervals
        if (episode + 1) % self.log_interval == 0:
            self._print_stats(episode + 1, prefix)

    def log_evaluation(
        self,
        episode: int,
        stats: Dict[str, float],
        prefix: str = "eval",
    ):
        """Log evaluation statistics."""
        for key, value in stats.items():
            self.log_scalar(f"{prefix}/{key}", value, episode)

        # Always print evaluation stats
        print(f"\n{'='*50}")
        print(f"📈 Evaluation at Episode {episode}")
        print(f"{'='*50}")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")
        print(f"{'='*50}\n")

    def _print_stats(self, episode: int, prefix: str):
        """Print running statistics to console."""
        window = min(self.log_interval, 100)

        print(f"\n[{prefix.upper()}] Episode {episode}")
        print("-" * 40)

        for key, values in self._running_stats.items():
            recent = values[-window:]
            avg = np.mean(recent)
            std = np.std(recent) if len(recent) > 1 else 0
            print(f"  {key:20s}: {avg:8.4f} ± {std:.4f}")

        print("-" * 40)

    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters."""
        if self.writer:
            # Convert to string for non-scalar types
            hparams_str = {
                k: str(v) if not isinstance(v, (int, float, bool)) else v
                for k, v in hparams.items()
            }
            self.writer.add_hparams(hparams_str, {})

        print("\n📋 Hyperparameters:")
        for key, value in hparams.items():
            print(f"  {key}: {value}")
        print()

    def close(self):
        """Close the logger."""
        if self.writer:
            self.writer.close()
            print(f"\n✅ Training logs saved to: {self.experiment_dir}")
