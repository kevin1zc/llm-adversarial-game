"""Self-play training loop for the censorship game."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from ..game.environment import CensorshipGameEnvironment, ActionType
from ..game.adversary import AdversaryAgent
from ..game.defender import DefenderAgent
from ..game.rewards import RewardFunction
from ..data.dataset import JailbreakDataset
from .logger import TrainingLogger


@dataclass
class EpisodeStep:
    """Single step in an episode (lightweight, no full state)."""

    history: str
    user_message: str
    action: ActionType
    reward: float


class SelfPlayTrainer:
    """Self-play trainer for the censorship game."""

    def __init__(
        self,
        defender: DefenderAgent,
        adversary: AdversaryAgent,
        reward_fn: RewardFunction,
        env: CensorshipGameEnvironment,
        learning_rate: float = 1e-5,
        gamma: float = 0.99,
        logger: Optional[TrainingLogger] = None,
    ):
        self.defender = defender
        self.adversary = adversary
        self.reward_fn = reward_fn
        self.env = env
        self.gamma = gamma
        self.logger = logger

        self.optimizer = (
            optim.Adam(defender.policy.parameters(), lr=learning_rate)
            if defender.use_policy and defender.policy
            else None
        )

    @torch.no_grad()
    def train_episode(self, dataset_example: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train on a single episode."""
        state = self.env.reset()
        steps: List[EpisodeStep] = []

        target_harmful = dataset_example.target_harmful if dataset_example else None
        turn_strategy = dataset_example.attack_type if dataset_example else "gradual"
        conv = dataset_example.turns if dataset_example else []

        while not state.is_terminal:
            history = state.get_history_text() if state.conversation_history else ""

            # Adversary generates message
            if state.turn_number < len(conv):
                turn_info = conv[state.turn_number]
                user_message = turn_info.user
                is_harmful_turn = bool(turn_info.is_harmful)
            else:
                user_message = self.adversary.generate_message(
                    state, target_harmful_content=target_harmful, turn_strategy=turn_strategy
                )
                is_harmful_turn = False

            # Defender action (policy does not use safety_score argument)
            action_type, response = self.defender.decide_action(
                state, user_message, 1.0, self.env.safety_threshold, explore=True
            )

            # Step environment with ground-truth harmfulness
            next_state, info, is_terminal = self.env.step(
                user_message, response, action_type, is_harmful_turn
            )

            # Compute reward
            reward_dict = self.reward_fn.compute_reward(
                state, action_type, is_harmful_turn, info, is_terminal
            )

            steps.append(
                EpisodeStep(
                    history=history,
                    user_message=user_message,
                    action=action_type,
                    reward=reward_dict["total"],
                )
            )
            state = next_state

        # Update policy
        loss = 0.0
        if self.optimizer and steps:
            loss = self._update_policy(steps)

        return {
            "total_reward": sum(s.reward for s in steps),
            "avg_reward": sum(s.reward for s in steps) / len(steps) if steps else 0,
            "violation": state.has_violation,
            "violation_turn": state.violation_turn,
            "episode_length": state.turn_number,
            "num_refusals": sum(1 for s in steps if s.action == ActionType.REFUSE),
            "num_responses": sum(1 for s in steps if s.action == ActionType.RESPOND),
            "loss": loss,
        }

    def _update_policy(self, steps: List[EpisodeStep]) -> float:
        """Update defender policy using REINFORCE with batched computation."""
        if not self.defender.policy or not self.defender.tokenizer:
            return 0.0

        # Compute discounted returns
        returns = []
        G = 0.0
        for step in reversed(steps):
            G = step.reward + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.defender.device)
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Prepare batch
        prompts = [f"{s.history}\nUser: {s.user_message}\nAssistant:" for s in steps]
        action_indices = [0 if s.action == ActionType.RESPOND else 1 for s in steps]

        inputs = self.defender.tokenizer(
            prompts, return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(self.defender.device)

        # Forward pass and compute loss
        self.optimizer.zero_grad()
        decision_logits, _ = self.defender.policy(**inputs)
        log_probs = F.log_softmax(decision_logits, dim=-1)

        action_indices_t = torch.tensor(action_indices, device=self.defender.device)
        selected_log_probs = log_probs.gather(1, action_indices_t.unsqueeze(1)).squeeze(1)

        # Add entropy bonus to encourage exploration
        entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1).mean()
        entropy_coef = 0.01

        pg_loss = -(selected_log_probs * returns_t).mean()
        loss = pg_loss - entropy_coef * entropy  # Subtract entropy to maximize it

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.defender.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        return pg_loss.item()  # Return PG loss for logging

    def train(
        self,
        num_episodes: int,
        datasets: List[JailbreakDataset],
        eval_interval: int = 500,
    ) -> Dict[str, List[float]]:
        """Train for multiple episodes."""
        all_examples = []
        for dataset in datasets:
            all_examples.extend(dataset.get_examples())

        stats: Dict[str, List[float]] = {
            "episode_reward": [],
            "episode_length": [],
            "violation_rate": [],
            "refusal_rate": [],
            "loss": [],
        }

        print(f"\n{'='*60}")
        print(f"🎮 Starting Self-Play Training")
        print(f"{'='*60}")
        print(f"  Episodes: {num_episodes}")
        print(f"  Gamma: {self.gamma}")
        print(f"  Eval interval: {eval_interval}")
        print(f"{'='*60}\n")

        pbar = tqdm(range(num_episodes), desc="Training", unit="ep")
        for episode in pbar:
            example = random.choice(all_examples) if all_examples else None
            ep_stats = self.train_episode(example)

            violation = 1.0 if ep_stats["violation"] else 0.0
            refusal_rate = (
                ep_stats["num_refusals"] / ep_stats["episode_length"]
                if ep_stats["episode_length"] > 0
                else 0.0
            )

            stats["episode_reward"].append(ep_stats["total_reward"])
            stats["episode_length"].append(float(ep_stats["episode_length"]))
            stats["violation_rate"].append(violation)
            stats["refusal_rate"].append(refusal_rate)
            stats["loss"].append(ep_stats["loss"])

            # Log to TensorBoard
            if self.logger:
                self.logger.log_episode(
                    episode,
                    {
                        "reward": ep_stats["total_reward"],
                        "violation": violation,
                        "episode_length": float(ep_stats["episode_length"]),
                        "refusal_rate": refusal_rate,
                        "loss": ep_stats["loss"],
                    },
                    prefix="self_play",
                )

            # Update progress bar
            if len(stats["episode_reward"]) >= 10:
                avg_reward = np.mean(stats["episode_reward"][-100:])
                avg_viol = np.mean(stats["violation_rate"][-100:])
                pbar.set_postfix(
                    {
                        "reward": f"{avg_reward:.2f}",
                        "viol": f"{avg_viol:.2%}",
                    }
                )

            # Periodic evaluation
            if (episode + 1) % eval_interval == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                eval_stats = {
                    "reward_avg": float(np.mean(stats["episode_reward"][-eval_interval:])),
                    "violation_rate_avg": float(np.mean(stats["violation_rate"][-eval_interval:])),
                    "refusal_rate_avg": float(np.mean(stats["refusal_rate"][-eval_interval:])),
                    "loss_avg": float(np.mean(stats["loss"][-eval_interval:])),
                }

                if self.logger:
                    self.logger.log_evaluation(episode + 1, eval_stats)
                else:
                    print(f"\n{'='*50}")
                    print(f"📈 Episode {episode + 1}")
                    print(f"{'='*50}")
                    print(f"  Avg Reward:     {eval_stats['reward_avg']:8.3f}")
                    print(f"  Violation Rate: {eval_stats['violation_rate_avg']:8.3%}")
                    print(f"  Refusal Rate:   {eval_stats['refusal_rate_avg']:8.3%}")
                    print(f"  Avg Loss:       {eval_stats['loss_avg']:8.4f}")
                    print(f"{'='*50}\n")

        return stats
