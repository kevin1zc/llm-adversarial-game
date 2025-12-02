"""Equilibrium learning for the two-player censorship game."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from ..game.environment import CensorshipGameEnvironment, GameState, ActionType
from ..game.adversary import AdversaryAgent
from ..game.defender import DefenderAgent
from ..game.rewards import RewardFunction
from ..data.dataset import JailbreakDataset
from .logger import TrainingLogger


@dataclass
class Experience:
    """Lightweight experience for policy updates."""

    history: str
    user_message: str
    reward: float
    turn_number: int
    action: Optional[ActionType] = None
    strategy_idx: Optional[int] = None


@dataclass
class EpisodeResult:
    """Result of a single episode."""

    defender_reward: float
    adversary_reward: float
    violation: bool
    violation_turn: Optional[int]
    episode_length: int
    num_refusals: int
    num_responses: int
    defender_actions: List[ActionType] = field(default_factory=list)
    adversary_messages: List[str] = field(default_factory=list)
    defender_experiences: List[Experience] = field(default_factory=list)
    adversary_experiences: List[Experience] = field(default_factory=list)


class EquilibriumTrainer:
    """Two-player equilibrium learning trainer."""

    def __init__(
        self,
        defender: DefenderAgent,
        adversary: AdversaryAgent,
        reward_fn: RewardFunction,
        env: CensorshipGameEnvironment,
        defender_lr: float = 1e-5,
        adversary_lr: float = 1e-5,
        gamma: float = 0.99,
        update_mode: str = "alternating",
        batch_size: int = 32,
        logger: Optional[TrainingLogger] = None,
    ):
        self.defender = defender
        self.adversary = adversary
        self.reward_fn = reward_fn
        self.env = env
        self.gamma = gamma
        self.update_mode = update_mode
        self.batch_size = batch_size
        self.logger = logger

        # Pending on-policy experience storage
        self.pending_defender_experiences: List[Experience] = []
        self.pending_adversary_experiences: List[Experience] = []

        # Setup optimizers
        self.defender_optimizer = (
            optim.Adam(defender.policy.parameters(), lr=defender_lr)
            if defender.use_policy and defender.policy
            else None
        )
        self.adversary_optimizer = (
            optim.Adam(adversary.policy.parameters(), lr=adversary_lr)
            if adversary.use_learned_policy and adversary.policy
            else None
        )

    @torch.no_grad()
    def play_episode(
        self,
        dataset_example: Optional[Dict[str, Any]] = None,
        collect_experience: bool = True,
    ) -> EpisodeResult:
        """Play a single episode of the game."""
        state = self.env.reset()
        target_harmful = dataset_example.target_harmful if dataset_example else None

        defender_rewards, adversary_rewards = [], []
        defender_actions, adversary_messages = [], []
        defender_experiences: List[Experience] = []
        adversary_experiences: List[Experience] = []

        while not state.is_terminal:
            history_text = state.get_history_text() if state.conversation_history else ""

            # Adversary selects strategy
            if self.adversary.use_learned_policy:
                strategy, _ = self.adversary.select_strategy(state)
                strategy_idx = self.adversary.STRATEGIES.index(strategy)
            else:
                strategy, strategy_idx = "gradual", 0

            # Generate adversary message
            conv = dataset_example.turns if dataset_example else []
            is_harmful_turn = False
            if state.turn_number < len(conv):
                turn_info = conv[state.turn_number]
                user_message = turn_info.user
                is_harmful_turn = bool(turn_info.is_harmful)
            else:
                user_message = self.adversary.generate_message(
                    state, target_harmful_content=target_harmful, turn_strategy=strategy
                )
                # For generated messages we don't have ground-truth harmfulness; assume benign.
                is_harmful_turn = False
            adversary_messages.append(user_message)

            # Defender action
            # NOTE: learned defender policy does not actually use safety_score argument.
            action_type, response = self.defender.decide_action(
                state, user_message, 1.0, self.env.safety_threshold, explore=True
            )
            defender_actions.append(action_type)

            # Step environment
            next_state, info, is_terminal = self.env.step(
                user_message, response, action_type, is_harmful_turn
            )

            # Compute rewards based on ground-truth harmfulness of the user turn
            reward_dict = self.reward_fn.compute_reward(
                state, action_type, is_harmful_turn, info, is_terminal
            )
            defender_reward = reward_dict["total"]
            adversary_reward = self._compute_adversary_reward(
                state, action_type, is_harmful_turn, info
            )
            defender_rewards.append(defender_reward)
            adversary_rewards.append(adversary_reward)

            # Collect experience
            if collect_experience:
                defender_experiences.append(
                    Experience(
                        history=history_text,
                        user_message=user_message,
                        reward=defender_reward,
                        turn_number=state.turn_number,
                        action=action_type,
                    )
                )
                adversary_experiences.append(
                    Experience(
                        history=history_text,
                        user_message=user_message,
                        reward=adversary_reward,
                        turn_number=state.turn_number,
                        strategy_idx=strategy_idx,
                    )
                )

            state = next_state

        return EpisodeResult(
            defender_reward=sum(defender_rewards),
            adversary_reward=sum(adversary_rewards),
            violation=state.has_violation,
            violation_turn=state.violation_turn,
            episode_length=state.turn_number,
            num_refusals=sum(1 for a in defender_actions if a == ActionType.REFUSE),
            num_responses=sum(1 for a in defender_actions if a == ActionType.RESPOND),
            defender_actions=defender_actions,
            adversary_messages=adversary_messages,
            defender_experiences=defender_experiences,
            adversary_experiences=adversary_experiences,
        )

    def _compute_adversary_reward(
        self, state: GameState, action: ActionType, is_harmful_turn: bool, info: Dict[str, Any]
    ) -> float:
        """Compute reward for the adversary.

        The adversary wants to:
        1. Get the defender to respond to harmful user messages (violations)
        2. Avoid being trivially refused on obviously harmful prompts
        """
        reward = 0.0

        # Reward for causing a violation (defender responded to harmful content)
        if info.get("violation", False):
            reward += 3.0

        # Small shaping based on action on harmful turns
        if is_harmful_turn:
            if action == ActionType.RESPOND:
                reward += 1.0  # harmful success
            elif action == ActionType.REFUSE:
                reward -= 0.5  # attack failed
        else:
            # On safe turns, encourage engagement
            if action == ActionType.RESPOND:
                reward += 0.1

        return reward

    def update_defender(self) -> float:
        """Update defender policy."""
        if not self.defender_optimizer or not self.pending_defender_experiences:
            return 0.0

        batch_n = (
            self.batch_size
            if len(self.pending_defender_experiences) >= self.batch_size
            else len(self.pending_defender_experiences)
        )
        experiences = self.pending_defender_experiences[:batch_n]
        self.pending_defender_experiences = self.pending_defender_experiences[batch_n:]

        return self._update_policy(
            self.defender,
            self.defender_optimizer,
            experiences,
            is_defender=True,
        )

    def update_adversary(self) -> float:
        """Update adversary policy."""
        if not self.adversary_optimizer or not self.pending_adversary_experiences:
            return 0.0

        batch_n = (
            self.batch_size
            if len(self.pending_adversary_experiences) >= self.batch_size
            else len(self.pending_adversary_experiences)
        )
        experiences = self.pending_adversary_experiences[:batch_n]
        self.pending_adversary_experiences = self.pending_adversary_experiences[batch_n:]

        return self._update_policy(
            self.adversary,
            self.adversary_optimizer,
            experiences,
            is_defender=False,
        )

    def _update_policy(
        self, agent, optimizer: optim.Optimizer, experiences: List[Experience], is_defender: bool
    ) -> float:
        """Perform batched policy gradient update with reward normalization."""
        if not agent.tokenizer or not agent.policy:
            return 0.0

        optimizer.zero_grad()

        prompts, rewards, action_indices = [], [], []
        for exp in experiences:
            history = exp.history or "Start of conversation"
            if is_defender:
                prompts.append(f"{history}\nUser: {exp.user_message}\nAssistant:")
                action_indices.append(0 if exp.action == ActionType.RESPOND else 1)
            else:
                prompts.append(history)
                action_indices.append(exp.strategy_idx or 0)
            rewards.append(exp.reward)

        # Normalize rewards (baseline subtraction + scaling)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=agent.device)
        if len(rewards_t) > 1:
            reward_mean = rewards_t.mean()
            reward_std = rewards_t.std() + 1e-8
            rewards_t = (rewards_t - reward_mean) / reward_std

        max_len = 512 if is_defender else agent.max_length
        inputs = agent.tokenizer(
            prompts, return_tensors="pt", truncation=True, padding=True, max_length=max_len
        ).to(agent.device)

        if is_defender:
            decision_logits, _ = agent.policy(**inputs)
            log_probs = F.log_softmax(decision_logits, dim=-1)
        else:
            _, _, strategy_logits = agent.policy(**inputs)
            log_probs = F.log_softmax(strategy_logits, dim=-1)

        action_indices_t = torch.tensor(action_indices, device=agent.device)
        selected_log_probs = log_probs.gather(1, action_indices_t.unsqueeze(1)).squeeze(1)

        # Policy gradient loss: -E[log_prob * advantage]
        # Add entropy bonus to encourage exploration
        entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1).mean()
        entropy_coef = 0.01

        pg_loss = -(selected_log_probs * rewards_t).mean()
        loss = pg_loss - entropy_coef * entropy  # Subtract entropy to maximize it

        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        return pg_loss.item()  # Return PG loss for logging (not including entropy)

    @torch.no_grad()
    def compute_exploitability(
        self, num_episodes: int = 100, datasets: Optional[List[JailbreakDataset]] = None
    ) -> float:
        """Compute approximate exploitability of current policies.

        Exploitability measures how far from equilibrium the policies are.
        A lower value indicates policies are closer to Nash equilibrium.

        We compute:
        1. Average game value (defender - adversary rewards)
        2. Variance in outcomes (indicates policy instability)
        3. Win rate differential
        """
        all_examples = []
        if datasets:
            for dataset in datasets:
                all_examples.extend(dataset.get_examples())

        defender_rewards, adversary_rewards = [], []
        violations, refusals = [], []

        for _ in range(num_episodes):
            example = random.choice(all_examples) if all_examples else None
            result = self.play_episode(example, collect_experience=False)
            defender_rewards.append(result.defender_reward)
            adversary_rewards.append(result.adversary_reward)
            violations.append(1.0 if result.violation else 0.0)
            refusals.append(result.num_refusals / max(result.episode_length, 1))

        # Compute exploitability components
        def_mean = np.mean(defender_rewards)
        adv_mean = np.mean(adversary_rewards)
        def_std = np.std(defender_rewards)
        adv_std = np.std(adversary_rewards)
        violation_rate = np.mean(violations)
        refusal_rate = np.mean(refusals)

        # Exploitability: combination of variance and imbalance
        # High variance = unstable policies
        # High absolute game value = one side is dominating
        variance_term = def_std + adv_std
        imbalance_term = abs(def_mean + adv_mean)  # Zero-sum: should sum to ~0

        # Penalty for extreme strategies (all-refuse or all-respond)
        strategy_penalty = 0.0
        if refusal_rate > 0.95 or refusal_rate < 0.05:
            strategy_penalty = 1.0

        return float(variance_term + 0.5 * imbalance_term + strategy_penalty)

    def train(
        self,
        num_episodes: int,
        datasets: List[JailbreakDataset],
        eval_interval: int = 500,
        update_frequency: int = 1,
        val_datasets: Optional[List[JailbreakDataset]] = None,
    ) -> Dict[str, List[float]]:
        """Train both agents using equilibrium learning.

        Args:
            num_episodes: Number of training episodes
            datasets: Training datasets (for episode sampling)
            eval_interval: How often to run evaluation
            update_frequency: How often to update policies
            val_datasets: Optional validation datasets (for evaluation)
        """
        all_examples = []
        for dataset in datasets:
            all_examples.extend(dataset.get_examples())

        # Use validation datasets for evaluation if provided, otherwise use training datasets
        eval_datasets = val_datasets if val_datasets else datasets

        stats: Dict[str, List[float]] = {
            "defender_reward": [],
            "adversary_reward": [],
            "violation_rate": [],
            "episode_length": [],
            "exploitability": [],
            "defender_loss": [],
            "adversary_loss": [],
            "refusal_rate": [],
        }

        print(f"\n{'='*60}")
        print(f"🎮 Starting Equilibrium Training")
        print(f"{'='*60}")
        print(f"  Episodes: {num_episodes}")
        print(f"  Update mode: {self.update_mode}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Eval interval: {eval_interval}")
        print(f"{'='*60}\n")

        pbar = tqdm(range(num_episodes), desc="Training", unit="ep")
        for episode in pbar:
            example = random.choice(all_examples) if all_examples else None
            result = self.play_episode(example, collect_experience=True)

            # Accumulate on-policy experiences for updates
            if result.defender_experiences:
                self.pending_defender_experiences.extend(result.defender_experiences)
            if result.adversary_experiences:
                self.pending_adversary_experiences.extend(result.adversary_experiences)

            # Collect stats
            violation = 1.0 if result.violation else 0.0
            refusal_rate = (
                result.num_refusals / result.episode_length if result.episode_length > 0 else 0.0
            )

            stats["defender_reward"].append(result.defender_reward)
            stats["adversary_reward"].append(result.adversary_reward)
            stats["violation_rate"].append(violation)
            stats["episode_length"].append(float(result.episode_length))
            stats["refusal_rate"].append(refusal_rate)
            # Update policies
            def_loss, adv_loss = 0.0, 0.0
            if (episode + 1) % update_frequency == 0:
                def_loss, adv_loss = self._do_updates(episode)
            stats["defender_loss"].append(def_loss)
            stats["adversary_loss"].append(adv_loss)

            # Log to TensorBoard
            if self.logger:
                self.logger.log_episode(
                    episode,
                    {
                        "defender_reward": result.defender_reward,
                        "adversary_reward": result.adversary_reward,
                        "violation": violation,
                        "episode_length": float(result.episode_length),
                        "refusal_rate": refusal_rate,
                        "defender_loss": def_loss,
                        "adversary_loss": adv_loss,
                    },
                    prefix="equilibrium",
                )

            # Update progress bar
            if len(stats["defender_reward"]) >= 10:
                avg_def = np.mean(stats["defender_reward"][-100:])
                avg_viol = np.mean(stats["violation_rate"][-100:])
                pbar.set_postfix({"def_r": f"{avg_def:.2f}", "viol": f"{avg_viol:.2%}"})

            # Periodic evaluation
            if (episode + 1) % eval_interval == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Use validation datasets for evaluation (if provided, otherwise use training datasets)
                exploitability = self.compute_exploitability(100, eval_datasets)
                stats["exploitability"].append(exploitability)

                eval_stats = {
                    "defender_reward_avg": float(
                        np.mean(stats["defender_reward"][-eval_interval:])
                    ),
                    "adversary_reward_avg": float(
                        np.mean(stats["adversary_reward"][-eval_interval:])
                    ),
                    "violation_rate_avg": float(np.mean(stats["violation_rate"][-eval_interval:])),
                    "refusal_rate_avg": float(np.mean(stats["refusal_rate"][-eval_interval:])),
                    "exploitability": exploitability,
                }

                if self.logger:
                    self.logger.log_evaluation(episode + 1, eval_stats)
                else:
                    self._log_progress(episode + 1, stats, exploitability)

        return stats

    def _do_updates(self, episode: int) -> tuple[float, float]:
        """Perform policy updates based on update mode."""
        if self.update_mode == "alternating":
            if episode % 2 == 0:
                return self.update_defender(), 0.0
            return 0.0, self.update_adversary()
        return self.update_defender(), self.update_adversary()

    def _log_progress(self, episode: int, stats: Dict[str, List[float]], exploitability: float):
        """Log training progress to console."""
        print(f"\n{'='*50}")
        print(f"📈 Episode {episode}")
        print(f"{'='*50}")
        print(f"  Defender Reward:  {np.mean(stats['defender_reward'][-100:]):8.3f}")
        print(f"  Adversary Reward: {np.mean(stats['adversary_reward'][-100:]):8.3f}")
        print(f"  Violation Rate:   {np.mean(stats['violation_rate'][-100:]):8.3%}")
        print(f"  Refusal Rate:     {np.mean(stats['refusal_rate'][-100:]):8.3%}")
        print(f"  Exploitability:   {exploitability:8.3f}")
        print(f"{'='*50}\n")

    def save_checkpoints(self, defender_path: str, adversary_path: str):
        """Save both agent checkpoints."""
        if self.defender.use_policy and self.defender.policy:
            torch.save(self.defender.policy.state_dict(), defender_path)
        if self.adversary.use_learned_policy and self.adversary.policy:
            self.adversary.save_policy(adversary_path)

    def load_checkpoints(self, defender_path: str, adversary_path: str):
        """Load both agent checkpoints."""
        if self.defender.use_policy and self.defender.policy:
            self.defender.load_checkpoint(defender_path)
        if self.adversary.use_learned_policy and self.adversary.policy:
            self.adversary.load_policy(adversary_path)
