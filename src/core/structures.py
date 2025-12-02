"""Canonical data structures shared across the project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.game.environment import ActionType


@dataclass
class Turn:
    """Single user/assistant exchange."""

    user: str
    assistant: Optional[str] = None
    is_harmful: bool = False
    turn_index: int = 0


@dataclass
class ConversationExample:
    """Normalized jailbreak conversation pulled from any dataset."""

    turns: List[Turn] = field(default_factory=list)
    attack_type: str = "multi_turn"
    target_harmful: str = ""


@dataclass
class ConversationDataset:
    """Lightweight dataset wrapper providing the trainer interface."""

    name: str
    examples: List[ConversationExample]

    def get_examples(self) -> List[ConversationExample]:
        return self.examples


@dataclass
class Experience:
    """Trajectory sample used for policy gradient updates."""

    history: str
    user_message: str
    reward: float
    turn_number: int
    action: Optional["ActionType"] = None
    strategy_idx: Optional[int] = None


@dataclass
class EpisodeStats:
    """Aggregate statistics emitted by a trainer."""

    defender_reward: float = 0.0
    adversary_reward: float = 0.0
    violation: bool = False
    violation_turn: Optional[int] = None
    episode_length: int = 0
    refusals: int = 0
    responses: int = 0
