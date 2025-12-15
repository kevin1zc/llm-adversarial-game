from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.game.environment import ActionType


@dataclass
class Turn:
    user: str
    assistant: Optional[str] = None
    is_harmful: bool = False
    turn_index: int = 0


@dataclass
class ConversationExample:
    turns: List[Turn] = field(default_factory=list)
    attack_type: str = "multi_turn"
    target_harmful: str = ""


@dataclass
class ConversationDataset:
    name: str
    examples: List[ConversationExample]

    def get_examples(self) -> List[ConversationExample]:
        return self.examples
