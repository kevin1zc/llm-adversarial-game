from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    RESPOND = "respond"
    REFUSE = "refuse"
    TERMINATE = "terminate"


@dataclass
class ConversationTurn:
    user_message: str
    assistant_message: Optional[str] = None
    action_taken: Optional[ActionType] = None
    safety_score: Optional[float] = None
    is_harmful: bool = False
    turn_number: int = 0


@dataclass
class GameState:
    conversation_history: List[ConversationTurn]
    turn_number: int
    has_violation: bool = False
    violation_turn: Optional[int] = None
    is_terminal: bool = False

    def get_history_text(self) -> str:
        history = []
        for turn in self.conversation_history:
            history.append(f"User: {turn.user_message}")
            if turn.assistant_message:
                history.append(f"Assistant: {turn.assistant_message}")
        return "\n".join(history)

    def get_last_k_turns(self, k: int) -> List[ConversationTurn]:
        return (
            self.conversation_history[-k:]
            if len(self.conversation_history) >= k
            else self.conversation_history
        )


class CensorshipGameEnvironment:
    def __init__(
        self,
        max_turns: int = 10,
        safety_threshold: float = 0.5,
        terminate_on_violation: bool = False,
    ):
        self.max_turns = max_turns
        self.safety_threshold = safety_threshold
        self.terminate_on_violation = terminate_on_violation
        self.reset()

    def reset(self) -> GameState:
        self.state = GameState(
            conversation_history=[],
            turn_number=0,
            has_violation=False,
            violation_turn=None,
            is_terminal=False,
        )
        return self.state

    def step(
        self,
        user_message: str,
        assistant_response: Optional[str],
        action_type: ActionType,
        is_harmful_turn: bool,
    ) -> Tuple[GameState, Dict[str, Any], bool]:
        turn = ConversationTurn(
            user_message=user_message,
            assistant_message=assistant_response,
            action_taken=action_type,
            safety_score=None,
            is_harmful=is_harmful_turn,
            turn_number=self.state.turn_number + 1,
        )

        self.state.conversation_history.append(turn)
        self.state.turn_number += 1

        if is_harmful_turn and action_type == ActionType.RESPOND:
            self.state.has_violation = True
            if self.state.violation_turn is None:
                self.state.violation_turn = self.state.turn_number

        is_terminal = (
            self.state.turn_number >= self.max_turns
            or action_type == ActionType.TERMINATE
            or (self.terminate_on_violation and self.state.has_violation)
        )
        self.state.is_terminal = is_terminal

        info = {
            "violation": self.state.has_violation,
            "violation_turn": self.state.violation_turn,
            "total_turns": self.state.turn_number,
            "action": action_type.value,
        }

        return self.state, info, is_terminal

    def get_state(self) -> GameState:
        return self.state

    def is_terminal(self) -> bool:
        return self.state.is_terminal
