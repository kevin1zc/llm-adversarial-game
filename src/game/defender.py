"""Defender agent that decides whether to respond or refuse."""

from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM
from .environment import GameState, ActionType


class DefenderPolicy(nn.Module):
    """Policy network for the defender agent."""

    def __init__(
        self,
        model_name: str = "gpt2",
        hidden_dim: int = 256,
    ):
        """Initialize defender policy."""
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        hidden_size = self.base_model.config.hidden_size

        # Decision head: respond vs refuse
        self.decision_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # [respond, refuse]
        )

        # Response generation uses base model

    def forward(self, input_ids, attention_mask=None):
        """Forward pass."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Use last hidden state for decision
        hidden_states = outputs.hidden_states[-1]
        # Pool over sequence length (mean pooling)
        pooled = hidden_states.mean(dim=1)
        decision_logits = self.decision_head(pooled)
        return decision_logits, outputs


class DefenderAgent:
    """Defender agent that decides whether to respond or refuse."""

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512,
        temperature: float = 0.7,
        use_policy: bool = True,
    ):
        """
        Initialize defender agent.

        Args:
            model_name: Name of the language model
            device: Device to run on
            max_length: Maximum generation length
            temperature: Sampling temperature
            use_policy: Whether to use learned policy or rule-based
        """
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.use_policy = use_policy
        self.model_name = model_name

        # Initialize model to None first
        self.model = None
        self.policy = None

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                # Add a proper pad token instead of reusing eos_token
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                # Set padding side to left for decoder-only models (like GPT-2)
                self.tokenizer.padding_side = "left"
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            self.tokenizer = None

        # Initialize policy if using learned policy
        if use_policy:
            try:
                self.policy = DefenderPolicy(model_name).to(device)
                self.policy.eval()
                # Policy has base_model, so we can use it for generation
            except Exception as e:
                print(f"Warning: Could not load policy: {e}")
                self.use_policy = False
                self.policy = None
                # Fall back to loading base model
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
                    self.model.eval()
                except Exception as e2:
                    print(f"Warning: Could not load model: {e2}")
                    self.model = None
        else:
            # Load base model for response generation
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
                self.model.eval()
            except Exception as e:
                print(f"Warning: Could not load model: {e}")
                self.model = None

    def decide_action(
        self,
        state: GameState,
        user_message: str,
        safety_score: float,
        safety_threshold: float = 0.5,
        explore: bool = False,
    ) -> Tuple[ActionType, Optional[str]]:
        """
        Decide whether to respond or refuse.

        Args:
            state: Current game state
            user_message: User's message
            safety_score: Safety score of the message
            safety_threshold: Threshold for safety

        Returns:
            Tuple of (action_type, response_message)
        """
        # Rule-based decision (can be replaced with learned policy)
        if self.use_policy and self.policy is not None:
            return self._decide_with_policy(state, user_message, safety_score, explore)
        else:
            return self._decide_rule_based(state, user_message, safety_score, safety_threshold)

    def _decide_with_policy(
        self,
        state: GameState,
        user_message: str,
        safety_score: float,
        explore: bool = False,
    ) -> Tuple[ActionType, Optional[str]]:
        """Decide using learned policy."""
        # Prepare input
        history = state.get_history_text()
        prompt = f"{history}\nUser: {user_message}\nAssistant:"

        if self.tokenizer is None:
            return ActionType.REFUSE, "I cannot respond to that."

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)

        with torch.no_grad():
            decision_logits, _ = self.policy(**inputs)
            decision_probs = torch.softmax(decision_logits, dim=-1)
            if explore:
                dist = Categorical(decision_probs)
                action_idx = dist.sample().item()
            else:
                action_idx = decision_probs.argmax(dim=-1).item()

        if action_idx == 0:  # respond
            response = self._generate_response(state, user_message)
            return ActionType.RESPOND, response
        else:  # refuse
            return ActionType.REFUSE, self._generate_refusal(user_message)

    def _decide_rule_based(
        self,
        state: GameState,
        user_message: str,
        safety_score: float,
        safety_threshold: float,
    ) -> Tuple[ActionType, Optional[str]]:
        """Rule-based decision making."""
        # Refuse if safety score is below threshold
        if safety_score < safety_threshold:
            return ActionType.REFUSE, self._generate_refusal(user_message)

        # Check conversation history for suspicious patterns
        if self._is_suspicious_conversation(state, user_message):
            return ActionType.REFUSE, self._generate_refusal(user_message)

        # Otherwise, respond
        response = self._generate_response(state, user_message)
        return ActionType.RESPOND, response

    def _is_suspicious_conversation(self, state: GameState, user_message: str) -> bool:
        """Check if conversation is suspicious."""
        suspicious_keywords = [
            "hack",
            "bomb",
            "harm",
            "illegal",
            "weapon",
            "drug",
            "violence",
            "attack",
            "exploit",
            "bypass",
        ]

        # Check current message
        user_lower = user_message.lower()
        if any(keyword in user_lower for keyword in suspicious_keywords):
            return True

        # Check history for escalation patterns
        if len(state.conversation_history) > 0:
            # If user keeps asking similar questions, might be suspicious
            recent_messages = [
                turn.user_message.lower() for turn in state.conversation_history[-2:]
            ]
            if len(recent_messages) >= 2:
                # Check for repeated probing
                if any(
                    keyword in msg for msg in recent_messages for keyword in suspicious_keywords
                ):
                    return True

        return False

    def _generate_response(self, state: GameState, user_message: str) -> str:
        """Generate a response to the user message."""
        # Check if we have a model or policy with base_model
        model_for_gen = None
        if self.model is not None:
            model_for_gen = self.model
        elif self.policy is not None and hasattr(self.policy, "base_model"):
            model_for_gen = self.policy.base_model
        else:
            return (
                "I understand your question, but I need more context to provide a helpful answer."
            )

        if self.tokenizer is None:
            return (
                "I understand your question, but I need more context to provide a helpful answer."
            )

        # Prepare prompt
        history = state.get_history_text()
        prompt = f"{history}\nUser: {user_message}\nAssistant:"

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)

        with torch.no_grad():
            outputs = model_for_gen.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the new response
        response = generated_text[len(prompt) :].strip()
        return response if response else "I'm not sure how to respond to that."

    def _generate_refusal(self, user_message: str) -> str:
        """Generate a refusal message."""
        refusal_templates = [
            "I can't help with that request.",
            "I'm not able to assist with that.",
            "I can't provide information on that topic.",
            "I'm sorry, but I can't help with that.",
        ]
        import random

        return random.choice(refusal_templates)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        if self.policy is not None:
            self.policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.policy.eval()
