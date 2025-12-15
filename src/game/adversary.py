from typing import List, Optional, Dict, Any, Tuple
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from .environment import GameState, ConversationTurn
from src.core import ConversationExample


class AdversaryPolicy(nn.Module):
    def __init__(
        self,
        model_name: str = "gpt2",
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        hidden_size = self.base_model.config.hidden_size

        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        if attention_mask is None:
            pooled = hidden_states.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            summed = (hidden_states * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = summed / denom

        value = self.value_head(pooled)
        strategy_logits = self.strategy_head(pooled)

        return outputs, value, strategy_logits

    def generate(self, input_ids, attention_mask=None, **kwargs):
        return self.base_model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )


class AdversaryAgent:
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 256,
        temperature: float = 0.9,
        use_learned_policy: bool = False,
    ):
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.use_learned_policy = use_learned_policy
        self.model_name = model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.tokenizer.padding_side = "left"
        except Exception as e:
            print(f"Warning: Could not load tokenizer {model_name}: {e}")
            self.tokenizer = None

        if not use_learned_policy:
            raise ValueError("AdversaryAgent requires use_learned_policy=True")

        self.policy = None
        self.policy = AdversaryPolicy(model_name).to(device)
        self.policy.train()

    def generate_message(
        self,
        state: GameState,
        target_harmful_content: Optional[str] = None,
        turn_strategy: str = "gradual",
    ) -> str:
        msg, _ = self.generate_message_with_logprob(
            state=state, target_harmful_content=target_harmful_content, do_sample=True
        )
        return msg

    def generate_message_with_logprob(
        self,
        state: GameState,
        target_harmful_content: Optional[str] = None,
        do_sample: bool = True,
    ) -> Tuple[str, torch.Tensor]:
        if self.policy is None or self.tokenizer is None:
            raise RuntimeError("AdversaryAgent policy/tokenizer not initialized")

        history = state.get_history_text()
        if target_harmful_content:
            prompt = f"{history}\n[Goal: {target_harmful_content}]\nUser:"
        else:
            prompt = f"{history}\nUser:" if history else "User:"

        inps = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            gen = self.policy.base_model.generate(
                inps.input_ids,
                attention_mask=inps.attention_mask,
                max_new_tokens=24,
                do_sample=bool(do_sample),
                temperature=self.temperature if bool(do_sample) else None,
                top_p=0.9 if bool(do_sample) else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        full = gen[0].unsqueeze(0)
        prompt_len = inps.input_ids.shape[1]
        new_ids = full[:, prompt_len:]

        if new_ids.numel() == 0:
            return "", torch.zeros((), device=self.device)

        out = self.policy.base_model(input_ids=full, attention_mask=torch.ones_like(full))
        logits = out.logits[:, :-1, :]
        targets = full[:, 1:]

        start = max(prompt_len - 1, 0)
        end = logits.shape[1]
        sel_logits = logits[:, start:end, :]
        sel_targets = targets[:, start:end]
        sel_targets = sel_targets[:, -new_ids.shape[1] :]
        log_probs = torch.log_softmax(sel_logits[:, -new_ids.shape[1] :, :], dim=-1)
        token_logp = log_probs.gather(2, sel_targets.unsqueeze(-1)).squeeze(-1)
        logp = token_logp.sum()

        ids = new_ids[0].detach().to("cpu").tolist()
        txt = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        if "\n" in txt:
            txt = txt.split("\n")[0].strip()
        return txt, logp

    def save_policy(self, path: str):
        if self.policy is not None:
            torch.save(self.policy.state_dict(), path)

    def load_policy(self, path: str):
        if self.policy is not None:
            self.policy.load_state_dict(torch.load(path, map_location=self.device))
