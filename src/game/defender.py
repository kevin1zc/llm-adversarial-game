from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM
from .environment import GameState, ActionType


class DefenderPolicy(nn.Module):
    def __init__(
        self,
        model_name: str = "gpt2",
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        hidden_size = self.base_model.config.hidden_size

        self.decision_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        if attention_mask is None:
            pooled = hidden_states[:, -1, :]
        else:
            last_idx = attention_mask.sum(dim=1).clamp_min(1) - 1
            pooled = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), last_idx]
        decision_logits = self.decision_head(pooled)
        return decision_logits, outputs


class DefenderAgent:
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512,
        temperature: float = 0.7,
        use_policy: bool = True,
    ):
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.use_policy = use_policy
        self.model_name = model_name

        self.policy = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.tokenizer.padding_side = "left"
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            self.tokenizer = None

        if not use_policy:
            raise ValueError("DefenderAgent requires use_policy=True")

        self.policy = DefenderPolicy(model_name).to(device)
        self.policy.train()

    def decide_action(
        self,
        state: GameState,
        user_message: str,
        safety_score: float,
        safety_threshold: float = 0.5,
        explore: bool = False,
        generate_response: bool = True,
        refuse_threshold: float = 0.5,
    ) -> Tuple[ActionType, Optional[str]]:
        action, resp, _ = self.decide_action_with_logprob(
            state=state,
            user_message=user_message,
            explore=explore,
            generate_response=generate_response,
            refuse_threshold=refuse_threshold,
            do_sample=True,
        )
        return action, resp

    def decide_action_with_logprob(
        self,
        state: GameState,
        user_message: str,
        explore: bool = True,
        generate_response: bool = True,
        refuse_threshold: float = 0.5,
        do_sample: bool = True,
    ) -> Tuple[ActionType, Optional[str], torch.Tensor]:
        if self.policy is None or self.tokenizer is None:
            raise RuntimeError("DefenderAgent policy/tokenizer not initialized")

        history = state.get_history_text()
        prompt = f"{history}\nUser: {user_message}\nAssistant:" if history else f"User: {user_message}\nAssistant:"
        inps = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)

        decision_logits, _ = self.policy(**inps)
        probs = torch.softmax(decision_logits, dim=-1)
        if explore:
            dist = Categorical(probs)
            action_idx = dist.sample()
            logp_dec = dist.log_prob(action_idx)
            action_idx = int(action_idx.item())
        else:
            p_refuse = float(probs[0, 0].item())
            action_idx = 0 if p_refuse >= float(refuse_threshold) else 1
            logp_dec = torch.log(probs[0, action_idx].clamp_min(1e-12))

        if action_idx == 0:
            refusal = self._generate_refusal_with_logprob(inps, do_sample=do_sample)
            return ActionType.REFUSE, refusal[0], (logp_dec + refusal[1]).squeeze()

        if not generate_response:
            return ActionType.RESPOND, "", logp_dec.squeeze()

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
            return ActionType.RESPOND, "", logp_dec.squeeze()

        out = self.policy.base_model(input_ids=full, attention_mask=torch.ones_like(full))
        logits = out.logits[:, :-1, :]
        targets = full[:, 1:]
        start = max(prompt_len - 1, 0)
        sel_logits = logits[:, start:, :]
        sel_targets = targets[:, start:]
        sel_targets = sel_targets[:, -new_ids.shape[1] :]
        log_probs = torch.log_softmax(sel_logits[:, -new_ids.shape[1] :, :], dim=-1)
        token_logp = log_probs.gather(2, sel_targets.unsqueeze(-1)).squeeze(-1)
        logp_gen = token_logp.sum()

        ids = new_ids[0].detach().to("cpu").tolist()
        txt = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        return ActionType.RESPOND, txt, (logp_dec + logp_gen).squeeze()

    def _generate_refusal_with_logprob(
        self, prompt_inputs, do_sample: bool = True
    ) -> Tuple[str, torch.Tensor]:
        with torch.no_grad():
            gen = self.policy.base_model.generate(
                prompt_inputs.input_ids,
                attention_mask=prompt_inputs.attention_mask,
                max_new_tokens=24,
                do_sample=bool(do_sample),
                temperature=self.temperature if bool(do_sample) else None,
                top_p=0.9 if bool(do_sample) else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        full = gen[0].unsqueeze(0)
        prompt_len = prompt_inputs.input_ids.shape[1]
        new_ids = full[:, prompt_len:]
        if new_ids.numel() == 0:
            return "", torch.zeros((), device=self.device)
        out = self.policy.base_model(input_ids=full, attention_mask=torch.ones_like(full))
        logits = out.logits[:, :-1, :]
        targets = full[:, 1:]
        start = max(prompt_len - 1, 0)
        sel_logits = logits[:, start:, :]
        sel_targets = targets[:, start:]
        sel_targets = sel_targets[:, -new_ids.shape[1] :]
        log_probs = torch.log_softmax(sel_logits[:, -new_ids.shape[1] :, :], dim=-1)
        token_logp = log_probs.gather(2, sel_targets.unsqueeze(-1)).squeeze(-1)
        logp = token_logp.sum()
        ids = new_ids[0].detach().to("cpu").tolist()
        txt = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        return txt, logp

    def _generate_refusal(self, user_message: str) -> str:
        action, txt, _ = self.decide_action_with_logprob(
            state=GameState(conversation_history=[], turn_number=0),
            user_message=user_message,
            explore=False,
            generate_response=True,
            refuse_threshold=0.0,
            do_sample=True,
        )
        return txt

    def load_checkpoint(self, checkpoint_path: str):
        if self.policy is not None:
            self.policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.policy.train()
