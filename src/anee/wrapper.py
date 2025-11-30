import math
from typing import Tuple, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from .config import ANEEConfig
from .controller import build_controller
from .profiler import ANEEProfiler


class ANEEWrapper(nn.Module):
    """
    ANEE: Adaptive Neural Execution Engine

    Wraps a GPT-2 style transformer and adds:
      - Per-layer profiling
      - Learned / heuristic controller
      - Dynamic layer skipping (with KV-cache safety)
      - Optional RL training hooks
    """

    def __init__(self, config: ANEEConfig):
        super().__init__()

        self.config = config

        # Base model (GPT-2 family)
        print(f"[ANEE] Loading base model: {config.model_name}")
        self.base_model = GPT2LMHeadModel.from_pretrained(config.model_name)
        self.base_model.eval()

        # Core transformer components
        self.transformer = self.base_model.transformer
        self.layers = self.transformer.h
        self.total_layers = len(self.layers)

        # Final LN + LM head
        self.ln_f = self.transformer.ln_f
        self.lm_head = self.base_model.lm_head

        # Profiler (entropy, norms, variance, etc.)
        self.profiler = ANEEProfiler()

        # Controller (heuristic or learned)
        self.controller = build_controller(
            config=self.config,
            total_layers=self.total_layers,
            state_dim=self.config.state_dim,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return next(self.base_model.parameters()).device

    def _manual_split_heads(
        self,
        x: torch.Tensor,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        """
        GPT-2 style split_heads: (B, T, C) -> (B, num_heads, T, head_dim)
        """
        new_shape = x.size()[:-1] + (num_heads, head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def _get_state_vector(
        self,
        hidden_states: torch.Tensor,
        prev_hidden_states: torch.Tensor,
        layer_idx: int,
        budget_fraction: float,
    ) -> List[float]:
        """
        Single source of truth for feature extraction.
        Used by forward(), forward_rl(), forward_collect_states().
        """
        # Feature 1-3: Profiler stats (log scaled)
        h_norm = math.log1p(self.profiler.hidden_norm(hidden_states))
        delta_norm = math.log1p(
            self.profiler.delta_hidden_norm(prev_hidden_states, hidden_states)
        )
        var = math.log1p(self.profiler.variance(hidden_states))

        # Feature 4: "Urgency" – are we ahead/behind budget?
        layers_left = self.total_layers - layer_idx
        cost_to_finish = layers_left / float(self.total_layers)
        urgency = budget_fraction - cost_to_finish

        # Feature 5: raw budget
        # Feature 6: "GPS" – where in the stack are we?
        layer_fraction = layer_idx / float(self.total_layers)

        return [
            float(h_norm),
            float(delta_norm),
            float(var),
            float(urgency),
            float(budget_fraction),
            float(layer_fraction),
        ]

    # ------------------------------------------------------------------
    # Inference with KV-safe skipping
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        energy_budget: float = 1.0,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> dict:
        """
        KV-cache compatible forward pass with dynamic skipping.

        Args:
            input_ids: (batch=1, seq_len)
            energy_budget: normalized budget in [0, 1]
            past_key_values: list of (key, value) for each layer, or None

        Returns:
            {
                "logits": tensor,
                "past_key_values": [(k, v), ...],
                "stats": {
                    "layers_executed": int,
                    "skipped_layers": int,
                    "remaining_budget_fraction": float,
                    "exited_early": bool,
                    "total_layers": int,
                },
            }
        """
        assert input_ids.dim() == 2, "Expected (batch, seq_len)"
        device = self.device
        input_ids = input_ids.to(device)

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * self.total_layers
        else:
            past_length = past_key_values[0][0].size(-2)

        # Embeddings
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(
            past_length, past_length + seq_len, dtype=torch.long, device=device
        ).unsqueeze(0)

        hidden_states = (
            self.transformer.wte(input_ids) + self.transformer.wpe(position_ids)
        )
        hidden_states = self.transformer.drop(hidden_states)

        budget_fraction = max(0.0, min(1.0, float(energy_budget)))
        layers_executed = 0
        skipped_layers = 0
        exited_early = False
        present_key_values: List[Any] = []

        prev_hidden_states = hidden_states

        for layer_idx, layer_module in enumerate(self.layers):
            layer_past = past_key_values[layer_idx]

            # 1) OBSERVE – build state vector
            state_list = self._get_state_vector(
                hidden_states, prev_hidden_states, layer_idx, budget_fraction
            )
            state_tensor = torch.tensor(
                [state_list], dtype=torch.float32, device=device
            )

            # 2) DECIDE – controller action
            decision = self.controller.decide(
                layer_idx=layer_idx,
                budget_fraction=budget_fraction,
                state=state_tensor,
            )
            action = decision["action"]
            layer_cost = decision["layer_cost"]

            # Always process final two layers (keep output head stable)
            if layer_idx >= (self.total_layers - 2):
                action = "PROCESS"

            # 3) EXECUTE
            if action == "EXIT":
                exited_early = True
                break

            elif action == "SKIP":
                # Partial skip: update KV-cache, keep hidden_states unchanged
                attn = layer_module.attn
                head_dim = attn.head_dim
                # c_attn: (B, T, C) -> (B, T, 3C)
                qkv = attn.c_attn(hidden_states)
                _, key, value = qkv.split(attn.split_size, dim=2)

                key = self._manual_split_heads(key, attn.num_heads, head_dim)
                value = self._manual_split_heads(value, attn.num_heads, head_dim)

                if layer_past is not None:
                    key = torch.cat((layer_past[0], key), dim=-2)
                    value = torch.cat((layer_past[1], value), dim=-2)

                present_key_values.append((key, value))

                budget_fraction = max(0.0, budget_fraction - layer_cost)
                skipped_layers += 1
                # hidden_states, prev_hidden_states unchanged
                continue

            elif action == "PROCESS":
                # Full layer with cache
                outputs = layer_module(
                    hidden_states,
                    layer_past=layer_past,
                    use_cache=True,
                )
                hidden_states = outputs[0]
                kv = outputs[1] if len(outputs) > 1 else None
                present_key_values.append(kv)

                prev_hidden_states = hidden_states
                budget_fraction = max(0.0, budget_fraction - layer_cost)
                layers_executed += 1

        # Final projection
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return {
            "logits": logits,
            "past_key_values": present_key_values,
            "stats": {
                "layers_executed": layers_executed,
                "skipped_layers": skipped_layers,
                "remaining_budget_fraction": budget_fraction,
                "exited_early": exited_early,
                "total_layers": self.total_layers,
            },
        }

    # ------------------------------------------------------------------
    # RL version (no cache, simpler inner loop)
    # ------------------------------------------------------------------
    def forward_rl(
        self,
        input_ids: torch.Tensor,
        energy_budget: float = 1.0,
    ) -> dict:
        """
        RL-enabled forward pass.
        - Uses the same features as forward()
        - Enables log_prob tracking in the controller
        """
        self.eval()
        assert input_ids.size(0) == 1, "RL path assumes batch_size=1 for now"

        device = self.device
        input_ids = input_ids.to(device)

        if hasattr(self.controller, "reset_rl_trace"):
            self.controller.reset_rl_trace()

        # Embeddings
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)

        hidden_states = (
            self.transformer.wte(input_ids) + self.transformer.wpe(position_ids)
        )
        hidden_states = self.transformer.drop(hidden_states)

        budget_fraction = max(0.0, min(1.0, float(energy_budget)))
        layers_executed = 0
        skipped_layers = 0
        exited_early = False

        prev_hidden_states = hidden_states

        for layer_idx, layer_module in enumerate(self.layers):
            # 1) OBSERVE (shared)
            state_list = self._get_state_vector(
                hidden_states, prev_hidden_states, layer_idx, budget_fraction
            )
            state_tensor = torch.tensor(
                [state_list], dtype=torch.float32, device=device
            )

            # 2) DECIDE with track_logprob=True
            decision = self.controller.decide(
                layer_idx=layer_idx,
                budget_fraction=budget_fraction,
                state=state_tensor,
                track_logprob=True,
            )
            action = decision["action"]
            layer_cost = decision["layer_cost"]

            # Training constraint: don't skip output head during RL
            if layer_idx >= (self.total_layers - 2):
                action = "PROCESS"

            # 3) EXECUTE
            if action == "EXIT":
                exited_early = True
                break

            elif action == "SKIP":
                budget_fraction = max(0.0, budget_fraction - layer_cost)
                skipped_layers += 1
                continue

            elif action == "PROCESS":
                outputs = layer_module(hidden_states)
                hidden_states = outputs[0]
                prev_hidden_states = hidden_states
                budget_fraction = max(0.0, budget_fraction - layer_cost)
                layers_executed += 1

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        # Gather log_probs from controller
        if hasattr(self.controller, "saved_log_probs") and self.controller.saved_log_probs:
            log_probs = torch.stack(self.controller.saved_log_probs)
        else:
            log_probs = torch.zeros(1, device=device)

        return {
            "logits": logits,
            "stats": {
                "layers_executed": layers_executed,
                "skipped_layers": skipped_layers,
                "remaining_budget_fraction": budget_fraction,
                "exited_early": exited_early,
            },
            "rl": {
                "log_probs": log_probs,
            },
        }

    # ------------------------------------------------------------------
    # Supervised trace collection (for offline training)
    # ------------------------------------------------------------------
    def forward_collect_states(
        self,
        input_ids: torch.Tensor,
        energy_budget: float = 1.0,
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Collects state vectors per layer (and dummy actions).
        Used for supervised training / visualization.
        """
        device = self.device
        input_ids = input_ids.to(device)

        batch, seq_len = input_ids.shape
        position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)

        h = self.transformer.wte(input_ids) + self.transformer.wpe(position_ids)
        h = self.transformer.drop(h)

        budget_fraction = float(energy_budget)
        prev_h = h

        collected_states: List[List[float]] = []
        collected_actions: List[int] = []  # placeholder (e.g., 0=PROCESS)

        for layer_idx, layer_module in enumerate(self.layers):
            state_vec = self._get_state_vector(h, prev_h, layer_idx, budget_fraction)
            collected_states.append(state_vec)
            collected_actions.append(0)

            new_h = layer_module(h)[0]
            prev_h = h
            h = new_h

            budget_fraction = max(0.0, budget_fraction - (1.0 / self.total_layers))

        return collected_states, collected_actions

    # ------------------------------------------------------------------
    # Simple generate() helper for demos
    # ------------------------------------------------------------------
    def generate(
        self,
        tokenizer: GPT2TokenizerFast,
        prompt: str,
        max_new_tokens: int = 20,
        energy_budget: float = None,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> str:
        """
        Greedy / top-k sampling generation using ANEE forward().
        Reuses KV-cache across steps.
        """
        if energy_budget is None:
            energy_budget = self.config.energy_budget

        self.eval()
        device = self.device

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        input_seq = input_ids
        past_key_values = None

        for _ in range(max_new_tokens):
            if past_key_values is not None:
                model_input = input_seq[:, -1:]
            else:
                model_input = input_seq

            out = self.forward(
                model_input,
                energy_budget=energy_budget,
                past_key_values=past_key_values,
            )

            logits = out["logits"]
            past_key_values = out["past_key_values"]

            next_token_logits = logits[:, -1, :]  # (1, vocab)
            next_token_logits = next_token_logits / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                values, _ = torch.topk(next_token_logits, top_k)
                min_keep = values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_keep,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_seq = torch.cat([input_seq, next_token], dim=-1)

        return tokenizer.decode(input_seq[0], skip_special_tokens=True)
