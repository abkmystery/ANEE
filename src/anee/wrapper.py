import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from .config import ANEEConfig
from .controller import ANEEController


class ANEEWrapper(nn.Module):
    """
    ANEE Phase-1:
    - Wraps GPT-2
    - Manually unrolls the transformer blocks
    - Applies early-exit based on a normalized energy budget
    """

    def __init__(self, config: ANEEConfig):
        super().__init__()

        self.config = config

        print(f"[ANEE] Loading base model: {config.model_name}")
        self.base_model = GPT2LMHeadModel.from_pretrained(config.model_name)
        self.base_model.eval()

        # Core transformer components
        self.transformer = self.base_model.transformer
        self.layers = self.transformer.h
        self.total_layers = len(self.layers)

        # Final layer norm and LM head
        self.ln_f = self.transformer.ln_f
        self.lm_head = self.base_model.lm_head

        # Controller
        self.controller = ANEEController(
            total_layers=self.total_layers,
            min_layers=config.min_layers,
            exit_budget_threshold=config.exit_budget_threshold,
        )

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    def forward(self, input_ids: torch.Tensor, energy_budget: float = 1.0):
        """
        Custom forward with early-exit.
        Args:
            input_ids: (batch, seq_len)
            energy_budget: normalized scalar in [0, 1]
        Returns:
            dict with:
              - 'logits'
              - 'stats' (layers_executed, remaining_budget_fraction, exited_early)
        """
        # For Phase-1 we assume batch_size = 1 (keep it simple)
        # Later we can generalize
        assert input_ids.dim() == 2, "Expected (batch, seq_len)"
        assert input_ids.size(0) == 1, "Phase-1 assumes batch_size == 1"

        input_ids = input_ids.to(self.device)

        # --- Embeddings ---
        # GPT-2: word + positional embeddings
        input_shape = input_ids.size()
        batch_size, seq_len = input_shape

        position_ids = torch.arange(
            0, seq_len, dtype=torch.long, device=self.device
        ).unsqueeze(0)  # (1, seq_len)

        inputs_embeds = self.transformer.wte(input_ids)        # (1, T, d_model)
        position_embeds = self.transformer.wpe(position_ids)   # (1, T, d_model)

        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.transformer.drop(hidden_states)

        # Budget tracking (fraction in [0, 1])
        budget_fraction = max(0.0, min(1.0, float(energy_budget)))
        layers_executed = 0
        exited_early = False

        # --- Unrolled Layer Loop ---
        for layer_idx, layer_module in enumerate(self.layers):
            decision = self.controller.decide(
                layer_idx=layer_idx,
                budget_fraction=budget_fraction,
            )
            action = decision["action"]
            layer_cost = decision["layer_cost"]

            if action == "EXIT":
                print(f"[ANEE] Early exit at layer {layer_idx} "
                      f"(budget_fraction={budget_fraction:.3f})")
                exited_early = True
                break

            # PROCESS
            # GPT-2 block returns a tuple: (hidden_states, present, ...)
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs[0]

            # Update budget
            budget_fraction = max(0.0, budget_fraction - layer_cost)
            layers_executed += 1

        # --- Final LayerNorm + LM Head ---
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return {
            "logits": logits,
            "stats": {
                "layers_executed": layers_executed,
                "remaining_budget_fraction": budget_fraction,
                "exited_early": exited_early,
                "total_layers": self.total_layers,
            },
        }

    def generate(
        self,
        tokenizer: GPT2TokenizerFast,
        prompt: str,
        max_new_tokens: int = 20,
        energy_budget: float = None,
    ) -> str:
        """
        Simple greedy decoding using ANEE forward.
        For Phase-1: we re-use the same energy_budget for each token.
        """
        if energy_budget is None:
            energy_budget = self.config.energy_budget

        self.eval()

        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated = input_ids

        for _ in range(max_new_tokens):
            outputs = self.forward(generated, energy_budget=energy_budget)
            logits = outputs["logits"]
            next_token_logits = logits[:, -1, :]  # last position

            # Greedy decode
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token_id], dim=-1)

        return tokenizer.decode(generated[0], skip_special_tokens=True)
