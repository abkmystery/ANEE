import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import math

from .config import ANEEConfig
from .controller import build_controller
from .profiler import ANEEProfiler


class ANEEWrapper(nn.Module):
    def __init__(self, config: ANEEConfig):
        super().__init__()
        self.config = config
        print(f"[ANEE] Loading base model: {config.model_name}")
        self.base_model = GPT2LMHeadModel.from_pretrained(config.model_name)
        self.base_model.eval()

        # Force Cache Config
        self.base_model.config.use_cache = True
        self.base_model.config.output_attentions = False
        self.base_model.config.output_hidden_states = False

        self.transformer = self.base_model.transformer
        self.layers = self.transformer.h
        self.total_layers = len(self.layers)
        self.ln_f = self.transformer.ln_f
        self.lm_head = self.base_model.lm_head

        self.profiler = ANEEProfiler()
        self.controller = build_controller(
            config=self.config,
            total_layers=self.total_layers,
            state_dim=self.config.state_dim,
            device=self.device,
        )

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    def _manual_split_heads(self, tensor, num_heads, head_dim):
        new_shape = tensor.size()[:-1] + (num_heads, head_dim)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    # --- SHARED STATE LOGIC (The Fix) ---
    def _get_state_vector(self, hidden_states, prev_hidden_states, layer_idx, budget_fraction):
        """
        Single source of truth for state extraction.
        Ensures collect_traces and forward see identical features.
        """
        # Feature 1-3: Profiler Stats (Log Scaled)
        h_norm = math.log1p(self.profiler.hidden_norm(hidden_states))
        delta_norm = math.log1p(self.profiler.delta_hidden_norm(prev_hidden_states, hidden_states))
        var = math.log1p(self.profiler.variance(hidden_states))

        # Feature 4: Urgency
        layers_left = self.total_layers - layer_idx
        cost_to_finish = layers_left / float(self.total_layers)
        urgency = budget_fraction - cost_to_finish
        # NEW FEATURE: GPS (Where am I?)
        layer_fraction = layer_idx / float(self.total_layers)

        # Return 6 features
        return [float(h_norm), float(delta_norm), float(var), float(urgency), float(budget_fraction),
                float(layer_fraction)]


    def forward(self, input_ids: torch.Tensor, energy_budget: float = 1.0, past_key_values=None):
        assert input_ids.dim() == 2
        device = self.device
        input_ids = input_ids.to(device)

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.layers)
        else:
            past_length = past_key_values[0][0].size(-2)

        # Embeddings
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_len)

        hidden_states = self.transformer.wte(input_ids) + self.transformer.wpe(position_ids)
        hidden_states = self.transformer.drop(hidden_states)

        budget_fraction = max(0.0, min(1.0, float(energy_budget)))
        layers_executed = 0
        skipped_layers = 0
        exited_early = False
        present_key_values = []
        prev_hidden_states = hidden_states

        for layer_idx, layer_module in enumerate(self.layers):
            layer_past = past_key_values[layer_idx]

            # 1. OBSERVE (Using Shared Method)
            state_list = self._get_state_vector(hidden_states, prev_hidden_states, layer_idx, budget_fraction)
            state_tensor = torch.tensor([state_list], dtype=torch.float32, device=device)

            # 2. DECIDE
            decision = self.controller.decide(layer_idx, budget_fraction, state_tensor)
            action = decision["action"]
            layer_cost = decision["layer_cost"]

            if layer_idx >= (self.total_layers - 2):
                action = "PROCESS"

            # 3. EXECUTE
            if action == "EXIT":
                # print(f"[ANEE] EXIT L{layer_idx}") # Uncomment for debug
                exited_early = True
                break
            elif action == "SKIP":
                # Partial Skip
                attn = layer_module.attn
                head_dim = attn.head_dim
                query, key, value = attn.c_attn(hidden_states).split(attn.split_size, dim=2)
                key = self._manual_split_heads(key, attn.num_heads, head_dim)
                value = self._manual_split_heads(value, attn.num_heads, head_dim)

                if layer_past is not None:
                    key = torch.cat((layer_past[0], key), dim=-2)
                    value = torch.cat((layer_past[1], value), dim=-2)
                present_key_values.append((key, value))

                budget_fraction = max(0.0, budget_fraction - layer_cost)
                skipped_layers += 1
                continue  # Hidden states pass through
            elif action == "PROCESS":
                outputs = layer_module(hidden_states, layer_past=layer_past, use_cache=True)
                hidden_states = outputs[0]
                present_key_values.append(outputs[1] if len(outputs) > 1 else None)

                prev_hidden_states = hidden_states
                budget_fraction = max(0.0, budget_fraction - layer_cost)
                layers_executed += 1

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return {
            "logits": logits,
            "past_key_values": present_key_values,
            "stats": {
                "layers_executed": layers_executed,
                "skipped_layers": skipped_layers,
                "exited_early": exited_early,
            }
        }

    def forward_rl(self, input_ids: torch.Tensor, energy_budget: float = 1.0, past_key_values=None):
        """ Turbo Mode for Training """
        self.eval()
        device = self.device
        input_ids = input_ids.to(device)
        past_length = 0  # Turbo mode ignores cache length for speed

        position_ids = torch.arange(past_length, past_length + input_ids.size(1), dtype=torch.long,
                                    device=device).unsqueeze(0)
        hidden_states = self.transformer.wte(input_ids) + self.transformer.wpe(position_ids)
        hidden_states = self.transformer.drop(hidden_states)

        if hasattr(self.controller, "reset_rl_trace"):
            self.controller.reset_rl_trace()

        budget_fraction = max(0.0, min(1.0, float(energy_budget)))
        layers_executed = 0
        prev_hidden_states = hidden_states

        for layer_idx, layer_module in enumerate(self.layers):
            # 1. OBSERVE (Shared)
            state_list = self._get_state_vector(hidden_states, prev_hidden_states, layer_idx, budget_fraction)
            state_tensor = torch.tensor([state_list], dtype=torch.float32, device=device)

            # 2. DECIDE
            decision = self.controller.decide(layer_idx, budget_fraction, state_tensor, track_logprob=True)
            action = decision["action"]

            # --- TRAINING CONSTRAINT ---
            # During RL training, forbid skipping the output head.
            # This helps the agent converge to a working policy faster.
            if layer_idx >= (self.total_layers - 2):
                action = "PROCESS"
            layer_cost = decision["layer_cost"]

            # 3. EXECUTE (Turbo)
            if action == "EXIT":
                break
            elif action == "SKIP":
                budget_fraction = max(0.0, budget_fraction - layer_cost)
                continue
            elif action == "PROCESS":
                outputs = layer_module(hidden_states)  # No cache args needed
                hidden_states = outputs[0]
                prev_hidden_states = hidden_states
                budget_fraction = max(0.0, budget_fraction - layer_cost)
                layers_executed += 1

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        log_probs = torch.zeros(1, device=device)
        if hasattr(self.controller, "saved_log_probs") and self.controller.saved_log_probs:
            log_probs = torch.stack(self.controller.saved_log_probs)

        return {
            "logits": logits,
            "stats": {"layers_executed": layers_executed},
            "rl": {"log_probs": log_probs}
        }

    def forward_collect_states(self, input_ids, energy_budget=1.0):
        """ Data Collection using Shared Logic """
        device = self.device
        input_ids = input_ids.to(device)
        position_ids = torch.arange(0, input_ids.size(1), device=device).unsqueeze(0)

        h = self.transformer.wte(input_ids) + self.transformer.wpe(position_ids)
        h = self.transformer.drop(h)

        budget_fraction = float(energy_budget)
        prev_h = h
        collected_states = []

        for layer_idx, layer_module in enumerate(self.layers):
            # 1. OBSERVE (Shared)
            state_vec = self._get_state_vector(h, prev_h, layer_idx, budget_fraction)
            collected_states.append(state_vec)

            # Run Layer (Always run to get next state)
            new_h = layer_module(h)[0]
            prev_h = h
            h = new_h

            # Decrease budget for simulation
            budget_fraction = max(0.0, budget_fraction - (1.0 / self.total_layers))

        return collected_states, []  # Actions ignored by collector anyway

    def generate(self, tokenizer, prompt, max_new_tokens=20, energy_budget=1.0, temperature=0.7, top_k=50):
        if energy_budget is None: energy_budget = self.config.energy_budget
        self.eval()
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_seq = input_ids
        past_key_values = None

        for _ in range(max_new_tokens):
            if past_key_values is not None:
                model_input = input_seq[:, -1:]
            else:
                model_input = input_seq

            out = self.forward(model_input, energy_budget=energy_budget, past_key_values=past_key_values)
            next_token_logits = out['logits'][:, -1, :] / temperature

            # Top-K Sampling
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            input_seq = torch.cat([input_seq, next_token_id], dim=-1)
            past_key_values = out['past_key_values']

        return tokenizer.decode(input_seq[0], skip_special_tokens=True)