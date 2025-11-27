import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import math

from .config import ANEEConfig
from .controller import build_controller
from .profiler import ANEEProfiler


class ANEEWrapper(nn.Module):
    """
    ANEE Phase-4:
    - Wraps GPT-2
    - Implements True Layer Skipping (Sparse Inference)
    - Implements Partial Skip (KV-Cache alignment) using robust manual reshaping
    - Uses 'Fast Observer' (No intermediate logit projection)
    """

    def __init__(self, config: ANEEConfig):
        super().__init__()

        self.config = config

        print(f"[ANEE] Loading base model: {config.model_name}")
        self.base_model = GPT2LMHeadModel.from_pretrained(config.model_name)
        self.base_model.eval()

        # --- CRITICAL FIX: FORCE CACHE CONFIGURATION ---
        self.base_model.config.use_cache = True
        self.base_model.config.output_attentions = False
        self.base_model.config.output_hidden_states = False

        # Core transformer components
        self.transformer = self.base_model.transformer
        self.layers = self.transformer.h
        self.total_layers = len(self.layers)

        # Final layer norm and LM head
        self.ln_f = self.transformer.ln_f
        self.lm_head = self.base_model.lm_head

        # Profiler for cheap state extraction
        self.profiler = ANEEProfiler()

        # Controller (heuristic or learned)
        self.controller = build_controller(
            config=self.config,
            total_layers=self.total_layers,
            state_dim=self.config.state_dim,  # Should be 5
            device=self.device,
        )

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    def _manual_split_heads(self, tensor, num_heads, head_dim):
        """
        Manually reshapes projection output to KV-Cache format.
        Input: (Batch, Seq_Len, Hidden_Dim)
        Output: (Batch, Num_Heads, Seq_Len, Head_Dim)
        """
        new_shape = tensor.size()[:-1] + (num_heads, head_dim)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, input_ids: torch.Tensor, energy_budget: float = 1.0, past_key_values=None):
        """
        Custom forward with Dynamic Skipping and Early Exit.
        Handles KV-Cache via Partial Skipping.
        """
        assert input_ids.dim() == 2, "Expected (batch, seq_len)"

        device = self.device
        input_ids = input_ids.to(device)

        # Handle Cache Setup
        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.layers)
        else:
            past_length = past_key_values[0][0].size(-2)

        # --- Embeddings ---
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_len)

        inputs_embeds = self.transformer.wte(input_ids)
        position_embeds = self.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.transformer.drop(hidden_states)

        # Trackers
        budget_fraction = max(0.0, min(1.0, float(energy_budget)))
        layers_executed = 0
        skipped_layers = 0
        exited_early = False

        present_key_values = []
        prev_hidden_states = hidden_states

        for layer_idx, layer_module in enumerate(self.layers):
            layer_past = past_key_values[layer_idx]

            # --- 1. OBSERVATION ---
            h_norm = math.log1p(self.profiler.hidden_norm(hidden_states))
            delta_norm = math.log1p(self.profiler.delta_hidden_norm(prev_hidden_states, hidden_states))
            var = math.log1p(self.profiler.variance(hidden_states))

            # Urgency
            layers_left = self.total_layers - layer_idx
            cost = layers_left / float(self.total_layers)
            urgency = budget_fraction - cost

            state_tensor = torch.tensor([[h_norm, delta_norm, var, urgency, budget_fraction]],
                                        dtype=torch.float32, device=device)

            # --- 2. DECISION ---
            decision = self.controller.decide(
                layer_idx=layer_idx,
                budget_fraction=budget_fraction,
                state=state_tensor,
            )
            action = decision["action"]
            layer_cost = decision["layer_cost"]

            # --- 3. EXECUTION ---
            if action == "EXIT":
                print(f"[ANEE] Early exit at layer {layer_idx} (budget={budget_fraction:.3f})")
                exited_early = True
                break

            elif action == "SKIP":
                # --- PARTIAL SKIP (Maintain Memory, Save Compute) ---
                attn = layer_module.attn
                head_dim = attn.head_dim

                # 1. Project to Q, K, V (Cheap)
                # GPT-2 specific: c_attn returns (Q,K,V) flattened
                query, key, value = attn.c_attn(hidden_states).split(attn.split_size, dim=2)

                # 2. Reshape (Using Manual Helper to avoid version errors)
                key = self._manual_split_heads(key, attn.num_heads, head_dim)
                value = self._manual_split_heads(value, attn.num_heads, head_dim)

                # 3. Append to Cache
                if layer_past is not None:
                    key = torch.cat((layer_past[0], key), dim=-2)
                    value = torch.cat((layer_past[1], value), dim=-2)

                present_key_values.append((key, value))

                # 4. Skip the rest
                budget_fraction = max(0.0, budget_fraction - layer_cost)
                skipped_layers += 1
                continue

            elif action == "PROCESS":
                # Run the layer
                outputs = layer_module(
                    hidden_states,
                    layer_past=layer_past,
                    use_cache=True
                )

                hidden_states = outputs[0]

                if len(outputs) > 1:
                    present_key_values.append(outputs[1])
                else:
                    present_key_values.append(None)

                prev_hidden_states = hidden_states
                budget_fraction = max(0.0, budget_fraction - layer_cost)
                layers_executed += 1

        # --- Final LayerNorm + LM Head ---
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

    def forward_rl(self, input_ids: torch.Tensor, energy_budget: float = 1.0, past_key_values=None):
        """
        RL-Enabled Forward Pass.
        """
        self.eval()
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

        if hasattr(self.controller, "reset_rl_trace"):
            self.controller.reset_rl_trace()

        budget_fraction = max(0.0, min(1.0, float(energy_budget)))
        layers_executed = 0
        skipped_layers = 0
        exited_early = False

        present_key_values = []
        prev_hidden_states = hidden_states

        for layer_idx, layer_module in enumerate(self.layers):
            layer_past = past_key_values[layer_idx]

            # --- Observation ---
            h_norm = math.log1p(self.profiler.hidden_norm(hidden_states))
            delta_norm = math.log1p(self.profiler.delta_hidden_norm(prev_hidden_states, hidden_states))
            var = math.log1p(self.profiler.variance(hidden_states))

            layers_left = self.total_layers - layer_idx
            cost = layers_left / float(self.total_layers)
            urgency = budget_fraction - cost

            state_tensor = torch.tensor([[h_norm, delta_norm, var, urgency, budget_fraction]],
                                        dtype=torch.float32, device=device)

            # --- Decision ---
            decision = self.controller.decide(
                layer_idx=layer_idx,
                budget_fraction=budget_fraction,
                state=state_tensor,
                track_logprob=True,
            )
            action = decision["action"]
            layer_cost = decision["layer_cost"]

            # --- Execution ---
            if action == "EXIT":
                exited_early = True
                break

            elif action == "SKIP":
                # Partial Skip with manual reshaping
                attn = layer_module.attn
                head_dim = attn.head_dim
                query, key, value = attn.c_attn(hidden_states).split(attn.split_size, dim=2)

                # MANUAL RESHAPE (Safe)
                key = self._manual_split_heads(key, attn.num_heads, head_dim)
                value = self._manual_split_heads(value, attn.num_heads, head_dim)

                if layer_past is not None:
                    key = torch.cat((layer_past[0], key), dim=-2)
                    value = torch.cat((layer_past[1], value), dim=-2)

                present_key_values.append((key, value))

                budget_fraction = max(0.0, budget_fraction - layer_cost)
                skipped_layers += 1
                continue

            elif action == "PROCESS":
                outputs = layer_module(
                    hidden_states,
                    layer_past=layer_past,
                    use_cache=True
                )

                hidden_states = outputs[0]
                if len(outputs) > 1:
                    present_key_values.append(outputs[1])
                else:
                    present_key_values.append(None)

                prev_hidden_states = hidden_states
                budget_fraction = max(0.0, budget_fraction - layer_cost)
                layers_executed += 1

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        if hasattr(self.controller, "saved_log_probs") and self.controller.saved_log_probs:
            log_probs = torch.stack(self.controller.saved_log_probs)
        else:
            log_probs = torch.zeros(1, device=device)

        return {
            "logits": logits,
            "past_key_values": present_key_values,
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

    def forward_collect_states(self, input_ids, energy_budget=1.0):
        """
        Data collection for supervised training.
        """
        profiler = self.profiler
        device = self.device
        input_ids = input_ids.to(device)

        batch, seq_len = input_ids.shape
        position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)
        h = self.transformer.wte(input_ids) + self.transformer.wpe(position_ids)
        h = self.transformer.drop(h)

        budget_fraction = float(energy_budget)
        prev_h = h

        collected_states = []
        collected_actions = []

        for layer_idx, layer_module in enumerate(self.layers):
            # Observe
            h_norm = math.log1p(profiler.hidden_norm(h))
            delta_norm = math.log1p(profiler.delta_hidden_norm(prev_h, h))
            var = math.log1p(profiler.variance(h))

            layers_left = self.total_layers - layer_idx
            cost = layers_left / float(self.total_layers)
            urgency = budget_fraction - cost

            state_vec = [
                h_norm,
                delta_norm,
                var,
                urgency,
                budget_fraction,
            ]
            collected_states.append(state_vec)

            # Heuristic
            exit_threshold = self.config.exit_budget_threshold
            if layer_idx >= self.config.min_layers and budget_fraction < exit_threshold:
                collected_actions.append(1)  # EXIT
                break
            else:
                collected_actions.append(0)  # PROCESS

            # Run Layer
            out = layer_module(h)
            new_h = out[0]

            prev_h = h
            h = new_h
            budget_fraction = max(0.0, budget_fraction - (1.0 / self.total_layers))

        return collected_states, collected_actions

    def generate(
            self,
            tokenizer: GPT2TokenizerFast,
            prompt: str,
            max_new_tokens: int = 20,
            energy_budget: float = None,
            temperature: float = 0.7,  # Controls randomness (0.7 is good balance)
            top_k: int = 50,  # Only look at top 50 words
    ) -> str:
        if energy_budget is None:
            energy_budget = self.config.energy_budget

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

            # 1. Get Logits
            next_token_logits = out['logits'][:, -1, :]

            # 2. Apply Temperature (Higher = More random, Lower = More deterministic)
            next_token_logits = next_token_logits / temperature

            # 3. Top-K Filtering
            # Keep only the top K tokens, set others to -inf
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')

            # 4. Sample
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            input_seq = torch.cat([input_seq, next_token_id], dim=-1)
            past_key_values = out['past_key_values']

        return tokenizer.decode(input_seq[0], skip_special_tokens=True)