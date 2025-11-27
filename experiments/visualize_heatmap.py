import sys
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from anee.wrapper import ANEEWrapper
from anee.config import ANEEConfig
from transformers import GPT2TokenizerFast


def collect_heatmap_data(model, tokenizer, prompt, budget):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    generated = input_ids

    layer_activity = []
    tokens = []

    # Setup Cache
    past_key_values = None

    print(f"\n[Visualizer] Generating text with Budget {budget}...")

    # Generate 15 tokens
    for i in range(15):

        # Setup inputs for this step
        device = model.device
        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(model.layers)
            model_input = generated
        else:
            # Robust Cache Check
            valid_cache = next((kv for kv in past_key_values if kv is not None), None)
            if valid_cache is not None:
                past_length = valid_cache[0].size(-2)
            else:
                past_length = 0

            model_input = generated[:, -1:]

        # Embeddings
        seq_len = model_input.size(1)
        position_ids = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_len)

        hidden_states = model.transformer.wte(model_input) + model.transformer.wpe(position_ids)
        hidden_states = model.transformer.drop(hidden_states)

        prev_hidden_states = hidden_states
        curr_budget = budget

        mask = []
        present_key_values = []

        # --- LAYER LOOP ---
        for layer_idx, layer_module in enumerate(model.layers):
            layer_past = past_key_values[layer_idx]

            # OBSERVE
            h_norm = math.log1p(model.profiler.hidden_norm(hidden_states))
            delta_norm = math.log1p(model.profiler.delta_hidden_norm(prev_hidden_states, hidden_states))
            var = math.log1p(model.profiler.variance(hidden_states))

            layers_left = model.total_layers - layer_idx
            cost = layers_left / float(model.total_layers)
            urgency = curr_budget - cost

            # --- FIX: ADD LAYER FRACTION (GPS) ---
            layer_fraction = layer_idx / float(model.total_layers)

            # State Vector: [Norm, Delta, Var, Urgency, Budget, Layer_Frac]
            state = torch.tensor([[h_norm, delta_norm, var, urgency, curr_budget, layer_fraction]],
                                 dtype=torch.float32, device=device)

            # DECIDE
            decision = model.controller.decide(layer_idx, curr_budget, state)
            action = decision['action']

            # --- SAFETY OVERRIDE (Match Wrapper Logic) ---
            if layer_idx >= (model.total_layers - 2):
                action = "PROCESS"
            # ---------------------------------------------

            if action == "PROCESS":
                mask.append(1)
                outputs = layer_module(hidden_states, layer_past=layer_past, use_cache=True)
                hidden_states = outputs[0]

                if len(outputs) > 1:
                    present_key_values.append(outputs[1])
                else:
                    present_key_values.append(None)

                prev_hidden_states = hidden_states
                curr_budget = max(0.0, curr_budget - decision['layer_cost'])

            elif action == "SKIP":
                mask.append(0)

                # Partial Skip
                attn = layer_module.attn
                head_dim = attn.head_dim
                query, key, value = attn.c_attn(hidden_states).split(attn.split_size, dim=2)

                # Use wrapper helper
                if hasattr(model, '_manual_split_heads'):
                    key = model._manual_split_heads(key, attn.num_heads, head_dim)
                    value = model._manual_split_heads(value, attn.num_heads, head_dim)
                else:
                    new_shape = key.size()[:-1] + (attn.num_heads, head_dim)
                    key = key.view(new_shape).permute(0, 2, 1, 3)
                    value = value.view(new_shape).permute(0, 2, 1, 3)

                if layer_past is not None:
                    key = torch.cat((layer_past[0], key), dim=-2)
                    value = torch.cat((layer_past[1], value), dim=-2)
                present_key_values.append((key, value))

                curr_budget = max(0.0, curr_budget - decision['layer_cost'])

            elif action == "EXIT":
                mask.append(0.5)
                mask.extend([0.5] * (model.total_layers - len(mask)))
                for _ in range(model.total_layers - len(present_key_values)):
                    present_key_values.append(None)
                break

        layer_activity.append(mask)
        past_key_values = present_key_values

        # Decode
        h_final = model.ln_f(hidden_states)
        logits = model.lm_head(h_final)
        next_token_logits = logits[:, -1, :]

        next_token_logits = next_token_logits / 0.8
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        token_str = tokenizer.decode(next_token[0])
        print(f"Token: {token_str.strip()} | Action: {mask}")
        tokens.append(token_str)
        generated = torch.cat((generated, next_token), dim=1)

    return np.array(layer_activity).T, tokens


def main():
    config = ANEEConfig(
        model_name="gpt2",
        controller_type="learned",
        controller_path="controllers/controller_rl.pt",
        state_dim=6  # UPDATE: Ensure this matches config/wrapper
    )
    model = ANEEWrapper(config)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    prompt = "The future of artificial intelligence is"

    print("Generating Heatmap Data...")
    activity, tokens = collect_heatmap_data(model, tokenizer, prompt, budget=0.20)

    # Plot
    plt.figure(figsize=(12, 6))
    sns.heatmap(activity, cmap="Blues", cbar=False, linewidths=0.5, linecolor='gray')

    display_tokens = tokens[:15]
    display_activity = activity[:, :15]

    plt.xticks(np.arange(len(display_tokens)) + 0.5, display_tokens, rotation=45, ha='right')
    plt.yticks(np.arange(12) + 0.5, [f"L{i}" for i in range(12)], rotation=0)
    plt.gca().invert_yaxis()
    plt.title(f"ANEE Brain Activity (Budget = 0.20)\nDark = Processed, Light = Skipped")

    plt.tight_layout()
    plt.savefig("anee_mri_scan.png")
    print("\nSaved MRI Scan to anee_mri_scan.png")


if __name__ == "__main__":
    main()