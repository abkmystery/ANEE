import os
import sys
import torch
import random
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.append(SRC)

from anee.config import ANEEConfig
from anee.wrapper import ANEEWrapper
from transformers import GPT2TokenizerFast


def collect_traces(model, tokenizer, prompts, save_path="datasets/traces.pt"):
    all_states = []
    all_actions = []

    model.eval()
    print(f"[ANEE] Generating SANDWICH traces...")

    for prompt in tqdm(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        budgets = [1.0, 0.7, 0.4, 0.1]

        for budget in budgets:
            states, _ = model.forward_collect_states(input_ids, energy_budget=budget)

            for i, state in enumerate(states):
                # State: [norm, delta, var, urgency, budget]
                urgency = state[3]
                layer_idx = i

                # --- TEACHER POLICY (Sandwich) ---
                action = 0  # Default Process

                # 1. Foundation (Layers 0, 1, 2) - KEEP
                if layer_idx < 3:
                    action = 0

                    # 2. Output Head (Layers 9, 10, 11) - KEEP
                # We need these to format the thought into English.
                elif layer_idx >= 9:
                    action = 0

                # 3. The "Swiss Cheese" Zone (Layers 3-8) - SKIP IF URGENT
                elif urgency < -0.05:  # If budget is even slightly tight
                    action = 1  # SKIP

                else:
                    action = 0  # PROCESS

                all_states.append(state)
                all_actions.append(action)

    states_tensor = torch.tensor(all_states, dtype=torch.float32)
    actions_tensor = torch.tensor(all_actions, dtype=torch.long)

    os.makedirs("datasets", exist_ok=True)
    torch.save({"states": states_tensor, "actions": actions_tensor}, save_path)
    print(f"[ANEE] Saved {len(states_tensor)} traces.")


def main():
    config = ANEEConfig(state_dim=5)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = ANEEWrapper(config)

    prompts = [
                  "The future of AI is", "The quick brown fox",
                  "Science is the study of", "To be or not to be",
                  "Machine learning models are", "Deep learning requires",
                  "Python is a programming language", "HuggingFace is awesome",
                  "The weather is nice", "I love coding"
              ] * 5

    collect_traces(model, tokenizer, prompts)


if __name__ == "__main__":
    main()