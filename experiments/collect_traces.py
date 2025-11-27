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

    # 0 = PROCESS, 1 = SKIP, 2 = EXIT
    # We map our heuristic choices to these indices
    # We want to teach it:
    # - If Urgent < -0.1 -> SKIP (1)
    # - If Urgent > 0.1 -> PROCESS (0)
    # - If Budget < 0.05 -> EXIT (2)

    model.eval()

    print(f"[ANEE] Generating synthetic traces for Supervised Warmup...")

    for prompt in tqdm(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        # We simulate different budgets for the SAME prompt to teach adaptivity
        budgets = [1.0, 0.7, 0.4, 0.1]

        for budget in budgets:
            # We call the wrapper's collect method
            # Note: We need to modify wrapper to return what we need,
            # OR we just implement the heuristic logic here manually using the profiler.
            # Let's use the wrapper's existing forward_collect_states which we updated previously.

            states, heuristic_actions = model.forward_collect_states(input_ids, energy_budget=budget)

            # forward_collect_states currently only returned 0 or 1 (Exit).
            # We need to Upgrade the heuristic in wrapper OR patch it here.
            # Let's patch it here to be safe and simple.

            for i, state in enumerate(states):
                # State: [norm, delta, var, urgency, budget]
                urgency = state[3]
                curr_budget = state[4]
                layer_idx = i

                # DEFINING THE TEACHER POLICY
                action = 0  # Default Process

                # Rule 1: Always process first 2 layers
                if layer_idx < 2:
                    action = 0  # PROCESS

                # Rule 2: If we are very broke, Exit
                elif curr_budget < 0.05:
                    action = 2  # EXIT

                # Rule 3: If we are behind schedule (negative urgency), Skip
                elif urgency < -0.05:
                    action = 1  # SKIP

                # Rule 4: Otherwise Process
                else:
                    action = 0

                all_states.append(state)
                all_actions.append(action)

    # Convert to tensors
    states_tensor = torch.tensor(all_states, dtype=torch.float32)
    actions_tensor = torch.tensor(all_actions, dtype=torch.long)

    os.makedirs("datasets", exist_ok=True)
    torch.save({"states": states_tensor, "actions": actions_tensor}, save_path)
    print(f"[ANEE] Saved {len(states_tensor)} teacher traces to {save_path}")


def main():
    config = ANEEConfig(state_dim=5)  # Ensure dim matches
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = ANEEWrapper(config)

    prompts = [
                  "The future of AI is", "The quick brown fox",
                  "Science is the study of", "To be or not to be",
                  "Machine learning models are", "Deep learning requires",
                  "Python is a programming language", "HuggingFace is awesome",
                  "The weather is nice", "I love coding"
              ] * 5  # Duplicate to get more data

    collect_traces(model, tokenizer, prompts)


if __name__ == "__main__":
    main()