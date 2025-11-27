import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from anee.wrapper import ANEEWrapper
from anee.config import ANEEConfig
from transformers import GPT2TokenizerFast


def verbose_generate(model, tokenizer, prompt, budget=1.0, max_tokens=10):
    print(f"\n--- GENERATION (Budget: {budget}) ---")
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    generated = input_ids

    total_skips = 0
    total_exits = 0

    for i in range(max_tokens):
        # Run forward pass
        with torch.no_grad():
            out = model.forward(generated, energy_budget=budget)

        # Extract stats
        stats = out['stats']
        logits = out['logits']

        # Decide next token
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        token_str = tokenizer.decode(next_token[0])
        generated = torch.cat((generated, next_token), dim=1)

        # LOGGING
        action_str = "FULL RUN"
        if stats['exited_early']:
            action_str = f"EXIT @ {stats['layers_executed']}"
            total_exits += 1
        elif stats['skipped_layers'] > 0:
            action_str = f"SKIPPED {stats['skipped_layers']} LAYERS"
            total_skips += 1

        print(f"Token {i + 1} [{token_str.strip():<10}] | Executed: {stats['layers_executed']:<2} | {action_str}")

    print(f"\nTotal Skips: {total_skips} | Total Exits: {total_exits}")
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main():
    # Load Learned Controller
    config = ANEEConfig(
        model_name="gpt2",
        controller_type="learned",
        controller_path="controllers/controller_rl.pt",
        state_dim=5  # Ensure this matches your config update
    )

    model = ANEEWrapper(config)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    prompt = "The future of artificial intelligence is"

    # Run High Budget
    verbose_generate(model, tokenizer, prompt, budget=1.0)

    # Run Low Budget
    verbose_generate(model, tokenizer, prompt, budget=0.05)


if __name__ == "__main__":
    main()