# import sys
# import os
# import torch
#
# # Add src to path
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
#
# from anee.wrapper import ANEEWrapper
# from anee.config import ANEEConfig
# from transformers import GPT2TokenizerFast
#
#
# def verbose_generate(model, tokenizer, prompt, budget=1.0, max_tokens=10):
#     print(f"\n--- GENERATION (Budget: {budget}) ---")
#     input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
#     generated = input_ids
#
#     total_skips = 0
#     total_exits = 0
#
#     for i in range(max_tokens):
#         # Run forward pass
#         with torch.no_grad():
#             out = model.forward(generated, energy_budget=budget)
#
#         # Extract stats
#         stats = out['stats']
#         logits = out['logits']
#
#         # Decide next token
#         next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
#         token_str = tokenizer.decode(next_token[0])
#         generated = torch.cat((generated, next_token), dim=1)
#
#         # LOGGING
#         action_str = "FULL RUN"
#         if stats['exited_early']:
#             action_str = f"EXIT @ {stats['layers_executed']}"
#             total_exits += 1
#         elif stats['skipped_layers'] > 0:
#             action_str = f"SKIPPED {stats['skipped_layers']} LAYERS"
#             total_skips += 1
#
#         print(f"Token {i + 1} [{token_str.strip():<10}] | Executed: {stats['layers_executed']:<2} | {action_str}")
#
#     print(f"\nTotal Skips: {total_skips} | Total Exits: {total_exits}")
#     return tokenizer.decode(generated[0], skip_special_tokens=True)
#
#
# def main():
#     # Load Learned Controller
#     config = ANEEConfig(
#         model_name="gpt2",
#         controller_type="learned",
#         controller_path="controllers/controller_rl.pt",
#         # controller_path="controllers/controller.pt",
#         state_dim=6
#     )
#
#     model = ANEEWrapper(config)
#     tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
#
#     prompt = "The future of artificial intelligence is"
#
#     # Run High Budget
#     verbose_generate(model, tokenizer, prompt, budget=1.0)
#
#     # Run Low Budget
#     verbose_generate(model, tokenizer, prompt, budget=0.05)
#
#
# if __name__ == "__main__":
#     main()

import os
import sys
import torch

# Add src/ to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_DIR)

from transformers import GPT2TokenizerFast
from anee.config import ANEEConfig
from anee.wrapper import ANEEWrapper
# IMPORT THE UTILS
from anee.utils import estimate_gpt2_flops, calculate_savings


def verbose_generate(model, tokenizer, prompt, budget=1.0, max_tokens=10):
    print(f"\n--- GENERATION (Budget: {budget}) ---")

    # 1. Get Baseline FLOPs (Static for GPT-2)
    # 1024 is standard context, but doesn't matter for per-token estimation
    flops_profile = estimate_gpt2_flops(None, seq_len=1)

    if hasattr(model.controller, "reset_rl_trace"):
        model.controller.reset_rl_trace()

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    generated = input_ids

    total_savings_pct = 0

    for i in range(max_tokens):
        # Run forward
        out = model.forward(generated, energy_budget=budget)

        stats = out['stats']
        logits = out['logits']

        # Calculate Savings for this specific token
        flops, pct = calculate_savings(
            flops_profile,
            executed_layers=stats['layers_executed'],
            total_layers=12
        )
        total_savings_pct += pct

        # Decode
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        token_str = tokenizer.decode(next_token[0])
        generated = torch.cat((generated, next_token), dim=1)

        action_str = "FULL RUN"
        if stats['exited_early']:
            action_str = f"EXIT @ {stats['layers_executed']}"
        elif stats['skipped_layers'] > 0:
            action_str = f"SKIP {stats['skipped_layers']}"

        print(
            f"Token {i + 1} [{token_str.strip():<10}] | Layers: {stats['layers_executed']:<2} | {action_str:<8} | Savings: {pct * 100:.1f}%")

    avg_savings = total_savings_pct / max_tokens
    print(f"\n>>> AVERAGE FLOPs SAVINGS: {avg_savings * 100:.1f}% <<<")
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main():
    config = ANEEConfig(
        model_name="gpt2",
        controller_type="learned",
        controller_path="controllers/controller_rl.pt",
        state_dim=6,  # Ensure this matches your current setup
    )

    model = ANEEWrapper(config)
    tokenizer = GPT2TokenizerFast.from_pretrained(config.model_name)
    prompt = "The future of artificial intelligence is"

    # Run with Low Budget to see the savings
    verbose_generate(model, tokenizer, prompt, budget=0.20)


if __name__ == "__main__":
    main()