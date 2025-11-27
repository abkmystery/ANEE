import os
import sys
import random
import torch
import torch.optim as optim
from torch.distributions import Categorical

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.append(SRC)

from anee.config import ANEEConfig
from anee.wrapper import ANEEWrapper
from anee.reward import ANEERewardEngine
from anee.controller import build_controller  # Ensure this is imported
from transformers import GPT2TokenizerFast


def main():
    # 1. Config: Start FRESH (Do not load old supervised weights if they are biased)
    # We set controller_path to None to force random init
    config = ANEEConfig(
        model_name="gpt2",
        controller_type="learned",
        controller_path="controllers/controller.pt",
        energy_budget=1.0,
        state_dim=5  # Match wrapper
    )

    tokenizer = GPT2TokenizerFast.from_pretrained(config.model_name)
    model = ANEEWrapper(config)

    # Freeze base model
    for p in model.base_model.parameters():
        p.requires_grad = False

    # 2. Aggressive Optimizer
    # Higher learning rate to learn fast from scratch
    optimizer = optim.Adam(model.controller.parameters(), lr=0.002)

    # 3. Aggressive Reward Engine
    # Efficiency is 5.0 (Huge reward for skipping)
    # Compliance is 5.0 (Huge penalty for ignoring budget)
    # Quality is 0.2 (We care less about grammar right now, just MECHANISM)
    reward_engine = ANEERewardEngine(
        lambda_efficiency=0.5,
        lambda_quality=1.0,
        lambda_compliance=5.0
    )

    prompts = [
        "The future of artificial intelligence is",
        "In a shocking discovery, scientists found",
        "The quick brown fox jumps over the",
        "To be or not to be, that is the",
        "The weather today is sunny with a chance of",
    ]

    num_epochs = 10  # More epochs to stabilize
    steps_per_epoch = 25

    running_baseline = None

    print("[ANEE-RL] Starting SHOCK THERAPY training (Fresh Weights)...")

    for epoch in range(num_epochs):
        epoch_rewards = []
        action_log_probs_collection = []

        for step in range(steps_per_epoch):
            prompt = random.choice(prompts)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

            # Force diverse budgets to teach adaptivity
            if random.random() < 0.5:
                target_budget = random.uniform(0.1, 0.4)  # Starvation mode
            else:
                target_budget = random.uniform(0.6, 0.9)  # Luxury mode

            # Teacher (Full Model)
            with torch.no_grad():
                teacher_out = model.base_model(input_ids)
                teacher_logits = teacher_out.logits

            # Student (ANEE)
            rl_out = model.forward_rl(input_ids, energy_budget=target_budget)
            student_logits = rl_out["logits"]
            layers_used = rl_out["stats"]["layers_executed"]

            # Get Log Probs for REINFORCE
            log_probs = rl_out["rl"]["log_probs"]

            # Calculate Entropy (Exploration Bonus)
            # We need access to the raw probs distribution, but we can approximate
            # or just rely on the fact that High Entropy = Less confident log_probs.
            # A simple trick: minimize the magnitude of log_probs (bring them closer to uniform)
            # But correct way is usually via the distribution.
            # For now, we rely on the aggressive Reward to force exploration.

            # Reward
            reward, comps = reward_engine.compute_reward(
                early_exit_logits=student_logits,
                full_model_logits=teacher_logits,
                layers_used=layers_used,
                total_layers=model.total_layers,
                target_budget=target_budget
            )

            # Baseline Update
            r_scalar = reward.item()
            if running_baseline is None:
                running_baseline = r_scalar
            else:
                running_baseline = 0.9 * running_baseline + 0.1 * r_scalar

            advantage = reward - running_baseline

            # Loss = -Policy_Gradient
            # We sum log_probs because they are independent decisions in the chain
            loss = -advantage * log_probs.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_rewards.append(r_scalar)

            # Logging actions to see if it's skipping
            if step == 0:
                print(f"  [Debug] Budget: {target_budget:.2f} | Executed: {layers_used} | Reward: {r_scalar:.2f}")

        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        print(f"Epoch {epoch + 1} | Avg Reward: {avg_reward:.4f}")

    # Save
    os.makedirs("controllers", exist_ok=True)
    torch.save(model.controller.state_dict(), "controllers/controller_rl.pt")
    print("[ANEE-RL] Saved fresh controller.")


if __name__ == "__main__":
    main()