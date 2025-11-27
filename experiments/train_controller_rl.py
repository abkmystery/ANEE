import os
import sys
import random
import torch
import torch.optim as optim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.append(SRC)

from anee.config import ANEEConfig
from anee.wrapper import ANEEWrapper
from anee.reward import ANEERewardEngine
from transformers import GPT2TokenizerFast


def main():
    # 1. Load the "Sandwich" Weights (Don't start from scratch!)
    config = ANEEConfig(
        model_name="gpt2",
        controller_type="learned",
        controller_path="controllers/controller.pt",  # <--- LOAD SMART WEIGHTS
        energy_budget=1.0,
        state_dim=6
    )

    tokenizer = GPT2TokenizerFast.from_pretrained(config.model_name)
    model = ANEEWrapper(config)
    for p in model.base_model.parameters(): p.requires_grad = False

    # 2. Gentle Optimizer (Don't break the pre-training)
    optimizer = optim.Adam(model.controller.parameters(), lr=0.0005)

    # 3. Reward Engine
    # We use the Gated Logic.
    # Efficiency is 3.0 to encourage sticking to the skip plan.
    # Penalty is High to prevent unlearning English.
    reward_engine = ANEERewardEngine(
        lambda_efficiency=3.0,
        lambda_penalty=15.0,
        lambda_compliance=5.0
    )

    prompts = ["The future of AI", "Science is great", "To be or not", "Python code"]
    num_epochs = 20
    steps_per_epoch = 20
    running_baseline = None

    print("[ANEE-RL] Starting Fine-Tuning on Sandwich Policy...")

    for epoch in range(num_epochs):
        epoch_rewards = []
        zones = []
        for step in range(steps_per_epoch):
            prompt = random.choice(prompts)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

            if random.random() < 0.5:
                target_budget = 0.3
            else:
                target_budget = 0.9

            # Teacher
            with torch.no_grad():
                teacher_logits = model.base_model(input_ids).logits

            # Student
            rl_out = model.forward_rl(input_ids, energy_budget=target_budget)

            # Reward
            reward, stats = reward_engine.compute_reward(
                early_exit_logits=rl_out["logits"],
                full_model_logits=teacher_logits,
                layers_used=rl_out["stats"]["layers_executed"],
                total_layers=model.total_layers,
                target_budget=target_budget
            )
            zones.append(stats["zone"])

            # Update
            r_scalar = reward.item()
            if running_baseline is None:
                running_baseline = r_scalar
            else:
                running_baseline = 0.9 * running_baseline + 0.1 * r_scalar

            loss = -(reward - running_baseline) * rl_out["rl"]["log_probs"].sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_rewards.append(r_scalar)

            if step == 0:
                print(
                    f"  [Debug] Budget: {target_budget:.2f} | Executed: {rl_out['stats']['layers_executed']} | Zone: {stats['zone']}")

        good_pct = zones.count("GOOD") / len(zones) * 100
        print(
            f"Epoch {epoch + 1} | Avg Reward: {sum(epoch_rewards) / len(epoch_rewards):.2f} | Quality: {good_pct:.0f}%")

    os.makedirs("controllers", exist_ok=True)
    torch.save(model.controller.state_dict(), "controllers/controller_rl.pt")
    print("[ANEE-RL] Saved Tuned Controller.")


if __name__ == "__main__":
    main()