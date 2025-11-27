import torch
import torch.nn.functional as F

class ANEERewardEngine:
    def __init__(self, lambda_efficiency=2.0, lambda_quality=1.0, lambda_compliance=5.0):
        self.lambda_eff = float(lambda_efficiency)
        self.lambda_qual = float(lambda_quality)
        self.lambda_comp = float(lambda_compliance) # New Penalty Weight

    def compute_reward(
        self,
        early_exit_logits: torch.Tensor,
        full_model_logits: torch.Tensor,
        layers_used: int,
        total_layers: int,
        target_budget: float, # New Argument
    ):
        # 1. Quality (KL Divergence) with CLIPPING
        student_log_probs = F.log_softmax(early_exit_logits[:, -1, :], dim=-1)
        teacher_probs = F.softmax(full_model_logits[:, -1, :], dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

        # INNOVATION: Tanh Clipping
        # Maps infinite loss to range [-1, 0] roughly
        # Or simple clamping. Let's use simple clamping to prevent gradients exploding.
        kl_clamped = torch.clamp(kl, min=0.0, max=5.0)

        quality_reward = -kl_clamped

        # 2. Efficiency (Raw Savings)
        actual_cost = layers_used / float(total_layers)
        savings = 1.0 - actual_cost
        efficiency_reward = torch.tensor(savings, device=early_exit_logits.device)

        # 3. Budget Compliance (The New Logic)
        # If we used MORE than the budget, punish heavily.
        # If we used LESS, that's fine (efficiency reward covers it).
        over_budget = max(0.0, actual_cost - target_budget)
        compliance_penalty = over_budget * self.lambda_comp

        # Total Reward
        total_reward = (
            (self.lambda_qual * quality_reward) +
            (self.lambda_eff * efficiency_reward) -
            compliance_penalty
        )

        return total_reward, {
            "qual": quality_reward.item(),
            "eff": efficiency_reward.item(),
            "comp": compliance_penalty
        }