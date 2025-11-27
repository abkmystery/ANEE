import torch
import torch.nn.functional as F


class ANEERewardEngine:
    def __init__(self, lambda_efficiency=2.0, lambda_penalty=10.0, lambda_compliance=5.0):
        self.lambda_eff = lambda_efficiency  # Reward for saving (if quality is good)
        self.lambda_pen = lambda_penalty  # Penalty for bad quality
        self.lambda_comp = lambda_compliance  # Penalty for ignoring budget

    def compute_reward(
            self,
            early_exit_logits: torch.Tensor,
            full_model_logits: torch.Tensor,
            layers_used: int,
            total_layers: int,
            target_budget: float,
    ):
        # 1. Calculate KL Divergence (The Quality Meter)
        student_log_probs = F.log_softmax(early_exit_logits[:, -1, :], dim=-1)
        teacher_probs = F.softmax(full_model_logits[:, -1, :], dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        kl_val = kl.item()

        # 2. Calculate Savings
        actual_cost = layers_used / float(total_layers)
        savings = 1.0 - actual_cost

        # 3. THE GATE (Your Idea)
        # Threshold: 0.1 is roughly "Very minimal difference"
        QUALITY_THRESHOLD = 0.5

        if kl_val > QUALITY_THRESHOLD:
            # ZONE A: GIBBERISH
            # Action: Heavy Punishment. Zero reward for efficiency.
            # We punish KL heavily to force it back to "Process All"
            reward = - (kl_val * self.lambda_pen)

            # Note: We do NOT penalize budget compliance here.
            # If the text is bad, priority #1 is fixing text, not fixing budget.
        else:
            # ZONE B: UNDERSTANDABLE
            # Action: Now we pay for Efficiency.
            reward = (savings * self.lambda_eff)

            # Apply Budget Compliance Penalty only if text is good
            over_budget = max(0.0, actual_cost - target_budget)
            reward -= (over_budget * self.lambda_comp)

        return torch.tensor(reward, device=early_exit_logits.device, requires_grad=True), {
            "kl": kl_val,
            "savings": savings,
            "zone": "GOOD" if kl_val <= QUALITY_THRESHOLD else "BAD"
        }