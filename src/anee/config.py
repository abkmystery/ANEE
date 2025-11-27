from dataclasses import dataclass

@dataclass
class ANEEConfig:
    model_name: str = "gpt2"
    # energy_budget is a normalized scalar in [0, 1]
    energy_budget: float = 1.0

    # Minimum number of layers we ALWAYS run (no skipping/exit before this)
    min_layers: int = 2

    # If budget fraction drops below this, we allow early exit
    exit_budget_threshold: float = 0.1
