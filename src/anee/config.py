from dataclasses import dataclass
from typing import Optional

@dataclass
class ANEEConfig:
    model_name: str = "gpt2"
    energy_budget: float = 1.0
    min_layers: int = 2
    exit_budget_threshold: float = 0.1
    controller_type: str = "heuristic"

    # CHANGE HERE: 7 -> 5
    # Removed entropy(1) and max_prob(1).
    # Remaining: [h_norm, delta_norm, var, layer_frac, budget]
    state_dim: int = 5

    controller_hidden_dim: int = 32
    controller_path: Optional[str] = None