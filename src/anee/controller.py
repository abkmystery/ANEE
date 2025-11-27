import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical



class HeuristicController:
    """
    Original Phase-1 controller.
    Uses only depth + budget to decide PROCESS vs EXIT.
    """

    def __init__(self, total_layers: int, min_layers: int = 2,
                 exit_budget_threshold: float = 0.1):
        self.total_layers = total_layers
        self.min_layers = min_layers
        self.exit_budget_threshold = exit_budget_threshold

    def decide(self,
               layer_idx: int,
               budget_fraction: float,
               state: Optional[torch.Tensor] = None):
        """
        Args:
            layer_idx: 0-based layer index
            budget_fraction: remaining budget in [0, 1]
            state: optional state tensor (ignored in heuristic mode)

        Returns:
            dict with 'action' and 'layer_cost'
        """
        layer_cost = 1.0 / self.total_layers

        if layer_idx < self.min_layers:
            return {"action": "PROCESS", "layer_cost": layer_cost}

        # if budget_fraction < self.exit_budget_threshold:
        #     return {"action": "EXIT", "layer_cost": 0.0}

        return {"action": "PROCESS", "layer_cost": layer_cost}


class LearnedController(nn.Module):
    """
    Tiny MLP controller:
      state (R^D) -> logits over actions {PROCESS, EXIT, SKIP}

    For now, the wrapper will only use PROCESS and EXIT actions.
    SKIP will be exploited in a later phase.
    """


    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        total_layers: int,
        min_layers: int = 2,
        exit_budget_threshold: float = 0.1,
    ):
        super().__init__()
        self.total_layers = total_layers
        self.min_layers = min_layers
        self.exit_budget_threshold = exit_budget_threshold
        self.action_space = ["PROCESS", "SKIP", "EXIT"]
        self.idx_to_action = {i: a for i, a in enumerate(self.action_space)}

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 0=PROCESS, 1=EXIT, 2=SKIP
        )

        # For RL
        self.saved_log_probs: list[torch.Tensor] = []

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (batch=1, state_dim)
        returns probs over actions: (1, 3)
        """
        logits = self.net(state)
        probs = F.softmax(logits, dim=-1)
        return probs

    def reset_rl_trace(self):
        """Call this at the start of an RL rollout."""
        self.saved_log_probs = []

    def decide(
        self,
        layer_idx: int,
        budget_fraction: float,
        state: torch.Tensor,
        track_logprob: bool = False,
    ):
        """
        Decide action from state + depth + budget.

        If track_logprob=True, we sample an action and store its log_prob
        for later RL policy gradient updates.
        Otherwise we use greedy argmax (for normal inference).
        """
        probs = self.forward(state)  # (1,3)
        action_idx: int

        if track_logprob:
            dist = Categorical(probs)
            sampled = dist.sample()  # (1,)
            log_prob = dist.log_prob(sampled)  # (1,)
            self.saved_log_probs.append(log_prob.squeeze(0))
            action_idx = int(sampled.item())
        else:
            # Greedy: use argmax
            action_idx = int(torch.argmax(probs, dim=-1).item())
            log_prob = None  # not used

        # idx2action = {0: "PROCESS", 1: "EXIT", 2: "SKIP"}
        # action = idx2action[action_idx]
        action = self.idx_to_action[action_idx]

        # Hard constraints: always process the first few layers
        if layer_idx < self.min_layers:
            action = "PROCESS"

        # If budget is very low, force EXIT regardless of policy
        # if budget_fraction < self.exit_budget_threshold and layer_idx >= self.min_layers:
        #     action = "EXIT"
        # exit_budget_threshold = 0.0
        #
        # # For now, treat SKIP as PROCESS (we'll activate true skipping later)
        # if action == "SKIP":
        #     action = "PROCESS"
        if action == "PROCESS":
            layer_cost = 1.0 / self.total_layers
        elif action == "SKIP":
            layer_cost = 0.05 * (1.0 / self.total_layers)  # 95% savings
        else:
            layer_cost = 0.0
        # layer_cost = 1.0 / self.total_layers if action == "PROCESS" else 0.0

        return {
            "action": action,
            "layer_cost": layer_cost,
            "log_prob": log_prob,
        }




def build_controller(config,
                     total_layers: int,
                     state_dim: int,
                     device: torch.device):
    """
    Factory to build the chosen controller (heuristic or learned).
    If a learned controller is created and a weights path exists, load it.
    """
    if config.controller_type == "heuristic":
        return HeuristicController(
            total_layers=total_layers,
            min_layers=config.min_layers,
            exit_budget_threshold=config.exit_budget_threshold,
        )

    if config.controller_type == "learned":
        ctrl = LearnedController(
            state_dim=state_dim,
            hidden_dim=config.controller_hidden_dim,
            total_layers=total_layers,
            min_layers=config.min_layers,
            exit_budget_threshold=config.exit_budget_threshold,
        ).to(device)

        if config.controller_path is not None and os.path.exists(config.controller_path):
            state_dict = torch.load(config.controller_path, map_location=device)
            ctrl.load_state_dict(state_dict)
            print(f"[ANEE] Loaded learned controller weights from {config.controller_path}")
        else:
            print("[ANEE] WARNING: Learned controller selected "
                  "but no weights found; using random initialization.")

        return ctrl

    raise ValueError(f"Unknown controller_type: {config.controller_type}")
