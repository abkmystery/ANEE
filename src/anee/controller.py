class ANEEController:
    """
    Phase-1 controller:
    - No randomness
    - Simple, depth + budget-based early exit
    - No MLP/attention/head/width logic yet (that’s Phase 2+)
    """
    def __init__(self, total_layers: int, min_layers: int = 2,
                 exit_budget_threshold: float = 0.1):
        self.total_layers = total_layers
        self.min_layers = min_layers
        self.exit_budget_threshold = exit_budget_threshold

    def decide(self, layer_idx: int, budget_fraction: float):
        """
        Decide what to do at this layer.

        Args:
            layer_idx: int, 0-based index of layer
            budget_fraction: remaining budget in [0, 1]

        Returns:
            dict with:
              - 'action': 'PROCESS' | 'EXIT'
              - 'layer_cost': estimated normalized cost of processing this layer
        """
        # Cost model (simple): each layer has equal cost
        # Normalized so that sum over all layers ≈ 1.0
        layer_cost = 1.0 / self.total_layers

        # Always process early layers to avoid trivially exiting too soon
        if layer_idx < self.min_layers:
            return {"action": "PROCESS", "layer_cost": layer_cost}

        # If budget is very low and we're past min_layers -> early exit
        if budget_fraction < self.exit_budget_threshold:
            return {"action": "EXIT", "layer_cost": 0.0}

        # Otherwise, process this layer as normal
        return {"action": "PROCESS", "layer_cost": layer_cost}
