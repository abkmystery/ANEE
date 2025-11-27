def estimate_gpt2_flops(config, seq_len):
    """
    Estimates FLOPs for a single forward pass of GPT-2 (Small).
    Formula derived from 'Scaling Laws for Neural Language Models' (Kaplan et al).

    GPT-2 Small:
    - Hidden (h): 768
    - Layers (L): 12
    - Heads: 12
    - Vocab (V): 50257
    - Context (T): seq_len

    Cost per token approx: 24 * h^2 * L
    """
    h = 768
    L = 12
    V = 50257

    # 1. Non-Embedding FLOPs per layer (Attn + MLP)
    # Attn: 4 * h^2 (projections) + 2 * T * h (attention score)
    # MLP: 8 * h^2
    # Total Block ~= 12 * h^2

    # We ignore the 'T' factor for attention in simple estimation as T << h usually,
    # but for counting:
    block_flops = 12 * (h ** 2)

    # 2. Logit Projection (Final Layer)
    head_flops = 2 * h * V

    return {
        "block_flops": block_flops,
        "head_flops": head_flops,
        "total_baseline": (block_flops * L) + head_flops
    }


def calculate_savings(ops_report, executed_layers, total_layers):
    """
    ops_report: dict from estimate_gpt2_flops
    """
    baseline = ops_report['total_baseline']

    # Cost = (Layers_Run * Block_Cost) + Head_Cost
    # Note: Even skipped layers cost ~5% for KV-proj, but let's assume pure skip for metric
    # To be precise: Partial skip is ~5% of block cost.

    # Let's say skipped layers cost 5% (Partial Skip overhead)
    skipped_layers = total_layers - executed_layers

    actual_flops = (executed_layers * ops_report['block_flops']) + \
                   (skipped_layers * 0.05 * ops_report['block_flops']) + \
                   ops_report['head_flops']

    savings_pct = 1.0 - (actual_flops / baseline)
    return actual_flops, savings_pct