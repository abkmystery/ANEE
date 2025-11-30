def estimate_gpt2_flops(model_config, seq_len):
    """
    Dynamic FLOPs estimator based on the actual model configuration.
    """
    # Fetch dimensions from config
    h = model_config.n_embd
    L = model_config.n_layer
    V = model_config.vocab_size

    # 1. Non-Embedding FLOPs per layer (Attn + MLP)
    # Attn: 4*h^2 + ...
    # MLP: 8*h^2
    # Approx Block Cost: 24 * h^2 (Kaplan et al.) - Reduced to 12*h^2 for FWD pass only estimate
    block_flops = 12 * (h ** 2)

    # 2. Logit Projection
    head_flops = 2 * h * V

    return {
        "block_flops": block_flops,
        "head_flops": head_flops,
        "total_baseline": (block_flops * L) + head_flops
    }


def calculate_savings(ops_report, executed_layers, total_layers):
    baseline = ops_report['total_baseline']

    # Skipped layers cost ~5% (KV Projection overhead)
    skipped_layers = total_layers - executed_layers

    actual_flops = (executed_layers * ops_report['block_flops']) + \
                   (skipped_layers * 0.05 * ops_report['block_flops']) + \
                   ops_report['head_flops']

    savings_pct = 1.0 - (actual_flops / baseline)
    return actual_flops, savings_pct