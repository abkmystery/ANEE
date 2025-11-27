import os
import sys

# Add src/ to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_DIR)

from transformers import GPT2TokenizerFast
from anee.config import ANEEConfig
from anee.wrapper import ANEEWrapper


def main():
    # 1. Setup config & model
    config = ANEEConfig(
        model_name="gpt2",
        energy_budget=1.0,
        min_layers=2,
        exit_budget_threshold=0.1,
    )
    model = ANEEWrapper(config)

    tokenizer = GPT2TokenizerFast.from_pretrained(config.model_name)

    prompt = "The future of artificial intelligence is"

    # 2. Full budget run
    print("\n=== RUN 1: Full Budget (1.0) ===")
    text_full = model.generate(
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=20,
        energy_budget=1.0,
    )
    print("Output:", text_full)

    # 3. Starved budget run
    print("\n=== RUN 2: Low Budget (0.05) ===")
    text_low = model.generate(
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=20,
        energy_budget=0.05,
    )
    print("Output:", text_low)


if __name__ == "__main__":
    main()
