# ANEE – Adaptive Neural Execution Engine (v0.0.1)

**ANEE** is an experimental *dynamic inference wrapper* for pretrained Transformer language models (currently GPT-2).  
Instead of always running all layers, ANEE exposes an **energy_budget** and performs **early exit** inside the model’s forward pass.

> This is **v0 (skeleton)** – a research prototype that:
> - Wraps HuggingFace GPT-2
> - Unrolls the Transformer blocks in Python
> - Applies a simple, budget-based early-exit policy
> - Demonstrates Inference-Time Energy Budgeting without retraining the base model

Later versions will add:
- Entropy-based and learned controllers
- Layer / head / MLP / width sparsity
- FLOPs-aware cost modeling and virtual accelerator simulation

---

## Features (v0)

- ✅ Wraps `GPT2LMHeadModel` from HuggingFace
- ✅ Manually unrolled Transformer layer loop
- ✅ `energy_budget ∈ [0, 1]` controlling how many layers run
- ✅ Early exit when budget runs out
- ✅ Simple greedy generation loop using the custom forward pass
- ✅ Installable as a Python package (`pip install -e .`)

---

## Installation

```bash
git clone https://github.com/<your-username>/ANEE.git
cd ANEE
pip install -r requirements.txt
pip install -e .
````

Requirements:

* Python 3.9+ recommended
* PyTorch
* `transformers` from HuggingFace

---

## Quickstart

```python
from anee.config import ANEEConfig
from anee.wrapper import ANEEWrapper
from transformers import GPT2TokenizerFast

config = ANEEConfig(
    model_name="gpt2",
    energy_budget=1.0,     # normalized [0, 1]
    min_layers=2,
    exit_budget_threshold=0.1,
)

model = ANEEWrapper(config)
tokenizer = GPT2TokenizerFast.from_pretrained(config.model_name)

prompt = "The future of artificial intelligence is"

# Full-ish budget (will usually exit near the last layer)
text_full = model.generate(
    tokenizer=tokenizer,
    prompt=prompt,
    max_new_tokens=20,
    energy_budget=1.0,
)
print("Full budget:", text_full)

# Starved budget (exits very early, degraded output)
text_low = model.generate(
    tokenizer=tokenizer,
    prompt=prompt,
    max_new_tokens=20,
    energy_budget=0.05,
)
print("Low budget:", text_low)
```

Or run the included experiment:

```bash
python experiments/01_sanity_check.py
```

---

## Current Limitations 

* Only GPT-2 is tested.
* Controller is a simple budget-based early-exit policy (no entropy / learned controller yet).
* KV-cache is not yet supported – we recompute full context per token for clarity.
* Early-exit uses the standard LM head on intermediate layer outputs, which may degrade quality at shallow depths.

These will be addressed in upcoming versions (v0.1+).

---

## License

Apache-2.0 
