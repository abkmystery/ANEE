# **ANEE: Adaptive Neural Execution Engine**

**Dynamic sparse inference for autoregressive Transformers**

ANEE is a lightweight research library that adds **per-token adaptive computation** to autoregressive Transformer models (GPT-2 + other GPT-style open models). It uses a profiler + learned controller to **skip redundant layers**, while keeping the **KV-cache aligned**, enabling coherent generation even when large portions of the network are bypassed.

ANEE has been tested on **GPT-2 small**, **GPT-2 medium**, **GPT-2 large**, and **GPT-2-XL**, achieving up to **50–55% theoretical FLOPs savings** on large models at low compute budgets.

---

## **Key Features**

### ✔ Dynamic Layer Skipping (Per Token)

ANEE decides, for every token, which layers are necessary and which can be skipped.

### ✔ Profiler-Driven State

Each layer is evaluated using:

* entropy
* hidden-state L2 norm
* delta-norm
* activation variance
* remaining compute budget
* depth position

These form the controller’s state vector.

### ✔ Safe Partial KV-Skipping

Skipped layers still update the KV-cache (keys/values only), keeping attention alignment intact while avoiding heavy matrix multiplications.

### ✔ RL or Heuristic Controller

Use either:

* a simple heuristic
* a learned controller trained via REINFORCE

### ✔ Plug-in Model Adapters (Extensible)

Current adapters:

* GPT-2 family (all sizes)

Ready for extension to:

* GPT-J
* LLaMA
* Falcon
* Mistral

via model adapters.

---

## **Install**

```bash
pip install anee
```

---

## **Quick Start**

```python
import torch
from transformers import GPT2TokenizerFast
from anee import ANEEConfig
from anee.wrapper import ANEEWrapper

config = ANEEConfig(model_name="gpt2-xl", energy_budget=0.2)
model = ANEEWrapper(config).eval()

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl")

text = model.generate(
    tokenizer=tokenizer,
    prompt="The future of AI",
    max_new_tokens=30,
)

print(text)
```

---

## **How ANEE Works**

For each token, ANEE:

* Profiles the hidden states using:
  entropy, max-softmax probability, L2 norm, delta-norm, variance, and remaining budget
* Builds a controller state vector
* Passes the state into an MLP controller to choose:

  * **PROCESS** — run the full layer
  * **SKIP** — update KV-cache only
  * **EXIT** — optional early stop
* Maintains safe KV-cache alignment
* Produces logits through the model’s final LN + LM head

---

## **FLOPs Savings Example (GPT-2-XL)**

```
Budget = 0.2
Layers executed per token: ~19–21 of 48
Layers skipped per token: 28–29
Average theoretical savings: 53–55%
```

The largest models show the strongest redundancy and highest savings.

---

## **Project Structure**

```
src/anee/
    wrapper.py            – core KV-safe executor
    controller.py         – heuristic + learned controller
    profiler.py           – entropy/norm/variance metrics
    utils.py              – FLOPs estimates, helpers
    reward.py             – RL reward functions
    config.py             – configuration dataclass

experiments/
    01_sanity_check.py    – simple text generation test
    visualize_heatmap.py  – layer-usage heatmaps
    train_controller.py   – supervised controller (optional)
    train_controller_rl.py – RL controller training
```

---

## **Supported Models**

| Model                 | Status            |
| --------------------- | ----------------- |
| **GPT-2 (all sizes)** | ✔ Full support    |
| **GPT-J 6B**          | ☐ Adapter planned |
| **LLaMA / Mistral**   | ☐ Adapter planned |
| **Falcon**            | ☐ Adapter planned |

Adapters can be added by implementing a `ModelAdapter` subclass.

---

## **Why ANEE?**

Transformers waste computation on many tokens.
ANEE reduces theoretical FLOPs while preserving sequence coherence by:

* identifying redundant layers
* skipping only semantic-middle layers
* preserving structure and output formatting layers

This creates a **“Sandwich Pattern”** of low–middle–low compute which appears consistently across models.

---

## **Citation**

If ANEE is useful in your work:

```bibtex
@software{anee2025,
  author = {Ahmed Bin Khalid},
  title  = {ANEE: Adaptive Neural Execution Engine},
  year   = {2025},
  doi    = {10.5281/zenodo.17741880},
  note   = {Dynamic sparse inference for autoregressive Transformers}
}
```

---

## **License**

**Apache 2.0**


