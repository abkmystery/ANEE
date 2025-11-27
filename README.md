# ANEE: Adaptive Neural Execution Engine  
**Dynamic Compute Budgeting for Transformer Inference**

## Overview  
ANEE is a dynamic inference engine that wraps existing pretrained transformer models (e.g., GPT-2) and introduces *adaptive compute control* during generation. Instead of executing every transformer layer for every token, ANEE learns when to:

- **PROCESS** a layer fully  
- **SKIP** a layer with partial computation (cache-safe skipping)  
- **EXIT** early from the network  

This enables significant real-time FLOPs savings while maintaining coherent output quality.

ANEE does **not** retrain the base language model.  
All intelligence lies in the *controller*, which makes per-layer decisions using a compact state vector extracted during inference.

---

## Key Contributions  

### 🔥 1. **Dynamic Layer Skipping (Stable & Cache-Aligned)**  
ANEE implements a method for safe layer skipping in autoregressive transformers:

- Key/Value projections are computed even when skipping  
- Attention and MLP computation are avoided  
- Full KV-cache alignment is maintained  
- Downstream tokens remain valid, avoiding context corruption  

This makes skipping feasible in long-sequence generation, something most early-exit papers do not support.

---

### 🔥 2. **Learned Controller (Supervised + RL-Finetuned)**  
ANEE equips the Transformer stack with a lightweight decision module:

- Input: 7-dimensional state features derived from hidden dynamics  
- Output: categorical policy over `{PROCESS, SKIP, EXIT}`  
- Training:  
  - *Phase 1:* Supervised learning using heuristic teacher signals  
  - *Phase 2:* Policy-gradient reinforcement learning  
  - *Reward:* KL-based self-distillation + compute savings  

This controller learns a token-adaptive compute strategy.

---

### 🔥 3. **State Vector Profiling**  
Each layer produces a compact set of “signals”:

- Hidden entropy (token uncertainty)  
- Max probability  
- Hidden norm  
- Delta-norm change  
- Activation variance  
- Relative depth position  
- Remaining energy budget  

These features allow the controller to estimate complexity and decide where compute can be trimmed safely.

---

### 🔥 4. **Energy-Budgeted Inference**  
Users can specify an `energy_budget ∈ [0,1]`:

- `1.0` → process nearly all layers  
- `0.05` → skip aggressively  
- Intermediate values lead to hybrid paths  

This turns inference into a **resource-aware process**, not a fixed pass.

---

## Architecture Summary  

```

┌──────────────────────────────┐
│   GPT-2 Transformer Blocks    │
└──────────────────────────────┘
▲
│ per-layer signals
│
┌──────────────────────────────┐
│          Controller          │
│  (MLP + sampling or argmax)  │
└──────────────────────────────┘
▲       ▲         ▲
│       │         │
PROCESS   SKIP     EXIT

```

Skipping uses *partial execution*: K/V projections only, no attention or MLP – preserving future KV consistency.

---

## Features  

### ✔ Dynamic skipping  
Fast, stable skipping with KV alignment.

### ✔ Budget-controlled inference  
Predictable compute usage across tokens.

### ✔ Learned behavior  
Controller improves via supervised + RL phases.

### ✔ Fully manual stack unrolling  
Inference loop written explicitly in PyTorch, allowing injection of custom logic.

### ✔ Metrics & heatmaps  
Logs how often layers were processed vs skipped across tokens.

---

## Installation  

```

pip install -e .

```

Your environment must include:

```

torch
transformers
tqdm

````

---

## Quickstart Usage  

```python
from transformers import GPT2TokenizerFast
from anee.wrapper import ANEEWrapper
from anee.config import ANEEConfig

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
config = ANEEConfig(model_name="gpt2", energy_budget=0.3)

model = ANEEWrapper(config)

output = model.generate(
    tokenizer,
    prompt="The future of AI is",
    max_new_tokens=15,
)

print(output)
````

---

## Project Structure

```
anee/
  ├── wrapper.py           # Main ANEE engine
  ├── controller.py        # Heuristic + learned controllers
  ├── profiler.py          # Signal extraction (entropy, norms, variance)
  ├── reward.py            # RL reward engine
  ├── config.py            # Central config dataclass
  └── utils.py             # Misc utilities (FLOPs, helpers)

experiments/
  ├── collect_traces.py    # Builds supervised dataset
  ├── train_controller.py  # SL training
  └── train_controller_rl.py # RL finetuning
```

---

## Experiments

### Supervised Phase

Collect traces using a heuristic early-exit teacher.

```
python experiments/collect_traces.py
python experiments/train_controller.py
```

### Reinforcement Learning Phase

Finetune controller using KL-based rewards.

```
python experiments/train_controller_rl.py
```

---

## Current Outputs (Example)

```
Budget 1.0:
  ~10 layers executed, ~2 skipped per token
  Output remains coherent

Budget 0.05:
  ~4–7 layers executed per token
  Heavier skipping but intelligible result
```

---

## Research Value

ANEE contributes to both **Efficient Inference** and **Adaptive Computation** by:

* combining skipping + early exit
* operating in the *generative* setting (most papers don’t)
* introducing KV-consistent partial skipping
* using structured RL for dynamic compute allocation
* generalizing to any HF transformer without retraining

This positions ANEE within:

* Model Compression
* Dynamic Neural Networks
* Efficient LLM Inference
* Autoregressive Transformers
* Energy-Aware Systems

---

## Potential Use Cases

* **On-device LLM inference** (phones, edge devices)
* **Low-latency applications**
* **LLM serving systems** that must scale with compute limits
* **LLM cost-budget APIs**
* **Adaptive generation pipelines (games, assistive apps)**
* **Research on sparsity, Mixture-of-Compute, or early-exit**



---

## License

APACHE 2.0

---

## Contact

For questions or collaborations:
**[ahmed.khalid2108@gmail.com](mailto:ahmed.khalid2108@gmail.com)**

---

```
```
