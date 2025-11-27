# ANEE v0.3 — Adaptive Neural Execution Engine

**Dynamic Sparse Inference for Pre-Trained Transformers**

ANEE is a lightweight framework for **token-wise, layer-wise adaptive computation** in transformer language models.
Instead of running every layer for every token, ANEE learns how to **allocate compute dynamically**, reducing unnecessary computation while preserving output quality.

ANEE wraps existing HuggingFace models (e.g., GPT-2) without modifying their weights.

---

## 🔧 Key Capabilities

### **• Dynamic Layer Skipping**

ANEE evaluates each transformer block at inference time and decides whether to:

* **PROCESS** — run full attention + MLP
* **SKIP** — bypass computation for that layer
* **EXIT** — terminate further processing (supported)

This produces **sparse execution patterns** that vary across tokens.

---

### **• RL-Trained Controller**

A small neural controller receives a per-layer state vector containing:

* entropy of logits
* hidden-state norms
* delta-norms
* variance
* layer position
* remaining budget

It learns policies via:

1. **Supervised warm-start** (from heuristic traces)
2. **Reinforcement learning** with a reward balancing:

   * similarity to full model (KL divergence)
   * compute savings
   * budget adherence

---

### **• Budget-Aware Inference**

Users provide an `energy_budget` in `[0,1]`.
The controller adjusts its behavior per token to meet the budget target while maintaining model output quality.

---

### **• Visual Execution Maps**

ANEE includes tooling to visualize:

* token-by-layer skip/process patterns
* per-token compute usage
* overall savings
* effective depth profiles

These “execution heatmaps” help interpret which layers the model relies on.

---

### **• Model-Agnostic Design**

The wrapper manually unrolls transformer layers and is structured for easy adaptation to other decoder-only architectures beyond GPT-2.

---

## 📦 Repository Structure

```
anee/
│
├── wrapper.py              # Core dynamic execution engine
├── controller.py           # Heuristic + learned controllers
├── profiler.py             # Layer-level state feature extractor
├── reward.py               # RL reward (quality + efficiency)
├── utils.py                # FLOPs proxy utilities
├── config.py               # ANEE configuration
│
├── experiments/
│   ├── train_controller.py
│   ├── train_controller_rl.py
│   ├── collect_traces.py
│   ├── 01_sanity_check.py
│   ├── visualize_heatmap.py
```

---

## 🚀 Getting Started

### Install

```bash
pip install -e .
```

### Warm-start Controller

```bash
python experiments/train_controller.py
```

### RL Fine-Tuning

```bash
python experiments/train_controller_rl.py
```

### Quick Test

```bash
python experiments/01_sanity_check.py
```

### Generate Heatmap Visualization

```bash
python experiments/visualize_heatmap.py
```

---

## 📈 Performance Snapshot (GPT-2 Small)

At moderate budgets, ANEE typically:

* executes ~6–9 of 12 layers per token
* achieves **~20–30% effective compute reduction**
* maintains coherent generation
* shows consistent “sparse middle, dense edges” execution profiles

Lower budgets naturally trade off output quality.

---

## 🔬 Intended Use & Applications

ANEE provides a clean, transparent platform for research in:

* dynamic depth / adaptive inference
* efficient transformer execution
* compute-aware LLM routing
* per-token sparsity patterns
* RL-driven execution policies

It is well-suited for experimentation, teaching, and further development.

---

## 📄 License

APACHE 2.0

---

## Citation

If you use ANEE in your research, please cite:

**Ahmed Bin Khalid. (2025). ANEE: Adaptive Neural Execution Engine. Zenodo.**  
DOI: https://doi.org/10.5281/zenodo.17741880

```bibtex
@software{anee,
  author       = {Ahmed Bin Khalid},
  title        = {ANEE: Adaptive Neural Execution Engine},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17741880},
  url          = {https://doi.org/10.5281/zenodo.17741880}
}

```

---

