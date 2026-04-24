# SymCouple: Autonomous Symbolic Couplers for Cross-Architecture PEFT

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19725634.svg)](https://doi.org/10.5281/zenodo.19725634)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UkwTpCl2w2B87dbAyxg_0gyUfoAuwe7c)

**Train Once, Run Anywhere** – A novel parameter‑efficient fine‑tuning paradigm that decouples task‑specific learning from model‑specific routing through **autonomously discovered symbolic equations**.

---

## Overview

Modern large language models (LLMs) demand lightweight adaptation methods. Existing approaches like LoRA are **architecture‑locked**: an adapter trained for one hidden dimension cannot be transferred to another model without retraining. **SymCouple** overcomes this by introducing a two‑component adapter:

1. **Universal Adapter** – a task‑aware module that maps any input text to a fixed‑size *universal representation* **U** (dₖ = 256). It is trained **once** on the target task (e.g., sentiment analysis) and never modified thereafter.
2. **SR‑Coupler** – an ultra‑lightweight translation layer discovered via **Symbolic Regression**. For any base LLM, it learns an explicit, human‑readable mathematical formula that projects **U** into the model’s hidden‑state clusters, enabling instantaneous task steering without gradient updates.

This *“symbolic coupler”* reduces adaptation to a few minutes of formula discovery, requires only kilobytes of storage, and provides full **interpretability** – the coupling rules are readable algebraic expressions.

---

## How SymCouple Works

1. **Phase 1 – Train the Universal Adapter**  
   A small projection head (Linear → ReLU → Dropout) is placed atop a frozen Sentence‑BERT encoder and trained on a downstream dataset. It outputs a 256‑dimensional vector **U** that encodes task‑relevant information.

2. **Phase 2 – Collect (U, H) Pairs**  
   For a new target model, a calibration set is used to simultaneously obtain **U** (from the universal adapter) and the **cluster‑wise means** of the model’s last‑layer hidden states. A simple K‑Means grouping (e.g., 50 clusters) is applied to the hidden dimensions beforehand.

3. **Phase 3 – Discover the SR‑Coupler**  
   Symbolic regression (using `gplearn`) fits an arithmetic expression for each cluster:  
   `H_cluster_i = f_i(U)`  
   Typical formulas look like:  
   - `div(X114, sub(X124, X94))`  
   - `sin(X213)`  
   - `add(X103, mul(X7, -0.234))`

4. **Inference‑time Injection**  
   Given a new input, **U** is computed, the SR‑Coupler predicts new hidden cluster means, and the model’s hidden state is overwritten – the LLM is steered to reflect the task without any fine‑tuning.

---

## Cross‑Architecture Transfer Results

All experiments use **one universal sentiment adapter** trained on IMDB (5k samples) and **200 calibration pairs** per model. The SR‑Couplers are discovered with only 5 generations of symbolic regression.

| Model (Hidden Dim)        | Original Top‑5                      | Injected Top‑5                     | Example Coupler Formula |
|----------------------------|-------------------------------------|------------------------------------|--------------------------|
| **TinyLlama‑1.1B** (2048)  | `'It' 'I' 'The' 'This' '\n'`        | `'(' '' ',' 'the' 'in'`            | `X103`, `sin(X213)`      |
| **GPT‑2 small** (768)      | `' I' ' It' ' The' '\n' ' There'`    | `',' ' and' ' in' ' the' ' to'`    | `X129`, `X184`           |
| **Qwen2.5‑0.5B** (896)     | `' I' ' It' ' The' ' This' ' There'`  | `' the' ' ' ' in' ' a' ' and'`     | `div(X114, sub(X124, X94))` |

✅ **Zero retraining of the universal adapter** – only a few minutes of SR‑Coupler discovery per architecture.

---
---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/venkatramanakurumalla/SymCouple.git
cd SymCouple
pip install -r requirements.txt
@software{kurumalla2026symcouple,
  author       = {VenkataRamana Kurumalla},
  title        = {SymCouple: Autonomous Symbolic Couplers for Cross-Architecture PEFT},
  month        = apr,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19725634},
  url          = {https://doi.org/10.5281/zenodo.19725634}
}

