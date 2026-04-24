# SymCouple: Autonomous Symbolic Couplers for Cross-Architecture PEFT

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19725634.svg)](https://doi.org/10.5281/zenodo.19725634)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/SymCouple/blob/main/notebooks/demo_all_models.ipynb)

**Train Once, Run Anywhere** – Adapt a single task‑specific Universal Adapter to any language model using **human‑readable symbolic equations**.

SymCouple introduces a novel **Parameter‑Efficient Fine‑Tuning (PEFT)** paradigm that decouples task learning from model‑specific routing. Instead of retraining adapters for each model, we use **Symbolic Regression** to discover ultra‑lightweight mathematical formulas (SR‑Couplers) that translate a fixed‑size universal representation into any model’s hidden space – enabling **zero‑shot cross‑architecture transfer** with interpretability and minimal storage.

---

## 🧠 How It Works

1. **Phase 1 – Universal Adapter**  
   A small neural module (e.g., a linear projection over a frozen Sentence‑BERT encoder) is trained once on a downstream task (sentiment analysis) and compresses any input text into a 256‑dimensional *universal vector* **U**.

2. **Phase 2 – Data Collection**  
   For a new target LLM (TinyLlama, GPT‑2, Qwen2.5, …), we collect pairs of **(U**, cluster‑means of the model’s last‑layer hidden state) using a small calibration set.

3. **Phase 3 – SR‑Coupler Discovery**  
   An off‑the‑shelf symbolic regression engine (gplearn) fits an explicit mathematical formula for each functional cluster:  
   `H_cluster_i = f_i(U)`  
   *Examples: `div(X114, sub(X124, X94))`, `sin(X213)`, `X129`*

4. **Inference‑time Injection**  
   During generation, the universal vector is computed for the input text, the SR‑Coupler predicts target cluster values, and the model’s hidden state is overwritten accordingly – **steering the output without any gradient steps**.

---

## 📊 Cross‑Architecture Transfer (Same Sentiment Adapter)

All experiments use **one universal sentiment adapter** trained on IMDB (5k samples).  
The SR‑Couplers are fitted with only **200 text pairs** and **5 generations** of symbolic regression.

| Model (hidden size)        | Original Top‑5                       | Injected Top‑5                  | Example Formula                      |
|----------------------------|--------------------------------------|---------------------------------|--------------------------------------|
| **TinyLlama‑1.1B** (2048)  | `'It' 'I' 'The' 'This' '\n'`        | `'(' '' ',' 'the' 'in'`         | `X103`, `sin(X213)`                  |
| **GPT‑2 small** (768)      | `' I' ' It' ' The' '\n' ' There'`    | `',' ' and' ' in' ' the' ' to'` | `X129`, `X184`                       |
| **Qwen2.5‑0.5B** (896)     | `' I' ' It' ' The' ' This' ' There'` | `' the' ' ' ' in' ' a' ' and'`  | `div(X114, sub(X124, X94))`          |

✅ **No retraining of the adapter** – only a few minutes of SR‑Coupler discovery per model.

---

