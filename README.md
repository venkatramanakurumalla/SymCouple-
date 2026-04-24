# SymCouple: Autonomous Symbolic Couplers for Cross-Architecture PEFT

[![DOI](https://zenodo.org/badge/DOI/YOUR-ZENODO-DOI.svg)](https://doi.org/YOUR-ZENODO-DOI)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Train Once, Run Anywhere** – adapt a single task-aware Universal Adapter to any language model using human-readable symbolic equations.

## 🧠 How It Works
1. **Phase 1:** Train a Universal Adapter (e.g., for sentiment analysis) that compresses text into a fixed 256‑dim vector **U**.
2. **Phase 2:** For a new target LLM, collect pairs of **(U, cluster‑means of its hidden state)**.
3. **Phase 3:** Discover an ultra‑lightweight **SR‑Coupler** (`f(U) = cluster_mean`) via Symbolic Regression.  
   *Example: `div(X114, sub(X124, X94))`*
4. **Injection:** During inference, use the coupler to overwrite the model’s hidden clusters — steering outputs without retraining.

## 📊 Cross‑Architecture Transfer (Same Sentiment Adapter)

| Model | Hidden Dim | Original Top‑5 | Injected Top‑5 | Example Formula |
|-------|-----------|-----------------|----------------|-----------------|
| TinyLlama‑1.1B | 2048 | `'It' 'I' 'The' 'This' '\n'` | `'(' '' ',' 'the' 'in'` | `X103, sin(X213)` |
| GPT‑2 (small) | 768 | `' I' ' It' ' The' '\n' ' There'` | `',' ' and' ' in' ' the' ' to'` | `X129, X184` |
| Qwen2.5‑0.5B | 896 | `' I' ' It' ' The' ' This' ' There'` | `' the' ' ' ' in' ' a' ' and'` | `div(X114, sub(X124, X94))` |

✅ **No retraining of the adapter for new models** – only a few minutes of SR‑Coupler discovery.

## 📂 Repo Structure
- `universal_adapter/` – Training & saving the IMDB sentiment adapter.
- `sr_coupler/` – Collecting (U,H) pairs and symbolic regression.
- `injection/` – Steering any model with discovered formulas.
- `notebooks/` – Ready‑to‑run Colab demonstrations for each model.

## 🚀 Quick Start
```bash
git clone https://github.com/your-username/SymCouple.git
cd SymCouple
pip install -r requirements.txt
python universal_adapter/train_adapter.py   # if you need to train adapter
python sr_coupler/collect_pairs.py --model gpt2
python sr_coupler/train_couplers.py
python injection/inject.py --model gpt2
