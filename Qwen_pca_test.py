
# =============================================================================
# SymCouple – PCA CONTINUOUS MANIFOLD RUN (The "Blur" Fix)
# =============================================================================

!pip install -q sentence-transformers transformers gplearn datasets scikit-learn

import torch
import torch.nn as nn
import numpy as np
import os
from google.colab import drive
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from gplearn.genetic import SymbolicRegressor
from datasets import load_dataset
import warnings

warnings.filterwarnings('ignore')
drive.mount('/content/drive', force_remount=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. UniversalAdapter class ---
class UniversalAdapter(nn.Module):
    def __init__(self, input_dim=384, univ_dim=256, dropout=0.2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, univ_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(univ_dim, 1)

    def forward(self, x):
        u = self.projection(x)
        logits = self.classifier(u)
        return u, logits

# --- 2. Load Models ---
print("🔄 Loading Models and Adapter...")
adapter_path = '/content/drive/MyDrive/universal_adapter_imdb.pt'
adapter = UniversalAdapter().to(device)
checkpoint = torch.load(adapter_path, map_location=device)
adapter.projection.load_state_dict(checkpoint['projection_state'])
adapter.classifier.load_state_dict(checkpoint['classifier_state'])
adapter.eval()

encoder = SentenceTransformer('all-MiniLM-L6-v2').to(device).eval()
new_model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(new_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(new_model_name).to(device)
print("✅ Models Loaded.")

# --- 3. Build H_calib & Train PCA ---
print("\n🔄 Generating Calibration Data for PCA...")
calib_prompts = [
    "The quick brown fox jumps over the lazy dog.",
    "Quantum mechanics explains particles at the smallest scales.",
    "Machine learning models learn patterns from data.",
    "The solar system contains eight planets and one star.",
    "Photosynthesis converts light into chemical energy.",
    "Albert Einstein developed the theory of relativity.",
    "The human brain processes information through neural networks.",
    "Water freezes at zero degrees Celsius and boils at one hundred.",
    "Gravity pulls objects toward the centre of the Earth.",
    "The Renaissance was a period of great cultural change."
]

all_hidden = []
for prompt in calib_prompts:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=20).to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    # Get all tokens to build a rich latent space map
    h = out.hidden_states[-1][0].float().cpu().numpy()
    all_hidden.append(h)

H_calib = np.concatenate(all_hidden, axis=0)  # Shape: (total_tokens, 896)

print("🔄 Training PCA (Compressing 896 dims -> 64 continuous components)...")
num_components = 64
pca = PCA(n_components=num_components, random_state=42)
pca.fit(H_calib)
print(f"✅ PCA Trained. Explained Variance: {sum(pca.explained_variance_ratio_)*100:.2f}%")

# --- 4. Collect Training Data (U_matrix -> PCA Space) ---
print("\n🔄 Collecting (U, H) pairs for PCA Mapping...")
dataset = load_dataset("imdb")
test_texts = dataset['test']['text'][:500] # 500 sentences for speed/accuracy balance

U_list = []
H_pca_list = []

for i, text in enumerate(test_texts):
    with torch.no_grad():
        emb = encoder.encode([text], convert_to_tensor=True, device=device)
        u, _ = adapter(emb)
    U_list.append(u.cpu().numpy().flatten())

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=50).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_token_hidden = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()

    # Compress target hidden state using our trained PCA
    h_pca = pca.transform(last_token_hidden.reshape(1, -1))[0]
    H_pca_list.append(h_pca)

    if (i+1) % 100 == 0:
        print(f"  processed {i+1}/{len(test_texts)}")

U_matrix = np.array(U_list)
H_matrix = np.array(H_pca_list)  # Shape: (500, 64)

# --- 5. Train SR on the 64 Continuous Components ---
scaler_U = StandardScaler()
scaler_H = StandardScaler()
U_scaled = scaler_U.fit_transform(U_matrix)
H_scaled = scaler_H.fit_transform(H_matrix)

print(f"\n🔄 Training {num_components} High-Fidelity SR-Couplers...")
sr_models = []
for c in range(num_components):
    y = H_scaled[:, c]
    sr = SymbolicRegressor(
        population_size=1000, generations=8,
        function_set=['add','sub','mul','div','sin','cos','log'],
        random_state=c, verbose=0,
        parsimony_coefficient=0.0001, const_range=(-1.0, 1.0)
    )
    sr.fit(U_scaled, y)
    sr_models.append(sr)
    if (c+1) % 16 == 0:
        print(f"  trained {c+1}/{num_components} math bridges...")
print("✅ All PCA SR-Couplers trained.")

# --- 6. The Ultimate Injection Test ---
test_sentence = "This movie is truly fantastic and wonderful."
print(f"\n--- INITIATING HIGH-FIDELITY INJECTION ---")
print(f"Test sentence: {test_sentence!r}")

with torch.no_grad():
    emb = encoder.encode([test_sentence], convert_to_tensor=True, device=device)
    u, _ = adapter(emb)
u_scaled = scaler_U.transform(u.cpu().numpy().reshape(1, -1))

# Predict the 64 components using Math
predicted_components_scaled = np.zeros(num_components)
for c in range(num_components):
    predicted_components_scaled[c] = sr_models[c].predict(u_scaled)[0]

# Unscale the predicted components
predicted_components = predicted_components_scaled * scaler_H.scale_ + scaler_H.mean_

# THE MAGIC: Invert the 64 components back out into 896 perfectly harmonized dimensions!
reconstructed_hidden = pca.inverse_transform(predicted_components.reshape(1, -1))[0]

# Inject into Qwen
inputs = tokenizer(test_sentence, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
hidden = outputs.hidden_states[-1][0]

modified_hidden = hidden.clone()
modified_hidden[-1] = torch.tensor(reconstructed_hidden, dtype=hidden.dtype, device=device)

# --- 7. Autoregressive Generation ---
with torch.no_grad():
    logits_mod = model.lm_head(modified_hidden[-1])
    forced_token_id = torch.argmax(logits_mod, dim=-1).unsqueeze(0)

forced_input_ids = torch.cat([inputs.input_ids, forced_token_id.unsqueeze(0)], dim=-1)

with torch.no_grad():
    gen_ids = model.generate(
        forced_input_ids, max_new_tokens=25, temperature=0.5,
        do_sample=True, repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id
    )

final_text = tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
injected_token = tokenizer.decode(forced_token_id)

print("\n" + "="*60)
print(f"Injected Math Token: {injected_token!r}")
print(f"Autoregressive Output: {final_text}")
print("="*60)
