
# =============================================================================
# SymCouple – THE HIGH-RESOLUTION MASTER RUN (Qwen2.5-0.5B)
# =============================================================================

!pip install -q sentence-transformers transformers gplearn datasets

import torch
import torch.nn as nn
import numpy as np
import pickle, os
from google.colab import drive
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from gplearn.genetic import SymbolicRegressor
from datasets import load_dataset
import warnings

warnings.filterwarnings('ignore')
drive.mount('/content/drive')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- UniversalAdapter class ---
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

# =============================================================================
# 1. IMDB Adapter
# =============================================================================
adapter_path = '/content/drive/MyDrive/universal_adapter_imdb.pt'
if not os.path.exists(adapter_path):
    raise FileNotFoundError("Adapter not found. Please train the IMDB adapter first.")
adapter = UniversalAdapter().to(device)
checkpoint = torch.load(adapter_path, map_location=device)
adapter.projection.load_state_dict(checkpoint['projection_state'])
adapter.classifier.load_state_dict(checkpoint['classifier_state'])
adapter.eval()
print("✅ IMDB Universal Adapter loaded.")

# =============================================================================
# 2. Sentence Encoder & Qwen2.5‑0.5B
# =============================================================================
encoder = SentenceTransformer('all-MiniLM-L6-v2').to(device).eval()
new_model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(new_model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(new_model_name).to(device)
hidden_size = model.config.hidden_size
print(f"✅ {new_model_name} loaded. Hidden size: {hidden_size}")

# =============================================================================
# 3. HIGH-RESOLUTION CLUSTERS (Upgraded to 150)
# =============================================================================
cluster_labels_path = '/content/drive/MyDrive/cluster_labels_qwen25_05B_HIGHRES.pkl'
num_clusters = 150 # 🚀 MASSIVE RESOLUTION UPGRADE

if os.path.exists(cluster_labels_path):
    with open(cluster_labels_path, 'rb') as f:
        cluster_labels = pickle.load(f)
    print("✅ Loaded HIGH-RES Qwen2.5 cluster labels.")
else:
    print("🔄 Creating HIGH-RES cluster labels (150 clusters)...")
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
        h = out.hidden_states[-1][0].float().cpu().numpy()
        all_hidden.append(h)
    H_calib = np.concatenate(all_hidden, axis=0)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(H_calib.T)
    with open(cluster_labels_path, 'wb') as f:
        pickle.dump(cluster_labels, f)
    print("✅ HIGH-RES cluster labels saved.")

# =============================================================================
# 4. HIGH-VOLUME DATA COLLECTION (Upgraded to 1000 sentences)
# =============================================================================
print("🔄 Collecting HIGH-VOLUME (U, H) pairs...")
dataset = load_dataset("imdb")
test_texts = dataset['test']['text'][:1000] # 🚀 5x MORE TRAINING DATA

U_list = []
H_means_list = []
for i, text in enumerate(test_texts):
    with torch.no_grad():
        emb = encoder.encode([text], convert_to_tensor=True, device=device)
        u, _ = adapter(emb)
    U_list.append(u.cpu().numpy().flatten())

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=50).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0]
    last_token = hidden[-1].float().cpu().numpy()

    cluster_means = np.zeros(num_clusters)
    for cl in range(num_clusters):
        mask = (cluster_labels == cl)
        # Handle empty clusters dynamically to prevent NaN errors in high-res mapping
        if np.any(mask):
            cluster_means[cl] = last_token[mask].mean()
        else:
            cluster_means[cl] = 0.0
    H_means_list.append(cluster_means)

    if (i+1) % 250 == 0:
        print(f"  processed {i+1}/{len(test_texts)}")

U_matrix = np.array(U_list)
H_matrix = np.array(H_means_list)
print(f"✅ Collected {U_matrix.shape[0]} pairs. U: {U_matrix.shape}, H: {H_matrix.shape}")

# =============================================================================
# 5. UNLEASHED SR‑COUPLER TRAINING
# =============================================================================
scaler_U = StandardScaler()
scaler_H = StandardScaler()
U_scaled = scaler_U.fit_transform(U_matrix)
H_scaled = scaler_H.fit_transform(H_matrix)

print(f"🔄 Training {num_clusters} High-Complexity SR-Couplers (This will take a few minutes)...")
sr_models = []
for cl in range(num_clusters):
    y = H_scaled[:, cl]
    # 🚀 UNLEASHED ALGORITHM: More population, more generations, less parsimony penalty
    sr = SymbolicRegressor(
        population_size=1000, generations=10,
        function_set=['add','sub','mul','div','sin','cos','log'],
        random_state=cl, verbose=0,
        parsimony_coefficient=0.0001, const_range=(-1.0, 1.0)
    )
    sr.fit(U_scaled, y)
    sr_models.append(sr)
    if (cl+1) % 25 == 0:
        print(f"  trained {cl+1}/{num_clusters} math bridges...")
print("✅ All High-Complexity SR-Couplers trained.")

# =============================================================================
# 6. THE FINAL INJECTION TEST
# =============================================================================
test_sentence = "This movie is truly fantastic and wonderful."
print(f"\nTest sentence: {test_sentence!r}")

with torch.no_grad():
    emb = encoder.encode([test_sentence], convert_to_tensor=True, device=device)
    u, _ = adapter(emb)
u_np = u.cpu().numpy().reshape(1, -1)
u_scaled = scaler_U.transform(u_np)

predicted_means = np.zeros(num_clusters)
for cl in range(num_clusters):
    pred_scaled = sr_models[cl].predict(u_scaled)[0]
    predicted_means[cl] = pred_scaled * scaler_H.scale_[cl] + scaler_H.mean_[cl]

inputs = tokenizer(test_sentence, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
hidden = outputs.hidden_states[-1][0]

modified_hidden = hidden.clone()
last_token = modified_hidden[-1]
for cl in range(num_clusters):
    dim_idx = np.where(cluster_labels == cl)[0]
    last_token[dim_idx] = predicted_means[cl]

with torch.no_grad():
    logits_orig = model.lm_head(hidden[-1])
    logits_mod  = model.lm_head(modified_hidden[-1])

top_orig = torch.topk(logits_orig, k=5).indices
top_mod  = torch.topk(logits_mod, k=5).indices

print("\nOriginal next token candidates (Qwen2.5):")
for idx in top_orig:
    print(f"  {tokenizer.decode([idx])!r}")
print("\nAfter injecting HIGH-RES SymCouple:")
for idx in top_mod:
    print(f"  {tokenizer.decode([idx])!r}")

print("\nExample Unleashed Math Formulas:")
for cl in [0, 10, 50, 100, 149]:
    print(f"  Cluster {cl}: {sr_models[cl]._program}")

print("\n✅ HIGH-RESOLUTION RUN COMPLETED!")
