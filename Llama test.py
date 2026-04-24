
import torch
import torch.nn as nn
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from gplearn.genetic import SymbolicRegressor
import warnings
warnings.filterwarnings('ignore')

# ===========================================================================
# 1. UniversalAdapter నిర్వచనం, పరికరం
# ===========================================================================
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
        u = self.projection(x)      # (batch, 256)
        logits = self.classifier(u) # (batch, 1)
        return u, logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===========================================================================
# 2. Universal Adapter, Sentence Transformer లోడ్
# ===========================================================================
encoder = SentenceTransformer('all-MiniLM-L6-v2').to(device).eval()
adapter = UniversalAdapter().to(device)
checkpoint = torch.load('universal_adapter.pt', map_location=device)
adapter.projection.load_state_dict(checkpoint['projection_state'])
adapter.classifier.load_state_dict(checkpoint['classifier_state'])
adapter.eval()

# ===========================================================================
# 3. TinyLlama, క్లస్టర్ లేబుల్స్, డేటా లోడ్
# ===========================================================================
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

with open('cluster_labels.pkl', 'rb') as f:
    cluster_labels = pickle.load(f)
num_clusters = 50

U_matrix = np.load('U_matrix.npy')   # (samples, 256)
H_matrix = np.load('H_matrix.npy')   # (samples, 50)

# ===========================================================================
# 4. స్కేలర్లు సిద్ధం చేయండి
# ===========================================================================
scaler_U = StandardScaler()
scaler_H = StandardScaler()
U_scaled = scaler_U.fit_transform(U_matrix)
H_scaled = scaler_H.fit_transform(H_matrix)

# ===========================================================================
# 5. 50 SR‑కప్లర్ల ట్రైనింగ్ (త్వరగా 5 తరాలతో)
# ===========================================================================
print("Training SR-Couplers for all 50 clusters...")
sr_models = []
for i in range(num_clusters):
    y = H_scaled[:, i]
    sr = SymbolicRegressor(
        population_size=500,
        generations=5,
        function_set=['add','sub','mul','div','sin','cos','log'],
        random_state=i,
        verbose=0,
        parsimony_coefficient=0.001,
        const_range=(-1.0, 1.0)
    )
    sr.fit(U_scaled, y)
    sr_models.append(sr)
    if i % 10 == 0:
        print(f"  cluster {i} done")
print("All SR-Couplers trained.\n")

# ===========================================================================
# 6. ఇంజెక్షన్ టెస్ట్
# ===========================================================================
test_sentence = "This movie is truly fantastic and wonderful."
print(f"Test sentence: {test_sentence!r}")

# 6a. Universal vector (U)
with torch.no_grad():
    emb = encoder.encode([test_sentence], convert_to_tensor=True, device=device)
    u, _ = adapter(emb)
u_np = u.cpu().numpy().reshape(1, -1)
u_scaled = scaler_U.transform(u_np)

# 6b. ప్రతి క్లస్టర్‌కు predicted mean (raw value)
predicted_means = np.zeros(num_clusters)
for i in range(num_clusters):
    pred_scaled = sr_models[i].predict(u_scaled)[0]
    # inverse transform to raw
    predicted_means[i] = pred_scaled * scaler_H.scale_[i] + scaler_H.mean_[i]

# 6c. TinyLlama hidden state (bfloat16 నేటివ్‌గా)
inputs = tokenizer(test_sentence, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
hidden = outputs.hidden_states[-1][0]          # bfloat16, (seq_len, 2048)

# 6d. క్లస్టర్‌లను ఓవర్‌రైట్
modified_hidden = hidden.clone()
last_token = modified_hidden[-1]              # (2048,)
for cluster_id in range(num_clusters):
    dim_indices = np.where(cluster_labels == cluster_id)[0]
    last_token[dim_indices] = predicted_means[cluster_id]   # float -> bfloat16

# 6e. తదుపరి పదాల పోలిక (indices ని సరిగా తీసుకోండి)
with torch.no_grad():
    logits_orig = model.lm_head(hidden[-1])
    logits_mod  = model.lm_head(modified_hidden[-1])

top_orig = torch.topk(logits_orig, k=5).indices      # shape (5,)
top_mod  = torch.topk(logits_mod, k=5).indices        # shape (5,)

print("\nOriginal next token candidates:")
for idx in top_orig:
    print(f"  {tokenizer.decode([idx])!r}")

print("\nAfter injecting via SymCouple:")
for idx in top_mod:
    print(f"  {tokenizer.decode([idx])!r}")

# 6f. కొన్ని ఫార్ములాలు చూపించండి
print("\nExample SR-Coupler formulas:")
for i in [0, 1, 2, 10, 30]:
    print(f"  Cluster {i}: {sr_models[i]._program}")
