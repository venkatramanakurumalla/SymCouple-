
# =============================================================================
# SymCouple – పూర్తి పైప్‌లైన్ (సరిచేసిన వెర్షన్)
# =============================================================================

# --- ప్యాకేజీలు ---
!pip install -q sentence-transformers transformers gplearn datasets

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle, os, re
from google.colab import drive
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from gplearn.genetic import SymbolicRegressor
from datasets import load_dataset

# =============================================================================
# 1. Google Drive మౌంట్
# =============================================================================
drive.mount('/content/drive')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 2. UniversalAdapter నిర్వచనం
# =============================================================================
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
# 3. IMDB Adapter లోడ్ లేదా ట్రైనింగ్
# =============================================================================
adapter_path = '/content/drive/MyDrive/universal_adapter_imdb.pt'

if os.path.exists(adapter_path):
    print("✅ IMDB adapter found, loading...")
    adapter = UniversalAdapter().to(device)
    checkpoint = torch.load(adapter_path, map_location=device)
    adapter.projection.load_state_dict(checkpoint['projection_state'])
    adapter.classifier.load_state_dict(checkpoint['classifier_state'])
    adapter.eval()
else:
    print("⚠️ IMDB adapter not found. Training on a small IMDB subset (5k samples)...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2').to(device).eval()

    dataset = load_dataset("imdb")
    train_texts = dataset['train']['text'][:5000]
    train_labels = dataset['train']['label'][:5000]
    val_texts = dataset['test']['text'][:1000]
    val_labels = dataset['test']['label'][:1000]

    def encode_batch(texts, batch_size=64):
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            with torch.no_grad():
                emb = encoder.encode(batch, convert_to_tensor=True, device=device)
                all_emb.append(emb)
        return torch.cat(all_emb, dim=0)

    train_emb = encode_batch(train_texts)
    val_emb   = encode_batch(val_texts)
    y_train = torch.tensor(train_labels, dtype=torch.float32).to(device)
    y_val   = torch.tensor(val_labels, dtype=torch.float32).to(device)

    adapter = UniversalAdapter().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(adapter.parameters(), lr=1e-3)
    epochs = 5
    batch_size = 64

    for epoch in range(epochs):
        adapter.train()
        perm = torch.randperm(len(train_emb))
        total_loss = 0.0
        for i in range(0, len(train_emb), batch_size):
            idx = perm[i:i+batch_size]
            emb = train_emb[idx]
            labels = y_train[idx]
            _, logits = adapter(emb)
            loss = criterion(logits.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Validation
        adapter.eval()
        with torch.no_grad():
            _, val_logits = adapter(val_emb)
            val_preds = (torch.sigmoid(val_logits.squeeze()) > 0.5).long()
            val_acc = accuracy_score(y_val.cpu().numpy(), val_preds.cpu().numpy())
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_emb):.4f} | Val Acc: {val_acc:.4f}")

    torch.save({
        'projection_state': adapter.projection.state_dict(),
        'classifier_state': adapter.classifier.state_dict(),
    }, adapter_path)
    print(f"✅ Adapter trained and saved to {adapter_path}")

# =============================================================================
# 4. ఎంబెడ్డర్ & TinyLlama లోడ్
# =============================================================================
print("🔄 Loading Sentence Encoder & TinyLlama...")
encoder = SentenceTransformer('all-MiniLM-L6-v2').to(device).eval()
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# =============================================================================
# 5. క్లస్టర్ లేబుల్స్ (సరిచేసిన భాగం: .to(device))
# =============================================================================
cluster_labels_path = '/content/drive/MyDrive/cluster_labels.pkl'
num_clusters = 50

if os.path.exists(cluster_labels_path):
    with open(cluster_labels_path, 'rb') as f:
        cluster_labels = pickle.load(f)
    print("✅ Loaded existing cluster_labels from Drive.")
else:
    print("🔄 Creating new cluster_labels...")
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
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=20).to(device)  # ✅ ఇక్కడ .to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        h = out.hidden_states[-1][0].float().cpu().numpy()  # numpy కోసం cpu కి తీసుకోండి
        all_hidden.append(h)
    H_calib = np.concatenate(all_hidden, axis=0)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(H_calib.T)
    with open(cluster_labels_path, 'wb') as f:
        pickle.dump(cluster_labels, f)
    print("✅ New cluster_labels saved.")

# =============================================================================
# 6. (U,H) జతల సేకరణ
# =============================================================================
print("🔄 Collecting (U,H) pairs...")
dataset = load_dataset("imdb")
test_texts = dataset['test']['text'][:200]

U_list = []
H_means_list = []
for i, text in enumerate(test_texts):
    # Universal vector
    with torch.no_grad():
        emb = encoder.encode([text], convert_to_tensor=True, device=device)
        u, _ = adapter(emb)
    U_list.append(u.cpu().numpy().flatten())

    # TinyLlama hidden state
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=50).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0]          # bfloat16
    last_token = hidden[-1].float().cpu().numpy()
    cluster_means = np.zeros(num_clusters)
    for cl in range(num_clusters):
        mask = (cluster_labels == cl)
        cluster_means[cl] = last_token[mask].mean()
    H_means_list.append(cluster_means)

    if (i+1) % 50 == 0:
        print(f"  processed {i+1}/{len(test_texts)}")

U_matrix = np.array(U_list)
H_matrix = np.array(H_means_list)
print(f"✅ Collected {U_matrix.shape[0]} pairs. U: {U_matrix.shape}, H: {H_matrix.shape}")

# =============================================================================
# 7. SR‑Coupler ట్రైనింగ్
# =============================================================================
scaler_U = StandardScaler()
scaler_H = StandardScaler()
U_scaled = scaler_U.fit_transform(U_matrix)
H_scaled = scaler_H.fit_transform(H_matrix)

print("🔄 Training SR-Couplers for all 50 clusters...")
sr_models = []
for cl in range(num_clusters):
    y = H_scaled[:, cl]
    sr = SymbolicRegressor(
        population_size=500, generations=5,
        function_set=['add','sub','mul','div','sin','cos','log'],
        random_state=cl, verbose=0,
        parsimony_coefficient=0.001, const_range=(-1.0, 1.0)
    )
    sr.fit(U_scaled, y)
    sr_models.append(sr)
print("✅ All SR-Couplers trained.")

# =============================================================================
# 8. ఇంజెక్షన్
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
hidden = outputs.hidden_states[-1][0]       # bfloat16

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

print("\nOriginal next token candidates:")
for idx in top_orig:
    print(f"  {tokenizer.decode([idx])!r}")
print("\nAfter injecting via SymCouple (IMDB adapter):")
for idx in top_mod:
    print(f"  {tokenizer.decode([idx])!r}")

# కొన్ని ఫార్ములాలు
print("\nExample SR-Coupler formulas:")
for cl in [0, 1, 2, 10, 30]:
    print(f"  Cluster {cl}: {sr_models[cl]._program}")

print("\n✅ SymCouple pipeline completed successfully!")
