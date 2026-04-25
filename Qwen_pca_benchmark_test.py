
# =============================================================================
# THE ULTIMATE PCA SYMCOUPLE BENCHMARK SUITE
# Run this after the PCA pipeline finishes to generate research paper data.
# =============================================================================
import torch
import torch.nn.functional as F
import numpy as np
from textblob import TextBlob

print("--- INITIATING PCA BENCHMARK SUITE ---")

# ---------------------------------------------------------
# TEST 1: IN-DISTRIBUTION SEMANTIC RETENTION
# ---------------------------------------------------------
print("\n[TEST 1] IN-DISTRIBUTION ACCURACY (IMDB Sentiment N=40)")

test_prompts = [
    # Positive Anchors
    "This movie is an absolute masterpiece.", "I loved every single second of it.",
    "Truly a fantastic and wonderful film.", "Brilliant acting and amazing plot.",
    "One of the best movies I have ever seen.", "A breathtaking cinematic experience.",
    "Highly recommended, ten out of ten.", "The cinematography was beautiful.",
    "An inspiring and uplifting story.", "I left the theater completely amazed.",
    "Flawless execution from the director.", "The best sci-fi movie of the decade.",
    "A heartwarming and joyful ride.", "Spectacular visuals and great sound.",
    "I would gladly watch this again.", "An unforgettable masterpiece.",
    "The character development is superb.", "A triumphant and glorious achievement.",
    "Pure perfection on the big screen.", "It exceeded all my expectations."
] + [
    # Negative Anchors
    "This movie is a complete disaster.", "I hated every single second of it.",
    "Truly a terrible and boring film.", "Awful acting and a stupid plot.",
    "One of the worst movies I have ever seen.", "A miserable and painful experience.",
    "Highly avoid, zero out of ten.", "The cinematography was incredibly ugly.",
    "A depressing and pointless story.", "I left the theater feeling ripped off.",
    "Flawed execution from the director.", "The worst sci-fi movie of the decade.",
    "A soulless and agonizing ride.", "Terrible visuals and bad sound.",
    "I would never watch this again.", "An absolutely forgettable trash fire.",
    "The character development is nonexistent.", "A total failure and a waste of time.",
    "Pure garbage on the big screen.", "It ruined all my expectations."
]

expected_sentiments = [1] * 20 + [-1] * 20
correct_directions = 0

for idx, text in enumerate(test_prompts):
    # 1. Encode & Map via PCA Math
    with torch.no_grad():
        emb = encoder.encode([text], convert_to_tensor=True, device=device)
        u, _ = adapter(emb)
    u_scaled = scaler_U.transform(u.cpu().numpy().reshape(1, -1))

    predicted_comps_scaled = np.zeros(num_components)
    for c in range(num_components):
        predicted_comps_scaled[c] = sr_models[c].predict(u_scaled)[0]

    predicted_comps = predicted_comps_scaled * scaler_H.scale_ + scaler_H.mean_
    reconstructed_hidden = pca.inverse_transform(predicted_comps.reshape(1, -1))[0]

    # 2. Inject into Qwen
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0]
    modified_hidden = hidden.clone()
    modified_hidden[-1] = torch.tensor(reconstructed_hidden, dtype=hidden.dtype, device=device)

    # 3. Generate
    with torch.no_grad():
        logits_mod = model.lm_head(modified_hidden[-1])
        forced_token_id = torch.argmax(logits_mod, dim=-1).unsqueeze(0)

    forced_input_ids = torch.cat([inputs.input_ids, forced_token_id.unsqueeze(0)], dim=-1)

    with torch.no_grad():
        gen_ids = model.generate(
            forced_input_ids, max_new_tokens=15, temperature=0.4,
            do_sample=True, repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id
        )
    gen_text = tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # 4. Score
    blob = TextBlob(gen_text)
    gen_polarity = blob.sentiment.polarity
    if (expected_sentiments[idx] == 1 and gen_polarity > 0.05) or (expected_sentiments[idx] == -1 and gen_polarity < -0.05):
        correct_directions += 1

accuracy = (correct_directions / len(test_prompts)) * 100

print(f"PCA SYMCUPLE SEMANTIC ACCURACY: {accuracy:.1f}%")
print(f"(Baseline to beat: 65.0%)")

# ---------------------------------------------------------
# TEST 2: OUT-OF-DISTRIBUTION (OOD) DEEP SPACE
# ---------------------------------------------------------
print("\n[TEST 2] OUT-OF-DISTRIBUTION LATENT STABILITY")

ood_prompts = [
    "The core principle of quantum entanglement is",
    "During the Renaissance, the most famous artists",
    "The mitochondria is known as the powerhouse of the",
    "In object-oriented programming, inheritance allows",
    "The gravitational pull of a black hole is so strong that"
]

confidence_scores = []

for idx, text in enumerate(ood_prompts):
    with torch.no_grad():
        emb = encoder.encode([text], convert_to_tensor=True, device=device)
        u, _ = adapter(emb)
    u_scaled = scaler_U.transform(u.cpu().numpy().reshape(1, -1))

    predicted_comps_scaled = np.zeros(num_components)
    for c in range(num_components):
        predicted_comps_scaled[c] = sr_models[c].predict(u_scaled)[0]

    predicted_comps = predicted_comps_scaled * scaler_H.scale_ + scaler_H.mean_
    reconstructed_hidden = pca.inverse_transform(predicted_comps.reshape(1, -1))[0]

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0]
    modified_hidden = hidden.clone()
    modified_hidden[-1] = torch.tensor(reconstructed_hidden, dtype=hidden.dtype, device=device)

    with torch.no_grad():
        logits_mod = model.lm_head(modified_hidden[-1])
        probabilities = F.softmax(logits_mod, dim=-1)
        top_prob, top_id = torch.max(probabilities, dim=-1)
        forced_token_id = top_id.unsqueeze(0)
        confidence = top_prob.item() * 100
        confidence_scores.append(confidence)

    forced_input_ids = torch.cat([inputs.input_ids, forced_token_id.unsqueeze(0)], dim=-1)

    with torch.no_grad():
        gen_ids = model.generate(
            forced_input_ids, max_new_tokens=15, temperature=0.3,
            do_sample=True, repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id
        )

    gen_text = tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    injected_token = tokenizer.decode(forced_token_id)

    print(f"\nOOD Prompt: {text}")
    print(f"Injected Math Token: {injected_token!r} (Confidence: {confidence:.2f}%)")
    print(f"Generated Continuation: {gen_text}")

average_confidence = sum(confidence_scores) / len(confidence_scores)
print("\n" + "="*60)
print(f"OOD AVERAGE CONFIDENCE: {average_confidence:.2f}%")
print("="*60)
