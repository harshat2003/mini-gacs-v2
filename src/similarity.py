import os
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_PATH = "data/embeddings.npy"
INDEX_PATH = "data/index.json"


# ---------- Safety checks ----------
if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError("Embeddings file not found. Run compute_embeddings.py first")

if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError("Index mapping not found")

# ---------- Load data ----------
embeddings = np.load(EMBEDDINGS_PATH)
print("Embeddings shape:", embeddings.shape)

# Verification checks
assert len(embeddings.shape) == 2, "Embeddings should be 2D"
assert not np.isnan(embeddings).any(), "NaN values detected in embeddings"

with open(INDEX_PATH, "r") as f:
    index = json.load(f)

# Helper function (fixes KeyError issue)
def get_name(i):
    return index.get(str(i), index.get(i, f"frame_{i}"))

# ---------- Similarity ----------
similarity_matrix = cosine_similarity(embeddings)
print("Similarity matrix shape:", similarity_matrix.shape)


# ---------- Retrieval Function ----------
def show_top_similar(query_idx, top_k=5):
    if len(embeddings) <= 1:
        print("\nOnly one frame available — similarity comparison not possible.")
        return

    similarities = similarity_matrix[query_idx]

    # exclude itself
    sorted_indices = similarities.argsort()[::-1]
    top_indices = [i for i in sorted_indices if i != query_idx][:top_k]

    print(f"\nQuery Frame: {get_name(query_idx)}")
    print("Top similar frames:")

    for rank, idx in enumerate(top_indices, start=1):
        print(f"{rank}. {get_name(idx)} | similarity: {similarities[idx]:.3f}")


# ---------- Run retrieval on multiple queries ----------
num_queries = min(3, len(embeddings))

for q in range(num_queries):
    show_top_similar(q)


# ---------- Heatmap ----------
plt.figure(figsize=(6, 5))
plt.imshow(similarity_matrix, cmap="viridis")
plt.colorbar(label="Cosine Similarity")
plt.title("Frame Similarity Heatmap")
plt.xlabel("Frame Index")
plt.ylabel("Frame Index")
plt.tight_layout()
plt.show()