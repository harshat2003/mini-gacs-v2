import os
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_PATH = "data/embeddings.npy"
INDEX_PATH = "data/index.json"



if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError("Embeddings file not found. Run compute_embeddings.py first")

if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError("Index mapping not found")


embeddings = np.load(EMBEDDINGS_PATH)
print("Embeddings shape:", embeddings.shape)


assert len(embeddings.shape) == 2, "Embeddings should be 2D"
assert not np.isnan(embeddings).any(), "NaN values detected in embeddings"

with open(INDEX_PATH, "r") as f:
    index = json.load(f)


def get_name(i):
    return index.get(str(i), index.get(i, f"frame_{i}"))


similarity_matrix = cosine_similarity(embeddings)
print("Similarity matrix shape:", similarity_matrix.shape)



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



num_queries = min(3, len(embeddings))

for q in range(num_queries):
    show_top_similar(q)


#code for heatmap
plt.figure(figsize=(6, 5))
plt.imshow(similarity_matrix, cmap="viridis")
plt.colorbar(label="Cosine Similarity")
plt.title("Frame Similarity Heatmap")
plt.xlabel("Frame Index")
plt.ylabel("Frame Index")
plt.tight_layout()


os.makedirs("results", exist_ok=True)
output_path = "results/heatmap.png"
plt.savefig(output_path, dpi=300)
print(f"\nHeatmap saved at: {output_path}")

plt.show()