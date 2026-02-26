import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("Checking embedding file...")

emb = np.load("data/embeddings.npy")


assert len(emb.shape) == 2, "Embeddings should be 2D"


sim = cosine_similarity([emb[0]], [emb[0]])[0][0]
assert abs(sim - 1.0) < 1e-5, "Self similarity failed"

print("All verification tests passed")