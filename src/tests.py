import numpy as np

embeddings = np.load("data/embeddings.npy")

print("Checking NaN values...")
assert not np.isnan(embeddings).any()

print("Checking embedding shape...")
assert len(embeddings.shape) == 2

print("All verification tests passed")