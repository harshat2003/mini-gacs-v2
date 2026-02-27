import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_PATH = "data/embeddings.npy"
INDEX_PATH = "data/index.json"

# Load embeddings
embeddings = np.load(EMBEDDINGS_PATH)

with open(INDEX_PATH, "r") as f:
    index = json.load(f)

# -------- Group embeddings by video --------
video_groups = {}

for i in range(len(embeddings)):
    frame_name = index[str(i)]
    video_name = frame_name.split("_")[0]   # video1.mp4_t2.0.jpg -> video1.mp4

    if video_name not in video_groups:
        video_groups[video_name] = []

    video_groups[video_name].append(embeddings[i])

# -------- Average embedding per video --------
video_vectors = {}
for video, vecs in video_groups.items():
    video_vectors[video] = np.mean(vecs, axis=0)

videos = list(video_vectors.keys())
vectors = np.array([video_vectors[v] for v in videos])

# -------- Similarity --------
similarity_matrix = cosine_similarity(vectors)

print("\n=== Cross Video Vibe Similarity ===")
for i in range(len(videos)):
    for j in range(i + 1, len(videos)):
        print(f"{videos[i]} ↔ {videos[j]} : {similarity_matrix[i][j]:.3f}")