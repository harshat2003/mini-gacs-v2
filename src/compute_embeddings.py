import os
import json
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

FRAME_DIR = "data/frames"
OUTPUT_DIR = "data"
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "embeddings.npy")
INDEX_PATH = os.path.join(OUTPUT_DIR, "index.json")

print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Using device:", device)

frame_files = sorted([
    f for f in os.listdir(FRAME_DIR)
    if f.lower().endswith(".jpg")
])

print(f"Found {len(frame_files)} frames")

embeddings = []
index = {}

for idx, frame_file in enumerate(frame_files):
    frame_path = os.path.join(FRAME_DIR, frame_file)
    image = Image.open(frame_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    # Fix for HF output object
    if hasattr(outputs, "pooler_output"):
        image_features = outputs.pooler_output
    else:
        image_features = outputs

    embedding = image_features.cpu().numpy().flatten()

    embeddings.append(embedding)
    index[idx] = frame_file

    if (idx + 1) % 5 == 0:
        print(f"Processed {idx + 1}/{len(frame_files)} frames")

embeddings = np.array(embeddings)

print("Embedding shape:", embeddings.shape)
assert not np.isnan(embeddings).any(), "NaN values found in embeddings!"

np.save(EMBEDDINGS_PATH, embeddings)

with open(INDEX_PATH, "w") as f:
    json.dump(index, f, indent=2)

print("Embeddings saved successfully.")