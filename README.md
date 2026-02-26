Mini GACS Prototype – Mood & Style Embedding Pipeline

A verification-first affective computing prototype inspired by GenTA GACS

1. Problem Framing (GenTA Context)

Marketing creatives and contemporary art communicate emotion before meaning.
Humans perceive this as vibe — mood, tone, atmosphere, visual energy.

Traditional ML systems classify objects.
But GenTA’s GACS aims to measure perceptual similarity of feeling.

This project demonstrates a minimal pipeline that answers:

Do two visuals feel similar even if they are not visually identical?

Instead of labels like cat or car, we compare embedding space proximity.

2. Approach

We approximate "vibe similarity" using a pretrained multimodal model (CLIP).
CLIP embeddings encode high-level semantics such as:

lighting mood

composition

color tone

aesthetic style

scene atmosphere

Similarity in embedding space ≈ similarity in perceived feeling.


3. Pipeline Architecture
Step 1 — Data Ingestion

Input: 2–3 short art / marketing videos

We sample frames at time intervals to represent visual moments rather than every frame.

Why:

Consecutive frames are redundant; vibe changes slower than pixels.

Step 2 — Frame Extraction

We extract frames every N seconds and store metadata.

Output:

video_id | timestamp | frame_path

Verification:

Checked FPS fallback

Ensured videos load correctly

Verified extracted frame counts

Step 3 — Embedding Generation

We use OpenAI CLIP (ViT-B/32) via HuggingFace + PyTorch.

For each frame:

image → CLIP encoder → 768-dim vector

Verification steps:

Assert embedding shape (N, 768)

Check NaN values

Ensure deterministic output

Confirm identical image → identical embedding

Step 4 — Similarity Computation

We compute pairwise cosine similarity:

Similarity(A,B)=(A⋅B)/(∣∣A∣∣∣∣B∣∣)

Meaning:

1.0 → same vibe

~0.9 → visually consistent

~0.7 → related style

<0.5 → different mood

Step 5 — Retrieval + Visualization

Outputs:

Top-5 similar frames for multiple queries

Similarity heatmap

Interpretation:
Clusters in heatmap represent consistent emotional tone segments.


4. Repository Structure
src/
  extract_frames.py      # video → frames
  compute_embeddings.py  # frames → CLIP embeddings
  similarity.py          # retrieval + heatmap
  tests.py               # verification checks

data/ (ignored in git)
  videos/
  frames/
  metadata.csv
  embeddings.npy
  index.json
5. How to Run
pip install -r requirements.txt

python src/extract_frames.py
python src/compute_embeddings.py
python src/similarity.py
python src/tests.py
6. Verification-First Engineering

Instead of assuming correctness, the pipeline validates:

Check	Purpose
File existence	prevents silent failure
FPS fallback	robustness to corrupted videos
Embedding shape	model output validation
NaN detection	numerical stability
Self-similarity	sanity test
Retrieval consistency	semantic correctness

Result:

All verification tests passed
7. Use of AI Coding Tools (Audited)

AI assistants (ChatGPT/Copilot) were used for:

boilerplate PyTorch loading

initial similarity calculation structure

plotting template

Human validation performed:

fixed incorrect HF output type handling

corrected embedding loop bug

added verification assertions

handled single-frame edge cases

debugged index mismatch

enforced reproducible structure

Final architecture decisions and debugging were manual.

8. Limitations

Current system:

visual only

frame-level independent

no temporal understanding

no audio mood cues

similarity ≠ human emotion ground truth

This is a perceptual proxy, not emotion classification.

9. Toward a Real GACS Engine

This prototype can evolve into a marketing intelligence system:

Step 1 — Multimodal Affect

Add audio embeddings:

music tone

speech energy

rhythm intensity

Step 2 — Temporal Mood Curve

Track emotional progression across time:

calm → tension → excitement → resolution
Step 3 — Performance Feedback Loop

Connect embeddings to ad metrics:

Metric	What model learns
CTR	attention grabbing visuals
Watch time	emotional engagement
ROAS	persuasive aesthetics

Model learns:

which "feel" converts best

10. Key Insight

This project demonstrates a shift:

From object recognition → perception modeling

Instead of asking:

What is in the image?

We ask:

How does it feel?