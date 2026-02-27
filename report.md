# Mini GACS Prototype — Affective Computing Pipeline Report

## 1. Problem Framing

Traditional computer vision systems recognize objects such as people, cars, or animals.  
However, marketing creatives and contemporary art communicate meaning primarily through mood, tone, and atmosphere rather than object identity.

GenTA’s GACS vision focuses on understanding how a visual feels instead of what it contains.

This project demonstrates a minimal prototype that compares perceptual similarity between visuals using representation learning instead of classification.

The goal is not emotion prediction but measuring visual vibe similarity.

---

## 2. System Design

The system is implemented as a simple end-to-end pipeline.

### Video Ingestion
Two short videos are provided as input.  
Instead of processing every frame, the system samples frames at fixed time intervals to capture representative visual moments.

This avoids redundancy because adjacent frames contain nearly identical visual information.

### Frame Extraction
OpenCV is used to read videos and extract frames every N seconds.

Metadata stored:
video_id | timestamp | frame_path

Verification:
- Video load checks
- FPS fallback handling
- Frame count validation

### Embedding Generation
A pretrained CLIP vision encoder is used to convert each frame into a fixed-length vector representation.

frame → CLIP encoder → embedding vector

CLIP is chosen because it learns semantic representations rather than raw pixel patterns, making it suitable for perceptual similarity.

Verification steps:
- Assert embedding shape consistency
- NaN value detection
- Deterministic inference using evaluation mode

### Similarity Computation
Cosine similarity is computed between embeddings.

Similarity(A,B) = (A·B) / (||A|| ||B||)

Higher similarity indicates perceptual closeness rather than object matching.

### Visualization
A similarity heatmap is generated to visualize clusters of visually consistent frames.

---

## 3. Cross-Video Similarity Extension

The pipeline was extended to compare entire videos rather than only individual frames.

Frame embeddings belonging to the same video are averaged to create a single video representation.

video_embedding = mean(frame_embeddings)

This allows measuring overall mood similarity between videos.

Example result:
video1.mp4 ↔ video2.mp4 : 0.526

This indicates partial stylistic similarity while still maintaining distinct tone.

---

## 4. Verification-First Approach

The project emphasizes correctness over scale.

Implemented checks:
- File existence validation
- Embedding shape verification
- NaN detection
- Self similarity equals 1.0
- Retrieval consistency

These tests ensure stable behavior before interpreting results.

---

## 5. Use of AI Coding Tools

AI coding assistants were used to accelerate development for:
- Initial code scaffolding
- Library usage examples
- Plotting templates

Manual validation performed:
- Fixed incorrect model output handling
- Corrected indexing issues
- Added verification assertions
- Implemented edge-case handling

Final design decisions and debugging were done manually.

---

## 6. Limitations

The current prototype only analyzes visual information.

Limitations:
- No audio understanding
- No temporal motion modeling
- Similarity is perceptual proxy, not psychological emotion ground truth

---

## 7. Future Extensions

Potential improvements toward a real GACS engine:

1. Add audio embeddings (music energy, speech tone)
2. Temporal modeling across scenes
3. Connect similarity features to engagement metrics such as CTR or watch time

---

## 8. Conclusion

This project demonstrates how pretrained multimodal representations can approximate perceptual similarity between visuals.

Instead of asking “what is in the image”, the system asks “how similar do these visuals feel”.

This aligns with affective computing goals and provides a foundation for scalable creative analysis systems.