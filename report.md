# Mini-GACS Technical Report

This project implements a minimal affective similarity pipeline using CLIP embeddings.

Key Idea:
Similarity is treated as a perceptual proxy rather than emotion classification.

Key Observation:
Cross-video similarity produced non-trivial scores (~0.52), indicating shared visual tone despite different content.

Conclusion:
Embedding distance can model aesthetic consistency, supporting the GACS hypothesis that affective similarity emerges from semantic embedding space.