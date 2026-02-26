import cv2
import os
import csv

VIDEO_DIR = "data/videos"
FRAME_DIR = "data/frames"
METADATA_PATH = "data/metadata.csv"

os.makedirs(FRAME_DIR, exist_ok=True)

TARGET_FRAMES_PER_VIDEO = 30   # <-- guarantees 20–50 frames

print("Looking for videos in:", VIDEO_DIR)
videos = os.listdir(VIDEO_DIR)
print("Files found:", videos)

metadata = []

for video_name in videos:
    if not video_name.lower().endswith(".mp4"):
        continue

    video_path = os.path.join(VIDEO_DIR, video_name)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate gap automatically
    frame_gap = max(1, total_frames // TARGET_FRAMES_PER_VIDEO)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_gap == 0:
            timestamp = round(frame_count / fps, 2)
            frame_filename = f"{video_name}_frame{saved_count}.jpg"
            frame_path = os.path.join(FRAME_DIR, frame_filename)

            cv2.imwrite(frame_path, frame)

            metadata.append([video_name, timestamp, frame_path])
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Processed {video_name} → {saved_count} frames extracted")

# Save metadata
with open(METADATA_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["video_id", "timestamp", "frame_path"])
    writer.writerows(metadata)

print("Frame extraction completed successfully.")