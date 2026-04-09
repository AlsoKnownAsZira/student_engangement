import cv2
import numpy as np
from ultralytics import YOLO

# Buat dummy image
img = np.zeros((100, 100, 3), dtype=np.uint8)

# Load model
model = YOLO('models/roboflow_weights.pt')

print("Class Names:", model.names)
print("Model Task:", model.task)

# Inference
results = model(img)
r = results[0]

print("\n--- INFERENCE RESULTS ---")
print("Has probs (Classification)?", r.probs is not None)
print("Has boxes (Detection)?", r.boxes is not None)
if r.probs is not None:
    print("top1:", r.probs.top1)
if r.boxes is not None and len(r.boxes) > 0:
    print("box cls:", r.boxes.cls)
