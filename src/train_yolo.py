import torch
from ultralytics import YOLO
import os

# ✅ Check device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", DEVICE)

# -------------------------
# 1. Load Base Model
# -------------------------
model = YOLO("yolov8n.pt")

# -------------------------
# 2. Dataset Path
# -------------------------
DATA_DIR = (
    r"C:\Users\seife\OneDrive\Desktop\MAIM-Internship\Final_Project\detection_data"
)
DATA_YAML = os.path.join(DATA_DIR, "dataset.yaml")

# -------------------------
# 3. Train Model
# -------------------------
SAVE_DIR = r"C:\Users\seife\OneDrive\Desktop\MAIM-Internship\Final_Project\models"

results = model.train(
    data=DATA_YAML,
    epochs=30,  # increase to 50+ if dataset is large
    imgsz=640,  # YOLO standard image size
    batch=16,  # adjust for GPU memory
    device=0 if DEVICE == "cuda" else "cpu",
    project=SAVE_DIR,  # ✅ where to save
    name="yolov8_waste",  # ✅ subfolder name inside SAVE_DIR
)

# -------------------------
# 4. Evaluate Model
# -------------------------
metrics = model.val()
print("Validation metrics:", metrics)

# -------------------------
# 5. Test on an Image
# -------------------------
test_image = os.path.join(
    DATA_DIR, "images/val/0052.jpg"
)  # replace with an actual test image
results = model.predict(source=test_image, conf=0.5, save=True, device=DEVICE)

# Print detections
for r in results:
    print(r.boxes)  # bounding boxes, confidence, class IDs
    print(r.names)  # class names
