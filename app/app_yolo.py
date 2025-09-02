# app_yolo.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_PATH = (
    r"C:\Users\seife\OneDrive\Desktop\MAIM-Internship\Final_Project\src\yolov8n.pt"
)

# Load YOLO
yolo_model = YOLO(YOLO_PATH)

st.title("♻️ Waste Object Detection (YOLOv8)")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img_path = "temp_detection.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    results = yolo_model.predict(source=img_path, conf=0.5, save=False, device=DEVICE)

    for r in results:
        st.image(r.plot(), caption="YOLOv8 Detection", use_column_width=True)
        st.json(r.names)
