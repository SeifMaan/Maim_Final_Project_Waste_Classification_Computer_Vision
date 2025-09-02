# app_resnet.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# ---------------------------
# Setup
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
MODEL_DIR = r"C:\Users\seife\OneDrive\Desktop\MAIM-Internship\Final_Project\models"
RESNET_PATH = os.path.join(MODEL_DIR, "resnet18_baseline.pth")

# Classes
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


# ---------------------------
# Load ResNet model
# ---------------------------
def load_resnet():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)


resnet_model = load_resnet()

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# ---------------------------
# Streamlit App
# ---------------------------
st.title("♻️ Waste Classification (ResNet)")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = resnet_model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_class = CLASS_NAMES[pred_idx]
        confidence = probs[0][pred_idx].item() * 100

    st.success(
        f"✅ Prediction: **{pred_class.capitalize()}** ({confidence:.2f}% confidence)"
    )
