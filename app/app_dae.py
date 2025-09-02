# app_dae.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
MODEL_DIR = r"C:\Users\seife\OneDrive\Desktop\MAIM-Internship\Final_Project\models"
DAE_PATH = os.path.join(MODEL_DIR, "dae.pth")


# ---------------------------
# DAE model (must match training)
# ---------------------------
class SimpleDAE(nn.Module):
    def __init__(self):
        super(SimpleDAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Load model
dae_model = SimpleDAE().to(DEVICE)
if os.path.exists(DAE_PATH):
    dae_model.load_state_dict(torch.load(DAE_PATH, map_location=DEVICE))
dae_model.eval()

# Transform
dae_transform = transforms.Compose(
    [transforms.Resize((128, 128)), transforms.ToTensor()]
)

# Streamlit UI
st.title("♻️ Denoising Autoencoder")
uploaded_file = st.file_uploader("Upload a noisy image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Noisy Input", use_column_width=True)

    input_tensor = dae_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = dae_model(input_tensor).cpu().squeeze().permute(1, 2, 0).numpy()

    st.image(output, caption="Denoised Output", use_column_width=True, clamp=True)
