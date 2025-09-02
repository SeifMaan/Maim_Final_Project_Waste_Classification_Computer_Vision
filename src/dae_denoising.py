import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# -------------------------
# Config
# -------------------------
DATA_DIR = r"C:\Users\seife\OneDrive\Desktop\MAIM-Internship\Final_Project\clean_data"
MODEL_DIR = r"C:\Users\seife\OneDrive\Desktop\MAIM-Internship\Final_Project\models"
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 32
IMG_SIZE = 128
EPOCHS = 20
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", DEVICE)

# -------------------------
# Transforms
# -------------------------
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
)

train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"), transform=transform
)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# -------------------------
# Add Noise Function
# -------------------------
def add_noise(imgs, noise_factor=0.3):
    noisy = imgs + noise_factor * torch.randn_like(imgs)
    return torch.clamp(noisy, 0.0, 1.0)


# -------------------------
# Autoencoder Model
# -------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# -------------------------
# Training Setup
# -------------------------
model = Autoencoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
best_loss = float("inf")
MODEL_PATH = os.path.join(MODEL_DIR, "dae.pth")

# -------------------------
# Training Loop
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs = imgs.to(DEVICE)
        noisy_imgs = add_noise(imgs).to(DEVICE)

        outputs = model(noisy_imgs)
        loss = criterion(outputs, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f}")

    # Save best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"💾 Best model saved with loss {best_loss:.4f}")

# -------------------------
# Evaluation + Visualization
# -------------------------
model.eval()
dataiter = iter(test_loader)
images, _ = next(dataiter)
noisy_images = add_noise(images).to(DEVICE)

with torch.no_grad():
    outputs = model(noisy_images)

# move back to CPU + numpy for plotting
images = images.numpy().transpose((0, 2, 3, 1))
noisy_images = noisy_images.cpu().numpy().transpose((0, 2, 3, 1))
outputs = outputs.cpu().numpy().transpose((0, 2, 3, 1))

# Show results
fig, axes = plt.subplots(3, 6, figsize=(12, 6))
for i in range(6):
    axes[0, i].imshow(images[i])
    axes[0, i].set_title("Original")
    axes[0, i].axis("off")

    axes[1, i].imshow(noisy_images[i])
    axes[1, i].set_title("Noisy")
    axes[1, i].axis("off")

    axes[2, i].imshow(outputs[i])
    axes[2, i].set_title("Denoised")
    axes[2, i].axis("off")

plt.tight_layout()
plt.show()

print(f"✅ Training complete. Final model saved to {MODEL_PATH}")
