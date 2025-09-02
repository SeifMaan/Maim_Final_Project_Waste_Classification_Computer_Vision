import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# -------------------------
# Config
# -------------------------
DATA_DIR = (
    r"C:\Users\seife\OneDrive\Desktop\MAIM-Internship\Final_Project\clean_data\train"
)
SAVE_DIR = (
    r"C:\Users\seife\OneDrive\Desktop\MAIM-Internship\Final_Project\generated_data"
)

os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 64  # GANs usually work on 64x64
BATCH_SIZE = 128
EPOCHS = 200
LATENT_DIM = 100
LR = 0.0002
BETA1 = 0.5  # recommended for GANs

# -------------------------
# Data (train only)
# -------------------------
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # normalize to [-1, 1]
    ]
)

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

num_classes = len(dataset.classes)
print("Classes:", dataset.classes)


# -------------------------
# Generator
# -------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


# -------------------------
# Discriminator
# -------------------------
class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img):
        return self.net(img).view(-1, 1).squeeze(1)


# -------------------------
# Init models
# -------------------------
generator = Generator(LATENT_DIM).to(DEVICE)
discriminator = Discriminator().to(DEVICE)

criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))

# -------------------------
# Training Loop
# -------------------------
for epoch in range(EPOCHS):
    for i, (imgs, _) in enumerate(dataloader):
        real_imgs = imgs.to(DEVICE)
        batch_size = real_imgs.size(0)

        # Labels
        real_labels = torch.full((batch_size,), 0.9, device=DEVICE)
        fake_labels = torch.zeros(batch_size, device=DEVICE)

        # -------------------------
        # Train Discriminator
        # -------------------------
        z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
        fake_imgs = generator(z)

        real_loss = criterion(discriminator(real_imgs), real_labels)
        fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels)
        d_loss = (real_loss + fake_loss) / 2

        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # -------------------------
        # Train Generator
        # -------------------------
        g_loss = criterion(discriminator(fake_imgs), real_labels)

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}"
    )

    # Save ONE sample image every 20 epochs
    if (epoch + 1) % 20 == 0:
        z = torch.randn(1, LATENT_DIM, 1, 1, device=DEVICE)  # only 1 noise vector
        sample = generator(z)
        save_image(
            sample,
            os.path.join(SAVE_DIR, f"epoch_{epoch+1}_sample.png"),
            normalize=True,
        )


# -------------------------
# Save Generator
# -------------------------
torch.save(
    generator.state_dict(),
    os.path.join(
        "C:\\Users\\seife\\OneDrive\\Desktop\\MAIM-Internship\\Final_Project\\models",
        "gan_generator.pth",
    ),
)
print("✅ Generator model saved to models/gan_generator.pth")
