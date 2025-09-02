import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os


# ==============================
# Config
# ==============================
DATA_DIR = (
    "C:\\Users\\seife\\OneDrive\\Desktop\\MAIM-Internship\\Final_Project\\clean_data"
)
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 20
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("✅ Using device:", DEVICE)


# ==============================
# Transforms
# ==============================
train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# ==============================
# Datasets & Loaders
# ==============================
train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
test_dataset = datasets.ImageFolder(f"{DATA_DIR}/test", transform=test_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,  # <--- speed up data loading
    pin_memory=True,  # <--- faster transfer to GPU
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)

class_names = train_dataset.classes
print("Classes:", class_names)


# ==============================
# Model (ResNet18)
# ==============================
model = models.resnet18(pretrained=True)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

model = model.to(DEVICE)
print("Model is on:", next(model.parameters()).device)


# ==============================
# Training Setup
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)


# ==============================
# Training Loop
# ==============================
train_losses, train_accs = [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(
            DEVICE, non_blocking=True
        )

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")


# ==============================
# Evaluation
# ==============================
model.eval()
correct, total = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(
            DEVICE, non_blocking=True
        )
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = 100 * correct / total
print(f"\n✅ Test Accuracy: {test_acc:.2f}%")
print(
    "\nClassification Report:\n",
    classification_report(all_labels, all_preds, target_names=class_names),
)

# ==============================
# Visualization
# ==============================
# Plot training loss & accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy")
plt.legend()

plt.show()

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names,
    cmap="Blues",
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


os.makedirs("models", exist_ok=True)
model_path = "models/resnet18_baseline.pth"
torch.save(model.state_dict(), model_path)
print(f"💾 Model saved to {model_path}")
