# ♻️ Waste Management AI System

An end-to-end **Deep Learning project** for **waste classification, detection, denoising, and data augmentation**, wrapped in a user-friendly **Streamlit web app**.

This project demonstrates how multiple AI techniques can work together to build a practical **smart waste management system**.

---

## 📌 Features

- **Classification (ResNet18)**
  - Classifies uploaded waste images into categories (cardboard, glass, metal, paper, plastic, trash).
- **Object Detection (YOLOv8)**
  - Detects multiple waste items in a single image with bounding boxes.
- **Denoising Autoencoder (DAE)**
  - Removes noise and restores clarity in blurry/dirty waste images.
- **Generative Adversarial Network (GAN)**
  - Generates synthetic waste images for dataset augmentation.
- **Streamlit App**
  - Unified interface with 4 tabs: Classification | Detection | Denoising | GAN Viewer.

---

## 📂 Project Structure

```
Final_Project/
│── data/                     # raw dataset
│── clean_data/               # cleaned/preprocessed dataset
│── detection_data/           # YOLO dataset (with labels)
│── gan_samples/              # GAN generated images
│── models/                   # trained models
│   ├── resnet18.pth
│   ├── yolov8n.pt
│   ├── dae.pth
│── src/                             # training scripts
│   ├── dataset.py                   # preprocessing & dataset splitting
│   ├── train_resnet.py              # train ResNet classifier
│   ├── train_yolo.py                # train YOLOv8 detector
│   ├── yolo_dataset_generation.py   # generate yolo dataset
│   ├── dae_denoising.py             # train autoencoder
│   ├── gan_generator.py             # train GAN
│   ├── test.ipynb                   # test notebook
│── app/                     # Streamlit applications
│   ├── app_dae.py           # DAE app
│   ├── app_gan.py           # GAN app
│   ├── app_resnet.py        # ResNet app
│   ├── app_yolo.py          # Yolo app
│── README.md                 # project documentation
│── requirements.txt          # project requirements

```

---

## ⚙️ Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

**Main requirements**:

- `torch`
- `torchvision`
- `ultralytics`
- `streamlit`
- `matplotlib`
- `pillow`

---

## 📊 Dataset

- Source: [Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data)
- Categories:
  - `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`
- Preprocessing done via `src/prepare_data.py`:
  - Cleans corrupted files
  - Splits into `train/` and `test/`
  - Normalizes & resizes

---

## 🧠 Models

### 1. **ResNet18 Classifier**

- Input: 224×224 RGB image
- Output: Waste category
- Accuracy: ~83% train / ~78% test

### 2. **YOLOv8 Detector**

- Input: Images with multiple objects
- Output: Bounding boxes + class labels

### 3. **Denoising Autoencoder (DAE)**

- Input: Noisy waste image
- Output: Denoised, reconstructed image

### 4. **GAN**

- Trains to generate realistic waste images
- Used for dataset augmentation

---

## 🚀 Usage

### Run the app:

```bash
streamlit run app_dae.py
streamlit run app_gan.py
streamlit run app_resnet.py
streamlit run app_yolo.py

```

### App Features:

- **Classification** → Upload image → Get category prediction
- **Detection** → Upload image → Detect multiple waste items
- **Denoising** → Upload noisy image → See cleaned version
- **GAN Viewer** → Browse generated synthetic samples

---

## 📈 Training

Each model can be retrained via scripts in `src/`.

Example (ResNet):

```bash
python src/train_resnet.py
```

Trained models are saved in `models/`.

---
