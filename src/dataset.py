import os
import shutil
from pathlib import Path
import random
from PIL import Image

# Params
RAW_DATA_DIR = "data"  # original dataset (subfolders = classes)
CLEAN_DATA_DIR = "clean_data"  # where cleaned dataset will be stored
IMG_SIZE = (224, 224)  # resize size
SPLIT_RATIO = 0.8  # train/test split


def create_clean_dataset(
    raw_dir=RAW_DATA_DIR,
    clean_dir=CLEAN_DATA_DIR,
    split_ratio=SPLIT_RATIO,
    img_size=IMG_SIZE,
):

    raw_dir = Path(raw_dir)
    clean_dir = Path(clean_dir)

    # Remove old clean dataset if exists
    if clean_dir.exists():
        shutil.rmtree(clean_dir)
    (clean_dir / "train").mkdir(parents=True, exist_ok=True)
    (clean_dir / "test").mkdir(parents=True, exist_ok=True)

    # Loop over each class folder
    for class_name in os.listdir(raw_dir):
        class_path = raw_dir / class_name
        if not class_path.is_dir():
            continue

        # Collect all images in this class
        images = list(class_path.glob("*.*"))
        random.shuffle(images)

        # Train/test split
        split_idx = int(len(images) * split_ratio)
        train_imgs, test_imgs = images[:split_idx], images[split_idx:]

        # Create class subfolders in clean dataset
        (clean_dir / "train" / class_name).mkdir(parents=True, exist_ok=True)
        (clean_dir / "test" / class_name).mkdir(parents=True, exist_ok=True)

        # Process and save train images
        for i, img_path in enumerate(train_imgs):
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(img_size)
                img.save(clean_dir / "train" / class_name / f"{i:04d}.jpg")
            except Exception as e:
                print(f"❌ Skipped {img_path}: {e}")

        # Process and save test images
        for i, img_path in enumerate(test_imgs):
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(img_size)
                img.save(clean_dir / "test" / class_name / f"{i:04d}.jpg")
            except Exception as e:
                print(f"❌ Skipped {img_path}: {e}")

    print(f"✅ Clean dataset created at: {clean_dir}")


# Run it
if __name__ == "__main__":
    create_clean_dataset()
