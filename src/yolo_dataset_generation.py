import os
import shutil
from pathlib import Path
from PIL import Image

# Adjust these paths if needed
CLEAN_DIR = Path("clean_data")   # already: clean_data/train, clean_data/test
YOLO_DIR  = Path("detection_data")  # output
MARGIN = 0.01   # shrink bbox slightly so it doesn't exactly touch edges

def build_yolo_dataset(clean_dir=CLEAN_DIR, yolo_dir=YOLO_DIR, margin=MARGIN):
    # class order pulled from train folders (must match labels)
    train_folder = clean_dir / "train"
    classes = sorted([p.name for p in train_folder.iterdir() if p.is_dir()])
    print("Classes (order):", classes)

    # make dirs
    for split in ["train", "val"]:
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Use clean_dataset/train -> images/train ; clean_dataset/test -> images/val
    mapping = [("train", "train"), ("val", "test")]

    for out_split, in_split in mapping:
        for cls_idx, cls_name in enumerate(classes):
            src_dir = clean_dir / in_split / cls_name
            if not src_dir.exists():
                continue
            for img_path in src_dir.glob("*.*"):
                try:
                    dst_img = yolo_dir / "images" / out_split / img_path.name
                    shutil.copy(img_path, dst_img)

                    # get image size (not strictly necessary here but included)
                    w, h = Image.open(img_path).size

                    # full-ish bbox centered (x_center, y_center, w, h), normalized
                    x_c = 0.5
                    y_c = 0.5
                    bw = max(0.01, 1.0 - 2 * margin)
                    bh = max(0.01, 1.0 - 2 * margin)

                    label_path = (yolo_dir / "labels" / out_split / img_path.with_suffix(".txt").name)
                    with open(label_path, "w") as f:
                        f.write(f"{cls_idx} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
                except Exception as e:
                    print("Skip", img_path, e)

    # write a small dataset config file
    yaml_path = yolo_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"train: {str((yolo_dir/'images'/'train').resolve())}\n")
        f.write(f"val:   {str((yolo_dir/'images'/'val').resolve())}\n")
        f.write(f"nc: {len(classes)}\n")
        f.write("names: [\n")
        for i, n in enumerate(classes):
            f.write(f'  \"{n}\"' + (",\n" if i != len(classes)-1 else "\n"))
        f.write("]\n")

    print(f"✅ YOLO dataset created at: {yolo_dir}")
    print(f"Dataset YAML: {yaml_path}")

if __name__ == "__main__":
    build_yolo_dataset()
