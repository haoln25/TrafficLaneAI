import os
import shutil
import yaml
import random

def prepare_dataset(images_dir, labels_dir, output_dir, split=(0.7, 0.2, 0.1)):
    """
    Chia dữ liệu thành train/val/test + tạo dataset.yaml cho YOLO.
    """
    for folder in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", folder), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", folder), exist_ok=True)

    images = os.listdir(images_dir)
    random.shuffle(images)

    n = len(images)
    n_train, n_val = int(split[0]*n), int(split[1]*n)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:]
    }

    for split_name, files in splits.items():
        for f in files:
            base = os.path.splitext(f)[0]
            shutil.copy(os.path.join(images_dir, f), os.path.join(output_dir, "images", split_name, f))
            shutil.copy(os.path.join(labels_dir, base+".txt"), os.path.join(output_dir, "labels", split_name, base+".txt"))

    dataset_yaml = {
        "path": output_dir,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 4,
        "names": ["car", "motorbike", "bus", "truck"]
    }

    with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
        yaml.dump(dataset_yaml, f)

    print("[INFO] Dataset prepared with train/val/test split.")
