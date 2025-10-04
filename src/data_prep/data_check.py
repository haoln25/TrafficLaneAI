import os
import cv2

def check_data(images_dir, labels_dir):
    """
    Kiểm tra ảnh + nhãn YOLO có khớp nhau không.
    """
    images = set(os.path.splitext(f)[0] for f in os.listdir(images_dir))
    labels = set(os.path.splitext(f)[0] for f in os.listdir(labels_dir))

    missing_labels = images - labels
    missing_images = labels - images

    if missing_labels:
        print("[WARN] Ảnh thiếu nhãn:", missing_labels)
    if missing_images:
        print("[WARN] Nhãn không có ảnh:", missing_images)
    if not missing_labels and not missing_images:
        print("[OK] Dataset hợp lệ!")
