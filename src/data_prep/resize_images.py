import cv2
import os

def resize_images(input_dir, output_dir, size=(640,640)):
    """
    Resize toàn bộ ảnh về kích thước chuẩn YOLO (640x640).
    """
    os.makedirs(output_dir, exist_ok=True)
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        resized = cv2.resize(img, size)
        cv2.imwrite(os.path.join(output_dir, img_name), resized)
    print(f"[INFO] Resized images saved to {output_dir}")
