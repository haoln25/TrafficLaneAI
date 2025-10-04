import cv2
import os
import random

def augment_images(input_dir, output_dir):
    """
    Augment ảnh: flip ngang, tăng giảm sáng.
    """
    os.makedirs(output_dir, exist_ok=True)
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # flip
        flipped = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(output_dir, f"{img_name}_flip.jpg"), flipped)

        # sáng tối ngẫu nhiên
        alpha = 1 + random.uniform(-0.3, 0.3)  # hệ số sáng
        bright = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        cv2.imwrite(os.path.join(output_dir, f"{img_name}_bright.jpg"), bright)
