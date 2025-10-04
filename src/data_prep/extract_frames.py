import cv2
import os

def extract_frames(video_path, output_dir, frame_rate=30):
    """
    Tách frame từ video.
    - video_path: đường dẫn video gốc
    - output_dir: nơi lưu frame
    - frame_rate: số frame mỗi giây
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count, frame_id = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:  # lấy frame theo tần suất
            cv2.imwrite(f"{output_dir}/frame_{frame_id}.jpg", frame)
            frame_id += 1
        count += 1
    cap.release()
    print(f"[INFO] Extracted {frame_id} frames to {output_dir}")
