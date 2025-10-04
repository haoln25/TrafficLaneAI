import cv2
import os
import pandas as pd
from ultralytics import YOLO
from detection.lane_detection import draw_lane_lines
from detection.violation_check import is_vehicle_in_wrong_lane

def detect_video(video_path, model_path="models/best.pt", output="output/results/out.mp4", progress_callback=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.normpath(os.path.join(base_dir, "..", "..", output))
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[ERROR] Không thể load model: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Thử các codec khác nhau
    codecs = [("mp4v", "mp4v"), ("XVID", "XVID"), ("MJPG", "MJPG")]
    out = None
    for fourcc_str, codec_name in codecs:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            print(f"[INFO] Sử dụng codec: {codec_name}")
            break
    if out is None:
        print(f"[ERROR] Không thể tạo file output với bất kỳ codec nào: {output_path}")
        cap.release()
        return

    violations = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame_viz, lane_lines = draw_lane_lines(frame, results, return_lines=True)

        lane_boxes = [box for box in results[0].boxes if int(box.cls[0]) in [2, 3]]
        lane_boundaries = (width // 3, 2 * width // 3) if not lane_boxes else (
            int(min([b.xyxy[0][0] for b in lane_boxes])),
            int(max([b.xyxy[0][2] for b in lane_boxes]))
        )
        if len(lane_lines) > 0:
            lane_boundaries = (int(min([x for x, _ in lane_lines])), int(max([x for _, x in lane_lines])))

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = box.conf[0]
                label = model.names[cls]
                cv2.rectangle(frame_viz, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame_viz, f"{label} {conf:.2f}", (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if cls in [0, 1, 2]:
                    status = is_vehicle_in_wrong_lane((x1, y1, x2, y2), lane_boundaries)
                    color = (0, 0, 255) if status != "Đúng làn" else (0, 255, 0)
                    cv2.putText(frame_viz, f"{label}-{status}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if status != "Đúng làn":
                        violations.append({
                            'frame': frame_id,
                            'time': frame_id / fps,
                            'class': label,
                            'confidence': float(conf),
                            'status': status,
                            'bbox': [x1, y1, x2, y2]
                        })

        out.write(frame_viz)
        frame_id += 1
        if progress_callback and total_frames > 0:
            progress_callback((frame_id / total_frames) * 100)
        print(f"[INFO] Đang xử lý frame {frame_id}/{total_frames}")

    cap.release()
    out.release()
    if violations:
        violations_df = pd.DataFrame(violations)
        violations_csv = os.path.join(output_dir, f"violations_{os.path.basename(output_path)}.csv")
        violations_df.to_csv(violations_csv, index=False)
        print(f"[INFO] Log vi phạm lưu tại: {violations_csv}")
    print(f"[INFO] Kết quả lưu tại: {output_path}")

if __name__ == "__main__":
    detect_video("data/raw/sample_video.mp4", "models/best.pt")