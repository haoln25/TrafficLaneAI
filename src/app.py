import streamlit as st
from detection.detect_video import detect_video, detect_frame  # Import thêm detect_frame
import os
import time
import cv2

base_dir = os.path.dirname(os.path.abspath(__file__))  # Đặt ở đầu file
timestamp = int(time.time())  # Thêm dòng này ngay sau base_dir

st.title("🚦 Nhận diện phương tiện sai làn đường")
uploaded_file = st.file_uploader("Upload video giao thông", type=["mp4", "avi"])

if uploaded_file is not None:
    save_dir = os.path.normpath(os.path.join(base_dir, "..", "data", "raw_videos"))
    os.makedirs(save_dir, exist_ok=True)
    timestamp = int(time.time())
    input_filename = f"temp_video_{timestamp}.mp4"
    save_path = os.path.join(save_dir, input_filename)
    
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"Video đã tải lên: {input_filename}")

    if st.button("Phân tích"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        output_dir = os.path.normpath(os.path.join(base_dir, "..", "output", "results"))
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"out_{timestamp}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        model_path = os.path.normpath(os.path.join(base_dir, "..", "models", "yolo11n.pt"))
        if not os.path.exists(model_path):
            st.error(f"Model không tồn tại tại: {model_path}. Vui lòng tải best.pt từ Colab.")
        else:
            try:
                def update_progress(percent):
                    progress_bar.progress(int(percent))
                    status_text.text(f"Đang phân tích... {int(percent)}%")

                detect_video(save_path, model_path=model_path, output=output_path, progress_callback=update_progress)
                progress_bar.progress(100)
                status_text.text("Phân tích hoàn tất!")

                st.success(f"Phân tích hoàn tất! Bạn có thể xem clip tại: {output_path}")
                st.video(output_path)
                with open(output_path, "rb") as video_file:
                    st.download_button(
                        label="Tải video kết quả",
                        data=video_file,
                        file_name=output_filename,
                        mime="video/mp4"
                    )
            except Exception as e:
                st.error(f"Lỗi khi phân tích: {e}")

        if os.path.exists(save_path):
            os.remove(save_path)
            st.success(f"Đã xóa file tạm: {save_path}")

        if not os.path.exists(output_path):
            st.warning("Không tạo được video output.")

st.subheader("Nhận diện trực tiếp từ camera (không lưu video)")
if st.button("Bắt đầu camera"):
    run_camera = st.empty()
    stop_button = st.button("Kết thúc")
    cap = cv2.VideoCapture(0)
    model_path = os.path.normpath(os.path.join(base_dir, "..", "models", "yolo11n.pt"))
    if not os.path.exists(model_path):
        st.error(f"Model không tồn tại tại: {model_path}. Vui lòng tải best.pt từ Colab.")
    else:
        st.info("Đang mở camera. Nhấn 'Kết thúc' để dừng.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Không lấy được hình ảnh từ camera.")
                break
            # Nhận diện trên frame
            try:
                frame_detected = detect_frame(frame, model_path=model_path)
            except Exception:
                frame_detected = frame
            run_camera.image(cv2.cvtColor(frame_detected, cv2.COLOR_BGR2RGB), channels="RGB")
            if stop_button:
                break
        cap.release()
        run_camera.empty()
        st.success("Đã dừng camera.")