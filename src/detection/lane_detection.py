import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def draw_lines(img, lines, color=[0, 255, 0], thickness=8):
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def average_slope_intercept(image, lines, max_slope=1000):
    left_lines, right_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:  # Tránh chia 0 (đường thẳng đứng)
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) > max_slope:  # Nếu slope quá lớn thì bỏ qua
            continue
        intercept = y1 - slope * x1
        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))

    height = image.shape[0]
    min_y = int(height * 0.6)
    max_y = height

    averaged_lines = []
    for lane_lines in [left_lines, right_lines]:
        if lane_lines:
            slope, intercept = np.mean(lane_lines, axis=0)
            if slope == 0:  # tránh chia 0
                continue
            x1 = int((min_y - intercept) / slope)
            x2 = int((max_y - intercept) / slope)
            averaged_lines.append([[x1, min_y, x2, max_y]])
    return averaged_lines

def draw_lane_lines(frame, yolo_results=None, return_lines=False):
    """
    Vẽ làn đường trên frame sử dụng Hough Transform, tùy chọn trả về tọa độ lane.
    
    Args:
        frame: Hình ảnh đầu vào.
        yolo_results: Kết quả từ YOLO (nếu có, dùng để hỗ trợ detect lane).
        return_lines: Boolean, nếu True thì trả về tọa độ lane.
    
    Returns:
        frame_viz: Frame đã vẽ làn đường.
        lane_lines: Danh sách tọa độ lane (nếu return_lines=True), dạng [(x1, x2), ...].
    """
    frame_viz = frame.copy()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Region of interest (ROI)
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (0, height * 0.6),
        (width, height * 0.6),
        (width, height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=150)
    lane_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
            lane_lines.append((x1, x2))  # Lưu tọa độ x1, x2 của lane

    if return_lines:
        return frame_viz, lane_lines
    return frame_viz