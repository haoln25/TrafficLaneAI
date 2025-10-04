def is_vehicle_in_wrong_lane(vehicle_box, lane_boundaries):
    """
    Kiểm tra xem xe có đi sai làn không.
    vehicle_box: (x1,y1,x2,y2)
    lane_boundaries: ngưỡng lane (ví dụ chia màn hình 3 phần)
    """
    x1,y1,x2,y2 = vehicle_box
    center_x = (x1+x2)//2

    if center_x < lane_boundaries[0]:
        return "Sai làn trái"
    elif center_x > lane_boundaries[1]:
        return "Sai làn phải"
    else:
        return "Đúng làn"
