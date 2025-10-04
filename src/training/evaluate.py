from ultralytics import YOLO

def evaluate_model(model_path, data_yaml):
    """
    Đánh giá model: Loss, Precision, Recall, F1, Accuracy
    """
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)

    print(f"Precision: {metrics.box.map50:.4f}")
    print(f"Recall: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
