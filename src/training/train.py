from ultralytics import YOLO

def train_model(data_yaml, model_out="models/best.pt", epochs=50):
    """
    Train YOLOv8 model trên dataset đã chuẩn bị.
    """
    model = YOLO("yolov8n.pt")  # Pretrained
    model.train(data=data_yaml, epochs=epochs, imgsz=640)
    model.export(format="onnx")  # Export ONNX nếu cần
