from ultralytics import YOLO

# Train YOLOv8n on yolo dataset with polygon annotations
model = YOLO('yolov8n.pt')
model.train(
    data='yolo dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
