from ultralytics import YOLO

# Train YOLOv8n with augmentation for better generalization
model = YOLO('yolov8n.pt')

# Train with aggressive augmentation to reduce overfitting
model.train(
    data='yolo dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    
    # Augmentation parameters for better generalization
    hsv_h=0.015,      # HSV-Hue augmentation (more color variation)
    hsv_s=0.7,        # HSV-Saturation augmentation
    hsv_v=0.4,        # HSV-Value augmentation
    degrees=15,       # Rotation 
    translate=0.2,    # Translation
    scale=0.5,        # Scale variation
    flipud=0.5,       # Flip upside down
    fliplr=0.5,       # Flip left-right
    mosaic=1.0,       # Mosaic augmentation
    mixup=0.1,        # Mix-up augmentation
    
    # Regularization to prevent overfitting
    weight_decay=0.0005,
    warmup_epochs=3,
    
    # Better validation
    patience=20,      # Early stopping patience
)
