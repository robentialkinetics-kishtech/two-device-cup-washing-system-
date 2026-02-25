import cv2
import os
from pathlib import Path
from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Test on test set images (handle folder name with spaces)
test_images_dir = Path('yolo dataset') / 'test' / 'images'
test_labels_dir = Path('yolo dataset') / 'test' / 'labels'

print("\n" + "="*80)
print("TESTING YOLO MODEL ON TEST DATASET")
print("="*80)

if not os.path.exists(str(test_images_dir)):
    print(f"❌ Test images directory not found: {test_images_dir}")
    exit(1)

# Run inference on test images
test_images = [f for f in os.listdir(str(test_images_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"\nFound {len(test_images)} test images")

predictions = []
ground_truths = []

for img_name in test_images:
    img_path = test_images_dir / img_name
    label_path = test_labels_dir / (Path(img_name).stem + '.txt')
    
    # Run detection
    results = model(str(img_path), verbose=False)
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []
    
    # Count detections
    num_detections = len(detections)
    
    # Read ground truth
    gt_count = 0
    if label_path.exists():
        with open(str(label_path), 'r') as f:
            gt_count = len(f.readlines())
    
    predictions.append(num_detections)
    ground_truths.append(gt_count)
    
    print(f"{img_name}: Detected {num_detections} cups (Ground truth: {gt_count})")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total test images: {len(test_images)}")
print(f"Total predictions: {sum(predictions)}")
print(f"Total ground truth: {sum(ground_truths)}")
print(f"Average detections per image: {sum(predictions)/len(predictions):.2f}")
print(f"Average ground truth per image: {sum(ground_truths)/len(ground_truths):.2f}")
print("="*80 + "\n")
