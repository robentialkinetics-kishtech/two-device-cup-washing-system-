import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

def main():
    # Paths
    dataset_dir = Path('yolo dataset')
    without_cup_dir = dataset_dir / 'without_cup'
    train_images_dir = dataset_dir / 'train' / 'images'
    train_labels_dir = dataset_dir / 'train' / 'labels'
    val_images_dir = dataset_dir / 'valid' / 'images'
    val_labels_dir = dataset_dir / 'valid' / 'labels'

    # Get all images without cups
    without_cup_images = list(without_cup_dir.glob('*.jpg')) + list(without_cup_dir.glob('*.png'))
    print(f"Found {len(without_cup_images)} background images without cups")

    # Split into train (80%) and validation (20%)
    random.shuffle(without_cup_images)
    split_idx = int(len(without_cup_images) * 0.8)
    train_bg_images = without_cup_images[:split_idx]
    val_bg_images = without_cup_images[split_idx:]

    print(f"Adding {len(train_bg_images)} background images to training set")
    print(f"Adding {len(val_bg_images)} background images to validation set")

    # Copy training background images and create empty labels
    for img_path in train_bg_images:
        # Copy image
        dst_path = train_images_dir / img_path.name
        shutil.copy2(img_path, dst_path)
        
        # Create empty annotation file (no objects)
        label_name = img_path.stem + '.txt'
        label_path = train_labels_dir / label_name
        label_path.touch()  # Create empty file
        
    print(f"✓ Copied {len(train_bg_images)} background images to train set with empty labels")

    # Copy validation background images and create empty labels
    for img_path in val_bg_images:
        # Copy image
        dst_path = val_images_dir / img_path.name
        shutil.copy2(img_path, dst_path)
        
        # Create empty annotation file (no objects)
        label_name = img_path.stem + '.txt'
        label_path = val_labels_dir / label_name
        label_path.touch()  # Create empty file
        
    print(f"✓ Copied {len(val_bg_images)} background images to validation set with empty labels")

    # Count total data
    train_with_cup = len(list(train_labels_dir.glob('*.txt')))
    val_with_cup = len(list(val_labels_dir.glob('*.txt')))
    print(f"\nDataset Summary:")
    print(f"  Training images: {train_with_cup} (includes {len(train_bg_images)} background)")
    print(f"  Validation images: {val_with_cup} (includes {len(val_bg_images)} background)")

    # Train YOLOv8n with background images included
    print("\n" + "="*60)
    print("Starting YOLOv8n training with cup + background images...")
    print("="*60)

    model = YOLO('yolov8n.pt')
    results = model.train(
        data='yolo dataset/data.yaml',
        epochs=200,
        imgsz=640,
        batch=16,
        patience=20,  # Early stopping
        device=0,  # GPU device (0 for CUDA:0)
        augment=True,  # Data augmentation
        mosaic=1.0,  # Use mosaic augmentation
        degrees=10,  # Rotation augmentation
        translate=0.1,  # Translation augmentation
        scale=0.5,  # Scale augmentation
        flipud=0.5,  # Vertical flip
        fliplr=0.5,  # Horizontal flip
        name='cup_detection_with_background'
    )

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Results saved to: {results}")
    print("="*60)

if __name__ == '__main__':
    main()
