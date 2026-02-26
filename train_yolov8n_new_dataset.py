"""
YOLOv8n Training Script for New Dataset Structure
Trains on the combined 'yolo dataset' with multiple area datasets
(picking, brushing, rinsing areas) + background images

Dataset structure:
yolo dataset/
├── picking area dataset/
│   ├── train/images & labels
│   ├── valid/images & labels
│   ├── test/images & labels
│   └── data.yaml
├── brushing area dataset/
│   ├── train/images & labels
│   ├── valid/images & labels
│   ├── test/images & labels
│   └── data.yaml
├── rinsing area dataset/
│   ├── train/images & labels
│   ├── valid/images & labels
│   ├── test/images & labels
│   └── data.yaml
├── without_cup/  (background images - no cups)
│   └── *.jpg, *.png

Background images are split 80/20 and added to train/val sets
with empty labels to teach the model what NOT to detect.
"""

import os
import shutil
import yaml
import random
from pathlib import Path
from ultralytics import YOLO

def create_combined_dataset(base_path="yolo dataset"):
    """
    Create a combined dataset from all area datasets + background images
    Combines train, val, test splits from all three datasets
    Also adds background images (without cups) to reduce false positives
    """
    base_path = Path(base_path)
    
    datasets = [
        base_path / "picking area dataset",
        base_path / "brushing area dataset",
        base_path / "rinsing area dataset"
    ]
    
    # Create combined dataset directory
    combined_path = base_path / "_combined"
    combined_path.mkdir(exist_ok=True)
    
    for split in ["train", "val", "test"]:
        split_img_path = combined_path / split / "images"
        split_lbl_path = combined_path / split / "labels"
        split_img_path.mkdir(parents=True, exist_ok=True)
        split_lbl_path.mkdir(parents=True, exist_ok=True)
    
    # Copy data from all datasets
    print("📦 Combining datasets...")
    image_count = {"train": 0, "val": 0, "test": 0}
    
    for dataset_path in datasets:
        if not dataset_path.exists():
            print(f"⚠ Skipping {dataset_path.name} - not found")
            continue
            
        print(f"\n📂 Processing {dataset_path.name}...")
        
        for split in ["train", "val", "test"]:
            src_img_dir = dataset_path / split / "images"
            src_lbl_dir = dataset_path / split / "labels"
            
            dst_img_dir = combined_path / split / "images"
            dst_lbl_dir = combined_path / split / "labels"
            
            if not src_img_dir.exists():
                print(f"   ⚠ {split}/images not found")
                continue
            
            # Copy image and label files
            img_files = list(src_img_dir.glob("*"))
            for img_file in img_files:
                shutil.copy2(img_file, dst_img_dir / img_file.name)
                image_count[split] += 1
                
                # Also copy corresponding label file
                label_file = src_lbl_dir / (img_file.stem + ".txt")
                if label_file.exists():
                    shutil.copy2(label_file, dst_lbl_dir / label_file.name)
            
            print(f"   ✓ {split}: {len(img_files)} images copied")
    
    # ═════════════════════════════════════════════════════════════
    # ADD BACKGROUND IMAGES (NO CUPS)
    # ═════════════════════════════════════════════════════════════
    print("\n🖼️  Adding background images (images without cups)...")
    
    # Check for background images in multiple locations
    background_sources = [
        base_path / "without_cup",  # From yolo dataset folder
        Path("diverse_dataset") / "without_cup",  # From diverse_dataset folder
    ]
    
    bg_images = []
    for bg_source in background_sources:
        if bg_source.exists():
            bg_imgs = list(bg_source.glob("*.jpg")) + list(bg_source.glob("*.png"))
            bg_images.extend(bg_imgs)
            print(f"   Found {len(bg_imgs)} background images in {bg_source.name}")
    
    if bg_images:
        # Shuffle and split: 80% train, 20% val
        random.shuffle(bg_images)
        split_idx = int(len(bg_images) * 0.8)
        train_bg = bg_images[:split_idx]
        val_bg = bg_images[split_idx:]
        
        # Add to training set
        for img_path in train_bg:
            dst_img = combined_path / "train" / "images" / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Create empty label file (no objects = background)
            label_path = combined_path / "train" / "labels" / (img_path.stem + ".txt")
            label_path.touch()
            
            image_count["train"] += 1
        
        # Add to validation set
        for img_path in val_bg:
            dst_img = combined_path / "val" / "images" / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Create empty label file
            label_path = combined_path / "val" / "labels" / (img_path.stem + ".txt")
            label_path.touch()
            
            image_count["val"] += 1
        
        print(f"   ✓ Added {len(train_bg)} background images to training set")
        print(f"   ✓ Added {len(val_bg)} background images to validation set")
    else:
        print(f"   ⚠ No background images found")
    
    print(f"\n✓ Combined dataset created at {combined_path}")
    print(f"  Train: {image_count['train']} images (includes background)")
    print(f"  Val:   {image_count['val']} images (includes background)")
    print(f"  Test:  {image_count['test']} images")
    
    return combined_path, image_count


def create_unified_data_yaml(combined_path, output_yaml="yolo dataset/data_combined.yaml"):
    """
    Create a unified data.yaml for the combined dataset
    """
    # Convert to absolute paths for YOLO
    combined_abs = Path(combined_path).resolve()
    
    data_yaml = {
        "path": str(combined_abs),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,  # Number of classes - cup detection
        "names": ["cup"],  # Class name
    }
    
    output_path = Path(output_yaml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\n✓ Unified data.yaml created at {output_path}")
    print(f"  Path: {data_yaml['path']}")
    print(f"  Classes: {data_yaml['names']}")
    
    return output_path


def train_model(data_yaml, model_name="yolov8n_combined"):
    """
    Train YOLOv8n model on the combined dataset
    """
    print(f"\n{'='*80}")
    print(f"🚀 STARTING YOLOv8n TRAINING")
    print(f"{'='*80}")
    print(f"Dataset: {data_yaml}")
    print(f"Model: yolov8n")
    print(f"{'='*80}\n")
    
    # Load pre-trained YOLOv8n model
    model = YOLO('yolov8n.pt')
    
    # Train with optimized parameters for cup detection
    results = model.train(
        # Dataset and paths
        data=str(data_yaml),
        epochs=200,              # Increased to 200 for better convergence
        imgsz=640,               # Image size
        
        # Batch and device
        batch=16,                # Batch size (reduce to 8 if GPU memory is limited)
        device=0,                # GPU device (0 = first GPU, or use CPU)
        
        # Optimization
        optimizer='SGD',         # SGD with momentum for stable training
        lr0=0.01,                # Initial learning rate
        lrf=0.01,                # Final learning rate ratio
        momentum=0.937,          # Momentum for SGD
        weight_decay=0.0005,     # L2 regularization
        warmup_epochs=3,         # Warmup period
        warmup_momentum=0.8,     # Warmup momentum
        
        # Augmentation for robustness across different areas
        augment=True,            # Enable augmentation
        hsv_h=0.015,            # HSV-Hue augmentation
        hsv_s=0.7,              # HSV-Saturation 
        hsv_v=0.4,              # HSV-Value
        degrees=10,             # Rotation degrees (reduced for better stability)
        translate=0.1,          # Translation
        scale=0.5,              # Scale variation
        flipud=0.5,             # Flip upside down
        fliplr=0.5,             # Flip left-right
        mosaic=1.0,             # Mosaic augmentation
        mixup=0.1,              # Mix-up augmentation
        copy_paste=0.0,         # Copy-paste augmentation
        
        # Loss and validation
        patience=20,            # Early stopping patience
        save=True,              # Save checkpoints
        save_period=10,         # Save every N epochs
        
        # Logging and monitoring
        project='runs/detect',
        name=model_name,
        exist_ok=False,
        verbose=True,
        
        # Hardware optimization
        workers=8,              # Number of workers for data loading
        close_mosaic=15,        # Close mosaic augmentation in final N epochs
    )
    
    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: runs/detect/{model_name}")
    print(f"\nBest weights: runs/detect/{model_name}/weights/best.pt")
    
    return results


def evaluate_model(model_path, data_yaml):
    """
    Evaluate the trained model on test set
    """
    print(f"\n{'='*80}")
    print(f"📊 EVALUATING MODEL")
    print(f"{'='*80}\n")
    
    model = YOLO(model_path)
    metrics = model.val()
    
    print(f"\nEvaluation Metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    
    return metrics


def main():
    """
    Main training pipeline
    """
    print("\n" + "="*80)
    print("🤖 YOLOv8n TRAINING PIPELINE FOR CUP DETECTION")
    print("="*80 + "\n")
    
    # Step 1: Create combined dataset
    print("Step 1: Creating combined dataset from all area datasets...")
    combined_path, image_count = create_combined_dataset()
    
    # Step 2: Create unified data.yaml
    print("\nStep 2: Creating unified data configuration...")
    data_yaml = create_unified_data_yaml(combined_path)
    
    # Step 3: Train model
    print("\nStep 3: Training YOLOv8n model...")
    results = train_model(data_yaml, model_name="yolov8n_areas_with_background")
    
    # Step 4: Evaluate model
    print("\nStep 4: Evaluating trained model...")
    best_model_path = Path("runs/detect/yolov8n_areas_with_background/weights/best.pt")
    if best_model_path.exists():
        metrics = evaluate_model(best_model_path, data_yaml)
    
    print("\n" + "="*80)
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nTrained model location:")
    print(f"  → runs/detect/yolov8n_areas_with_background/weights/best.pt")
    print(f"\nTo use this model in the system:")
    print(f"  1. Copy to: models/cup_detection_latest.pt")
    print(f"  2. Update model_path in controller.py")
    print(f"  3. Restart the application")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
