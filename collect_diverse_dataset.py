import cv2
import os
from pathlib import Path
from datetime import datetime

def collect_diverse_data(cam_index=1, samples_per_condition=10):
    """
    Collect diverse training data for better model generalization.
    Helps reduce false positives from camera movement and background changes.
    """
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)
    
    if not cap.isOpened():
        print(f"❌ Could not open camera {cam_index}")
        return
    
    # Create output directories
    base_dir = Path("diverse_dataset")
    (base_dir / "with_cup").mkdir(parents=True, exist_ok=True)
    (base_dir / "without_cup").mkdir(parents=True, exist_ok=True)
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║        DIVERSE TRAINING DATA COLLECTION                        ║
    ║    This helps improve model generalization and reduce          ║
    ║    false positives from camera movement                        ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Collect images WITH cup in different positions/angles
    print(f"\n📸 PHASE 1: Collecting {samples_per_condition} images WITH CUP")
    print("   Position cup at different angles, distances, and locations")
    print("   Press SPACE to capture, Q to skip to next phase")
    count = 0
    while count < samples_per_condition:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture WITH CUP (press SPACE)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            img_path = base_dir / "with_cup" / f"cup_{count:03d}_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(str(img_path), frame)
            print(f"   ✓ Saved: with_cup/{img_path.name}")
            count += 1
        elif key == ord('q'):
            break
    
    # Collect images WITHOUT cup (negative examples) 
    print(f"\n📸 PHASE 2: Collecting {samples_per_condition} images WITHOUT CUP")
    print("   Move camera to different locations, angles, and lighting")
    print("   Press SPACE to capture, Q to finish")
    count = 0
    while count < samples_per_condition:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture WITHOUT CUP (press SPACE)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            img_path = base_dir / "without_cup" / f"no_cup_{count:03d}_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(str(img_path), frame)
            print(f"   ✓ Saved: without_cup/{img_path.name}")
            count += 1
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Data collection complete!")
    print(f"   Images saved to: {base_dir.absolute()}")
    print(f"\nNext steps:")
    print(f"   1. Manually annotate images in 'diverse_dataset/with_cup' using labelImg or similar")
    print(f"   2. Merge with existing training data")
    print(f"   3. Retrain model with augmentation for better generalization")

if __name__ == "__main__":
    collect_diverse_data(cam_index=1, samples_per_condition=20)
