"""
Advanced Manual Image Annotation Tool for YOLOv8 Cup Detection
Features:
- Auto-fit display with preserved aspect ratio
- Multiple boxes per image support
- Visual box counter and saved annotations display
- Undo last box functionality
- Progress tracking
- Keyboard shortcuts help

Controls:
- Click and drag: Draw bounding box around cup
- 's': Save current annotation and continue (stays on same image)
- 'n': Next image
- 'p': Previous image
- 'u': Undo last saved box
- 'r': Reset current drawing
- 'c': Clear all boxes on current image
- 'h': Show help
- 'q': Quit
"""

import cv2
import os
import glob
import json

# ===== UPDATE THESE PATHS =====
IMAGE_DIR = "C:\\zkbot-cup-washing-system\\cup_images"
ANNOTATION_DIR = "C:\\zkbot-cup-washing-system\\cup_images\\annotated"
# ==============================

# Display settings
MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 900

# Global variables
drawing = False
ix, iy = -1, -1
ex, ey = -1, -1
current_box = []
saved_boxes = []  # Boxes saved in current session for this image
image = None
image_copy = None
display_image = None
original_image = None
scale_factor = 1.0
window_name = "Cup Annotation Tool - YOLOv8"

def resize_for_display(img, max_width=MAX_DISPLAY_WIDTH, max_height=MAX_DISPLAY_HEIGHT):
    """Resize image to fit display while maintaining aspect ratio"""
    h, w = img.shape[:2]
    
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale
    
    return img, 1.0

def draw_saved_boxes(img, boxes):
    """Draw all previously saved boxes on the image"""
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow for saved
        cv2.putText(img, "SAVED", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

def draw_info_overlay(img, box_count, image_name):
    """Draw information overlay on image"""
    h, w = img.shape[:2]
    # Semi-transparent overlay at top
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    
    # Text info
    cv2.putText(img, f"Image: {image_name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f"Boxes saved: {box_count}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, "Press 'h' for help", (w-180, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, ex, ey, drawing, display_image, image_copy, current_box, saved_boxes
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            display_image = image_copy.copy()
            draw_saved_boxes(display_image, saved_boxes)
            cv2.rectangle(display_image, (ix, iy), (x, y), (0, 255, 0), 2)  # Green for current
            draw_info_overlay(display_image, len(saved_boxes), param)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ex, ey = x, y
        display_image = image_copy.copy()
        draw_saved_boxes(display_image, saved_boxes)
        cv2.rectangle(display_image, (ix, iy), (ex, ey), (0, 255, 0), 2)
        draw_info_overlay(display_image, len(saved_boxes), param)
        current_box = [ix, iy, ex, ey]

def convert_to_yolo_format(box, img_width, img_height):
    """Convert bounding box to YOLO format (normalized)"""
    x1, y1, x2, y2 = box
    
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)
    
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = abs(x_max - x_min) / img_width
    height = abs(y_max - y_min) / img_height
    
    return x_center, y_center, width, height

def save_annotation(image_path, box, annotation_dir, original_img, class_id=0):
    """Save annotation in YOLO format"""
    img_height, img_width = original_img.shape[:2]
    
    # Validate box
    if len(box) != 4:
        return False
    
    # Check if box has minimum size
    if abs(box[2] - box[0]) < 5 or abs(box[3] - box[1]) < 5:
        print("✗ Box too small, skipped")
        return False
    
    # Scale box coordinates back to original image size
    scaled_box = [
        int(box[0] / scale_factor),
        int(box[1] / scale_factor),
        int(box[2] / scale_factor),
        int(box[3] / scale_factor)
    ]
    
    # Convert to YOLO format
    x_center, y_center, width, height = convert_to_yolo_format(scaled_box, img_width, img_height)
    
    # Get annotation path
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    annotation_path = os.path.join(annotation_dir, f"{image_name}.txt")
    
    # Append annotation
    with open(annotation_path, 'a') as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    return True

def load_existing_annotations(image_path, annotation_dir, original_img):
    """Load existing annotations for the current image"""
    img_height, img_width = original_img.shape[:2]
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    annotation_path = os.path.join(annotation_dir, f"{image_name}.txt")
    
    boxes = []
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, x_c, y_c, w, h = map(float, parts)
                    # Convert YOLO to pixel coordinates
                    x_center = x_c * img_width
                    y_center = y_c * img_height
                    width = w * img_width
                    height = h * img_height
                    
                    x1 = int((x_center - width/2) * scale_factor)
                    y1 = int((y_center - height/2) * scale_factor)
                    x2 = int((x_center + width/2) * scale_factor)
                    y2 = int((y_center + height/2) * scale_factor)
                    
                    boxes.append([x1, y1, x2, y2])
    return boxes

def clear_annotations(image_path, annotation_dir):
    """Clear all annotations for current image"""
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    annotation_path = os.path.join(annotation_dir, f"{image_name}.txt")
    if os.path.exists(annotation_path):
        os.remove(annotation_path)

def show_help():
    """Display help window"""
    help_text = [
        "=== KEYBOARD SHORTCUTS ===",
        "",
        "s - Save current box (stay on image)",
        "n - Next image",
        "p - Previous image", 
        "u - Undo last saved box",
        "r - Reset current drawing",
        "c - Clear all boxes on image",
        "h - Show this help",
        "q - Quit annotation tool",
        "",
        "=== MOUSE ===",
        "Click & Drag - Draw bounding box",
        "",
        "Press any key to close this help..."
    ]
    
    help_img = np.zeros((500, 600, 3), dtype=np.uint8)
    y_pos = 30
    for line in help_text:
        if line.startswith("==="):
            color = (0, 255, 255)
            thickness = 2
        else:
            color = (255, 255, 255)
            thickness = 1
        cv2.putText(help_img, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        y_pos += 30
    
    cv2.imshow("Help", help_img)
    cv2.waitKey(0)
    cv2.destroyWindow("Help")

def main():
    global image, image_copy, display_image, current_box, saved_boxes, scale_factor, original_image
    
    # Import numpy here
    global np
    import numpy as np
    
    # Create annotation directory
    os.makedirs(ANNOTATION_DIR, exist_ok=True)
    
    # Get all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    
    if not image_files:
        print(f"❌ No images found in {IMAGE_DIR}")
        return
    
    image_files.sort()
    total_images = len(image_files)
    print(f"✓ Found {total_images} images")
    print("\n" + "="*50)
    print("Cup Annotation Tool for YOLOv8")
    print("="*50)
    print("Press 'h' at any time for help\n")
    
    # Create window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    img_idx = 0
    
    while 0 <= img_idx < total_images:
        image_path = image_files[img_idx]
        original_image = cv2.imread(image_path)
        
        if original_image is None:
            print(f"❌ Failed to load: {image_path}")
            img_idx += 1
            continue
        
        # Resize for display
        display_image, scale_factor = resize_for_display(original_image)
        image_copy = display_image.copy()
        current_box = []
        
        # Load existing annotations
        saved_boxes = load_existing_annotations(image_path, ANNOTATION_DIR, original_image)
        
        image_name = os.path.basename(image_path)
        cv2.setMouseCallback(window_name, draw_rectangle, image_name)
        
        print(f"\n[{img_idx + 1}/{total_images}] {image_name}")
        print(f"Size: {original_image.shape[1]}x{original_image.shape[0]} | Scale: {scale_factor:.2f} | Existing boxes: {len(saved_boxes)}")
        
        # Initial display
        display_image = image_copy.copy()
        draw_saved_boxes(display_image, saved_boxes)
        draw_info_overlay(display_image, len(saved_boxes), image_name)
        
        while True:
            cv2.imshow(window_name, display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # Save
                if len(current_box) == 4:
                    if save_annotation(image_path, current_box, ANNOTATION_DIR, original_image):
                        print(f"✓ Box saved ({len(saved_boxes) + 1} total)")
                        saved_boxes.append(current_box.copy())
                        current_box = []
                        # Redraw
                        display_image = image_copy.copy()
                        draw_saved_boxes(display_image, saved_boxes)
                        draw_info_overlay(display_image, len(saved_boxes), image_name)
                else:
                    print("✗ Draw a box first (click and drag)")
                    
            elif key == ord('n'):  # Next
                img_idx += 1
                break
                
            elif key == ord('p'):  # Previous
                if img_idx > 0:
                    img_idx -= 1
                    break
                else:
                    print("Already at first image")
                    
            elif key == ord('u'):  # Undo
                if saved_boxes:
                    saved_boxes.pop()
                    # Rewrite annotation file
                    clear_annotations(image_path, ANNOTATION_DIR)
                    for box in saved_boxes:
                        save_annotation(image_path, box, ANNOTATION_DIR, original_image)
                    print(f"↶ Undone. Boxes: {len(saved_boxes)}")
                    # Redraw
                    display_image = image_copy.copy()
                    draw_saved_boxes(display_image, saved_boxes)
                    draw_info_overlay(display_image, len(saved_boxes), image_name)
                else:
                    print("✗ No boxes to undo")
                    
            elif key == ord('r'):  # Reset current
                current_box = []
                display_image = image_copy.copy()
                draw_saved_boxes(display_image, saved_boxes)
                draw_info_overlay(display_image, len(saved_boxes), image_name)
                print("↻ Reset current drawing")
                
            elif key == ord('c'):  # Clear all
                clear_annotations(image_path, ANNOTATION_DIR)
                saved_boxes = []
                current_box = []
                display_image = image_copy.copy()
                draw_info_overlay(display_image, len(saved_boxes), image_name)
                print("🗑 Cleared all boxes")
                
            elif key == ord('h'):  # Help
                show_help()
                
            elif key == ord('q'):  # Quit
                print("\n" + "="*50)
                print("Exiting annotation tool")
                print("="*50)
                cv2.destroyAllWindows()
                return
    
    print(f"\n✓ Completed all {total_images} images!")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
