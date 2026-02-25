"""
Computer Vision System - YOLOv8 Object Detection
"""
import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠ YOLOv8 not available - vision features disabled")

class VisionSystem:
    """Computer vision for cup detection and tracking"""
    
    def __init__(self, model_path="runs/detect/train/weights/best.pt"):
        self.model = None
        self.camera = None
        self.is_running = False
        self.current_frame = None
        self.detections = []
        self.annotated_frame = None
        
        # Detection stability tracking
        self.stable_count = 0
        self.stable_frames_required = 8
        self.last_detection_time = 0.0
        self.detection_cooldown = 0.5  # seconds
        
        # Detection thresholds
        self.conf_threshold = 0.85  # Increased from 0.6 to reduce false positives
        self.iou_threshold = 0.5   # Match test file
        
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                print(f"✓ Vision system initialized with {model_path}")
            except Exception as e:
                print(f"⚠ Vision system init failed: {e}")
        else:
            print("⚠ Vision system disabled (ultralytics not installed)")
    
    def start_camera(self, camera_id: int = 0) -> bool:
        """Initialize camera with web camera support (matches test file)"""
        try:
            # Try with CAP_DSHOW first (better for web cameras on Windows)
            self.camera = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            
            if not self.camera.isOpened():
                print(f"⚠ CAP_DSHOW backend failed for camera {camera_id}, trying default...")
                self.camera = cv2.VideoCapture(camera_id)
            
            if self.camera.isOpened():
                self.is_running = True
                print(f"✓ Camera {camera_id} initialized successfully")
                return True
            else:
                print(f"✗ Could not open camera at index {camera_id}")
                print(f"   Try changing camera_id parameter (0, 1, 2, ...)")
                return False
        except Exception as e:
            print(f"✗ Camera init failed: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera"""
        self.is_running = False
        if self.camera:
            self.camera.release()
            print("✓ Camera stopped")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Get frame from camera"""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.current_frame = frame
                return frame
        return None
    
    def detect_objects(self, frame: np.ndarray, conf_threshold: Optional[float] = None) -> List:
        """Run YOLOv8 detection with configurable threshold"""
        if not self.model or not YOLO_AVAILABLE:
            return []
        
        try:
            conf = conf_threshold or self.conf_threshold
            results = self.model(frame, conf=conf, iou=self.iou_threshold, verbose=False)
            self.detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []
            return self.detections
        except Exception as e:
            print(f"⚠ Detection error: {e}")
            return []
    
    def get_cup_position(self, frame: np.ndarray, conf_threshold: Optional[float] = None) -> Optional[Dict]:
        """Detect cup and return center position"""
        detections = self.detect_objects(frame, conf_threshold)
        
        # Find cups (class_id = 0, assuming cup is first class)
        for box in detections:
            class_id = int(box[5])
            confidence = float(box[4])
            
            # Check if detected object is a cup (class 0)
            if class_id == 0 and confidence >= (conf_threshold or self.conf_threshold):
                x1, y1, x2, y2 = map(int, box[:4])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                return {
                    "x": center_x,
                    "y": center_y,
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "area": (x2 - x1) * (y2 - y1),
                    "confidence": confidence,
                    "class": "cup"
                }
        
        return None
    
    def detect_cup_stable(self, frame: np.ndarray) -> Tuple[bool, int]:
        """
        Detect cup with stability counting (like test file)
        Returns: (cup_detected, stable_count)
        """
        cup_position = self.get_cup_position(frame)
        
        if cup_position is not None:
            self.stable_count += 1
        else:
            self.stable_count = 0
        
        return cup_position is not None, self.stable_count
    
    def is_stable_detection(self) -> bool:
        """Check if detection is stable (enough consecutive frames)"""
        return self.stable_count >= self.stable_frames_required
    
    def reset_detection_state(self):
        """Reset detection counters"""
        self.stable_count = 0
        self.last_detection_time = time.time()
    
    def detect_dirt(self, frame: np.ndarray, roi: tuple = None) -> Dict:
        """Estimate cup cleanliness (simple color-based method)"""
        if roi:
            x1, y1, x2, y2 = roi
            frame = frame[y1:y2, x1:x2]
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define brown/dirt color range
        brown_lower = np.array([10, 100, 100])
        brown_upper = np.array([20, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, brown_lower, brown_upper)
        dirt_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        dirt_percentage = (dirt_pixels / total_pixels) * 100
        
        return {
            "dirt_detected": dirt_percentage > 15,
            "dirt_percentage": dirt_percentage,
            "cleanliness": max(0, 100 - dirt_percentage)
        }
    
    def annotate_frame(self, frame: np.ndarray, show_stable_count: bool = False) -> np.ndarray:
        """Draw detections on frame with optional stability counter"""
        annotated = frame.copy()
        
        # Run detection
        detections = self.detect_objects(frame)
        
        # Draw all detections
        for box in detections:
            x1, y1, x2, y2 = map(int, box[:4])
            confidence = float(box[4])
            class_id = int(box[5])
            
            # Draw bounding box (green)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with confidence
            label = f"Cup: {confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show stability count if any
        if show_stable_count and self.stable_count > 0:
            status_text = f"Stable: {self.stable_count}/{self.stable_frames_required}"
            cv2.putText(annotated, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        self.annotated_frame = annotated
        return annotated
    
    def get_display_frame(self, show_detections: bool = True) -> Optional[np.ndarray]:
        """Get frame for UI display (with optional annotations)"""
        if self.current_frame is None:
            return None
        
        if show_detections:
            return self.annotate_frame(self.current_frame, show_stable_count=True)
        else:
            return self.current_frame
    
    def pixel_to_robot_coords(self, pixel_x: int, pixel_y: int, 
                             calibration: Dict) -> tuple:
        """Convert pixel coordinates to robot coordinates"""
        # TODO: Implement camera calibration matrix transformation
        # This requires camera calibration data (intrinsic/extrinsic parameters)
        
        # Simplified transformation (needs proper calibration)
        robot_x = (pixel_x - calibration.get("center_x", 320)) * calibration.get("scale_x", 0.5)
        robot_y = (pixel_y - calibration.get("center_y", 240)) * calibration.get("scale_y", 0.5)
        
        return robot_x, robot_y
