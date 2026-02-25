# Computer Vision ROI Test - Quick Guide

## Overview
These test scripts detect cups in a selected ROI (Region of Interest) and trigger arm movement when a cup is detected.

## Files Created

1. **`test_vision_roi.py`** - Full-featured test with YOLOv8 support
   - Uses YOLOv8 for object detection (if available)
   - Falls back to color-based detection
   - More accurate but requires ultralytics package

2. **`test_vision_roi_simple.py`** - Simple color-based test
   - No YOLO dependency
   - Quick to test
   - Uses white/light color detection

## Quick Start

### Option 1: Simple Test (Recommended for first test)
```bash
python test_vision_roi_simple.py
```

### Option 2: Full Test (with YOLOv8)
```bash
python test_vision_roi.py
```

## How It Works

1. **Connect to Robot**: Attempts to connect to COM3 at 115200 baud
2. **Initialize Camera**: Opens your Lenovo FHD webcam
3. **Select ROI**: 
   - Click and drag to select the region where cups will appear
   - Press SPACE/ENTER to confirm
   - Press ESC to cancel
4. **Detection Loop**:
   - Continuously monitors the ROI for cup presence
   - Requires multiple consecutive detections (default: 5-10 frames)
   - When cup is detected, moves arm to: X=20, Y=-20, Z=-20

## Configuration

### Change Target Position
Edit the `target_position` in the script:
```python
self.target_position = {
    "x": 20.0,    # Change these values
    "y": -20.0,
    "z": -20.0
}
```

### Adjust Detection Sensitivity
- **Simple version**: Change `min_area` in `detect_cup_in_roi()` method
- **Full version**: Change `self.detection_threshold` and `self.required_detections`

### Change Camera ID
If your webcam is not camera 0:
```python
self.camera_id = 1  # or 2, 3, etc.
```

## Controls During Test

- **'q'** - Quit the test
- **'r'** - Reset detection count
- **'m'** - Manually trigger arm movement (for testing)

## Troubleshooting

### Camera Not Opening
- Check if another application is using the camera
- Try changing `camera_id` (0, 1, 2, etc.)
- On Windows, check Device Manager for camera status

### Robot Not Connecting
- Check COM port (default: COM3)
- Verify baudrate (default: 115200)
- Check USB cable connection
- The script will continue in simulation mode if connection fails

### Detection Not Working
- **ROI too small**: Make sure ROI covers the area where cups appear
- **Lighting**: Ensure good lighting in the ROI area
- **Color detection**: Adjust HSV color ranges in `detect_cup_in_roi()` for your cup color
- **Area threshold**: Adjust `min_area` value (larger = bigger objects only)

### False Detections
- Increase `required_detections` (more consecutive detections needed)
- Adjust color detection thresholds
- Improve lighting conditions

## Integration with Main System

Once tested, you can integrate this into the main system:

1. Add ROI selection to Developer Mode
2. Save ROI coordinates to calibration file
3. Integrate detection into `controller.py` before `pick_cup()`
4. Add vision check to washing cycle

## Example Integration Code

```python
# In controller.py, before pick_cup():
def check_cup_present(self) -> bool:
    """Check if cup is present in ROI"""
    frame = self.vision.capture_frame()
    if frame is None:
        return False
    
    detected, _ = self.vision.detect_cup_in_roi(frame, self.roi)
    return detected

# In single_cup_cycle():
if not self.check_cup_present():
    print("⚠ No cup detected - waiting...")
    time.sleep(1)
    continue  # Skip this cycle
```

## Next Steps

1. Run the simple test first to verify camera and ROI selection
2. Adjust detection parameters for your setup
3. Test arm movement with detected cups
4. Integrate into main washing cycle
5. Add ROI calibration to Developer Mode

## Notes

- The ROI coordinates are in pixels (relative to camera frame)
- For FHD camera (1920x1080), adjust detection area thresholds accordingly
- Color-based detection works best with white/light colored cups
- For better accuracy, use YOLOv8 version with trained model
