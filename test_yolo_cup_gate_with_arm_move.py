import cv2
import time
from ultralytics import YOLO
from models.robot import ZKBotController
from data.storage import DataStorage
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CONFIG
CAM_INDEX = 1  # Change if needed
ROI = (220, 120, 200, 200)  # x, y, w, h
STABLE_FRAMES_REQUIRED = 8
COOLDOWN_SEC = 3.0
TARGET_X, TARGET_Y, TARGET_Z = 20, -20, -20
FEEDRATE = 10
MODEL_PATH = 'runs/detect/train/weights/best.pt'  # Path to trained YOLOv8n model from yolo dataset


def open_camera(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"❌ Could not open camera at index {index} with CAP_DSHOW. Trying default backend...")
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"❌ Could not open camera at index {index}. Try changing CAM_INDEX at the top of the script (0, 1, 2, ...)")
        return None
    return cap


def main():
    # Camera
    cap = open_camera(CAM_INDEX)
    if cap is None:
        print("❌ Could not open camera. Try CAM_INDEX=1 or 2.")
        return

    # Load YOLOv8n model
    model = YOLO(MODEL_PATH)

    # Robot
    settings = DataStorage.load_settings()
    robot_cfg = settings.get("robot", {})
    port = robot_cfg.get("port", "COM3")
    baudrate = robot_cfg.get("baudrate", 9600)
    print(f"[DEBUG] Using robot port={port}, baudrate={baudrate}")
    robot = ZKBotController(port=port, baudrate=baudrate)
    ok, msg = robot.connect()
    print(f"[DEBUG] Robot connect() returned: ok={ok}, msg={msg}")
    if isinstance(ok, tuple):
        ok = ok[0]
    if not ok:
        print(f"❌ Robot connect failed: {msg}")
        cap.release()
        return
    if not robot.connected:
        print("❌ Robot not marked as connected after connect(). Check hardware and port.")
        cap.release()
        return
    print("[DEBUG] Resetting robot errors...")
    robot.reset_errors()
    time.sleep(0.5)
    print("[DEBUG] Homing robot...")
    home_ok, home_msg = robot.home()
    print(f"[DEBUG] home() returned: ok={home_ok}, msg={home_msg}")
    if not home_ok:
        print(f"❌ Robot homing failed: {home_msg}")
        cap.release()
        robot.disconnect()
        return
    print("✅ Robot ready for movement!")
    print("➡️ When cup is detected in ROI stably, robot will move.")

    stable_count = 0
    last_fire = 0.0
    CONF_THRESH = 0.6  # Increased threshold to reduce false positives
    iou_thresh = 0.5  # IoU threshold for NMS

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Frame read failed")
            break

        # Run YOLO detection on full frame with higher confidence threshold
        results = model(frame, verbose=False, conf=CONF_THRESH, iou=iou_thresh)
        boxes = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []
        print("[DEBUG] Full frame detections (conf >= {:.2f}):".format(CONF_THRESH))
        for d in boxes:
            print(f"  class={int(d[5])} conf={d[4]:.2f} bbox={d[:4]}")
        cups = [d for d in boxes if int(d[5]) == 0 and d[4] >= CONF_THRESH]
        cup_detected = len(cups) > 0

        if cup_detected:
            stable_count += 1
            cv2.putText(frame, f"Cup detected! ({stable_count}/{STABLE_FRAMES_REQUIRED})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            stable_count = 0
            cv2.putText(frame, "No cup", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        if stable_count >= STABLE_FRAMES_REQUIRED and (time.time() - last_fire) > COOLDOWN_SEC:
            print("✅ Cup detected -> MOVING ARM")
            last_fire = time.time()
            stable_count = 0
            success, resp = robot.move_point_to_point(TARGET_X, TARGET_Y, TARGET_Z, FEEDRATE)
            print(f"[DEBUG] move_point_to_point returned: success={success}, resp={resp}")
            if hasattr(robot, 'get_position'):
                pos = robot.get_position()
                print(f"[DEBUG] Robot position after move: {pos}")

        cv2.imshow("Cup Detection Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    robot.disconnect()

if __name__ == "__main__":
    main()
