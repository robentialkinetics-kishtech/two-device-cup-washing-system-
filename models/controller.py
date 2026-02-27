"""
#controller.py
Master Controller - Orchestrates entire washing system
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from datetime import datetime
from typing import Dict, Optional, Tuple
from config.constants import SystemState, WashingMode
from models.robot import ZKBotController
from models.wash_station import WashStationController
from models.sensors import SensorSystem
from models.vision import VisionSystem
from data.storage import DataStorage


class CupWashingController:
    """Master controller orchestrating entire system"""
    
    def __init__(self):
        # Initialize subsystems
        settings = DataStorage.load_settings()
        robot_config = settings.get("robot", {})
        
        self.robot = ZKBotController(
            port=robot_config.get("port", "COM3"),
            baudrate=robot_config.get("baudrate", 115200)
        )
        self.wash_station = WashStationController()
        self.sensors = SensorSystem()
        
        # Initialize vision with trained model
        model_path = "runs\detect\runs\detect\yolov8n_areas_with_background3\weights\best.pt"  # Use latest trained model
        self.vision = VisionSystem(model_path=model_path)
        
        # Load calibration
        self.calibration = DataStorage.load_calibration()
        self.positions = self.calibration.get("positions", {})
        
        # Check if positions are calibrated
        if not self.positions:
            print("⚠ No positions calibrated yet!")
            print("   Go to Developer Mode to teach positions")
        else:
            print(f"📍 Loaded {len(self.positions)} calibrated positions:")
            for name, pos in self.positions.items():
                print(f"   {name}: X={pos['x']:.1f}, Y={pos['y']:.1f}, Z={pos['z']:.1f}")
        
        # System state
        self.state = SystemState.IDLE
        self.washing_mode = WashingMode.SINGLE_CYCLE
        self.target_cups = 1
        self.washed_cups = 0
        self.failed_cups = 0
        self.connected = False
        
        # Settings
        self.arm_speed = robot_config.get("arm_speed", 300)  # Increased from 100 for faster movement
        self.wash_duration = robot_config.get("wash_time", 3)  # Reduced from 10 to 3 seconds
        self.rinse_duration = robot_config.get("rinse_time", 2)  # Reduced from 5 to 2 seconds
        
        # Runtime tracking
        self.is_running = False
        self.error_log = []
        self.start_time = None
        self.cycle_times = []
    
    def connect_robot(self, port: str = "COM3", baudrate: int = 115200) -> Tuple[bool, str]:
        """Connect to robot"""
        self.robot.port = port
        self.robot.baudrate = baudrate
        success = self.robot.connect()
        self.connected = success
        
        if success:
            # Update current position
            self.robot.get_position()
            return True, "Connected successfully"
        else:
            return False, "Connection failed"
    
    def disconnect_robot(self) -> bool:
        """Disconnect from robot"""
        success = self.robot.disconnect()
        self.connected = False
        return success
        
    def initialize(self) -> bool:
        """Initialize all systems"""
        print("🤖 Initializing cup washing system...")
        
        # Connect robot
        if not self.robot.connected:
            success = self.robot.connect()
            if not success:
                self.log_error("Robot connection failed")
                return False
            self.connected = True
        
        # Clear any existing errors
        print("🔄 Clearing errors...")
        self.robot.reset_errors()
        time.sleep(0.05)  # Minimal delay
        
        # Check E-stop
        print("🔍 Checking E-stop...")
        estop_ok, estop_response = self.robot.check_estop()
        if not estop_ok:
            print(f"❌ E-stop check failed: {estop_response}")
            print("   Please check E-stop button and try again")
            # Don't return False - continue anyway
        else:
            print(f"   ✓ E-stop OK: {estop_response}")
        
        # Home robot
        print("🏠 Homing robot...")
        success, response = self.robot.home()
        if not success:
            print(f"❌ Homing failed: {response}")
            self.log_error(f"Homing failed: {response}")
            return False
        
        time.sleep(0.05)  # Minimal delay
        
        # Check sensors
        print("🔍 Checking sensors...")
        if not self.sensors.check_all_sensors():
            print("⚠ Some sensors not ready - continuing anyway")
        
        # Initialize camera (optional) - try web camera first
        try:
            # Try indices: 0 (web cam), 1 (laptop), 2 (USB)
            camera_started = False
            for camera_id in [0, 1, 2]:
                if self.vision.start_camera(camera_id=camera_id):
                    camera_started = True
                    break
            if not camera_started:
                print("⚠ No camera available - vision disabled")
        except Exception as e:
            print(f"⚠ Camera initialization error: {e}")
        
        print("✓ System ready!")
        return True
    
    def shutdown(self):
        """Safely shutdown system"""
        print("🛑 Shutting down system...")
        self.stop_washing()
        self.robot.disconnect()
        self.vision.stop_camera()
        print("✓ System shutdown complete")
    
    def reload_positions(self):
        """Reload positions from calibration file"""
        self.calibration = DataStorage.load_calibration()
        self.positions = self.calibration.get("positions", {})
        print(f"📍 Reloaded {len(self.positions)} positions")
    
    # ═══════════════════════════════════════════════════════════════
    # MOVEMENT OPERATIONS
    # ═══════════════════════════════════════════════════════════════
    
    def move_to(self, position_name: str, feedrate: Optional[int] = None) -> bool:
        """Move arm to named position"""
        if position_name not in self.positions:
            error_msg = f"Position '{position_name}' not calibrated! Go to Developer Mode to teach it."
            self.log_error(error_msg)
            print(f"❌ {error_msg}")
            return False
        
        pos = self.positions[position_name]
        x, y, z = pos["x"], pos["y"], pos["z"]
        feedrate = feedrate or self.arm_speed
        
        print(f"➡️  Moving to '{position_name}': X={x:.1f}, Y={y:.1f}, Z={z:.1f}, F={feedrate}")
        
        success, response = self.robot.move_point_to_point(x, y, z, feedrate)
        
        if not success:
            self.log_error(f"Move failed to {position_name}: {response}")
            self.state = SystemState.ERROR
            return False
        
        return True
    
    # ═══════════════════════════════════════════════════════════════
    # WASHING OPERATIONS
    # ═══════════════════════════════════════════════════════════════
    
    def detect_cup_before_pickup(self, confidence_threshold: float = 0.8, max_wait_frames: int = 200) -> Tuple[bool, str]:  # Changed default from 0.6 to 0.8
        """
        Detect if a cup is present in the pickup area before moving arm
        Uses stable frame detection (needs 8 consecutive frames)
        
        Args:
            confidence_threshold: Minimum confidence for detection (0-1)
            max_wait_frames: Maximum frames to wait for stable detection
        
        Returns:
            Tuple[bool, str]: (cup_detected, message)
        """
        try:
            print("\n🎥 Checking for cup (waiting for stable detection)...")
            self.vision.reset_detection_state()
            
            frame_count = 0
            while frame_count < max_wait_frames:
                # Capture frame from camera
                frame = self.vision.capture_frame()
                if frame is None:
                    frame_count += 1
                    time.sleep(0.005)  # Minimal retry delay
                    continue
                
                # Check for cup with stability counting
                cup_detected, stable_count = self.vision.detect_cup_stable(frame)
                
                # Check if stable detection achieved
                if self.vision.is_stable_detection():
                    cup_pos = self.vision.get_cup_position(frame, confidence_threshold)
                    if cup_pos:
                        confidence = cup_pos.get("confidence", 0)
                        print(f"✓ Cup detected stably! Confidence: {confidence:.2f}")
                        success_msg = f"Cup detected with {confidence:.2f} confidence ({stable_count} frames)"
                        return True, success_msg
                
                frame_count += 1
                time.sleep(0.01)  # Reduced from 0.05 to 0.01 seconds for faster detection
            
            # No stable detection found
            error_msg = "No cup detected in pickup area (timeout)"
            print(f"❌ {error_msg}")
            self.log_error(error_msg)
            return False, error_msg
            
        except Exception as e:
            error_msg = f"Cup detection error: {str(e)}"
            self.log_error(error_msg)
            print(f"❌ {error_msg}")
            return False, error_msg
    
    def pick_cup(self) -> bool:
        """Execute cup pickup sequence"""
        try:
            self.state = SystemState.PICKING_UP
            print("\n📦 Starting cup pickup...")
            
            # Check if positions exist
            if "pickup" not in self.positions:
                raise Exception("Position 'pickup' not calibrated")
            
            # Move to pickup position
            print("  Step 1: Moving to pickup position...")
            if not self.move_to("pickup", feedrate=200):
                return False
            
            # Lower to cup (if position exists)
            if "pickup_lower" in self.positions:
                print("  Step 2: Lowering to cup...")
                if not self.move_to("pickup_lower", feedrate=100):
                    return False
            
            # Activate suction
            print("  Step 3: Activating suction cup...")
            self.robot.pump_on()
            # NO DELAY - move immediately
            
            # Lift with cup
            print("  Step 4: Lifting with cup...")
            if not self.move_to("pickup", feedrate=150):
                return False
            
            print("✓ Cup pickup complete")
            return True
            
        except Exception as e:
            self.log_error(f"Pickup failed: {e}")
            return False
    
    def place_at_wash(self) -> bool:
        """Place cup at wash station"""
        try:
            self.state = SystemState.MOVING_TO_WASH
            print("\n🚿 Moving to wash station...")
            
            if "wash_station" not in self.positions:
                raise Exception("Position 'wash_station' not calibrated")
            
            if not self.move_to("wash_station", feedrate=200):
                return False
            
            # Release cup
            print("  Releasing cup...")
            self.robot.pump_off()
            # NO DELAY - move immediately
            
            print("✓ Cup placed at wash station")
            return True
            
        except Exception as e:
            self.log_error(f"Place at wash failed: {e}")
            return False
    
    def wash_cycle(self, duration: Optional[int] = None) -> bool:
        """Execute washing cycle"""
        try:
            self.state = SystemState.WASHING
            duration = duration or self.wash_duration
            
            print(f"\n🧼 Washing for {duration} seconds...")
            self.wash_station.execute_wash_cycle(duration)
            
            print("✓ Washing complete")
            return True
            
        except Exception as e:
            self.log_error(f"Wash cycle failed: {e}")
            return False
    
    def pick_from_wash(self) -> bool:
        """Pick cup from wash station"""
        try:
            print("\n📦 Picking cup from wash station...")
            
            # Activate suction
            print("  Activating suction...")
            self.robot.pump_on()
            # NO DELAY - move immediately
            
            # Move to safe position if it exists
            if "safe" in self.positions:
                print("  Moving to safe position...")
                if not self.move_to("safe", feedrate=200):
                    return False
            
            print("✓ Cup picked from wash station")
            return True
            
        except Exception as e:
            self.log_error(f"Pick from wash failed: {e}")
            return False
    
    def place_at_rinse(self) -> bool:
        """Place cup at rinse station"""
        try:
            self.state = SystemState.MOVING_TO_RINSE
            print("\n💧 Moving to rinse station...")
            
            if "rinse_station" not in self.positions:
                raise Exception("Position 'rinse_station' not calibrated")
            
            if not self.move_to("rinse_station", feedrate=200):
                return False
            
            print("✓ Cup placed at rinse station")
            return True
            
        except Exception as e:
            self.log_error(f"Place at rinse failed: {e}")
            return False
    
    def rinse_cycle(self, duration: Optional[int] = None) -> bool:
        """Execute rinse cycle"""
        try:
            self.state = SystemState.RINSING
            duration = duration or self.rinse_duration
            
            print(f"\n💦 Rinsing for {duration} seconds...")
            self.wash_station.execute_rinse_cycle(duration)
            
            print("✓ Rinsing complete")
            return True
            
        except Exception as e:
            self.log_error(f"Rinse cycle failed: {e}")
            return False
    
    def place_at_stack(self) -> bool:
        """Place cup at stacking area"""
        try:
            self.state = SystemState.MOVING_TO_STACK
            print("\n📚 Moving to stack area...")
            
            # Move to safe position first if it exists
            if "safe" in self.positions:
                if not self.move_to("safe", feedrate=200):
                    return False
            
            if "stack" not in self.positions:
                raise Exception("Position 'stack' not calibrated")
            
            if not self.move_to("stack", feedrate=200):
                return False
            
            # Release cup
            print("  Releasing cup...")
            self.robot.pump_off()
            # NO DELAY - move immediately
            
            self.state = SystemState.STACKING
            print("✓ Cup placed at stack")
            return True
            
        except Exception as e:
            self.log_error(f"Place at stack failed: {e}")
            return False
    
    # ═══════════════════════════════════════════════════════════════
    # COMPLETE CYCLE
    # ═══════════════════════════════════════════════════════════════
    
    def single_cup_cycle(self) -> bool:
        """Complete washing cycle for one cup"""
        cycle_start = time.time()
        
        print("\n" + "="*60)
        print(f"🚀 STARTING CUP #{self.washed_cups + 1} CYCLE")
        print("="*60)
        
        try:
            # Check required positions
            required_positions = ["pickup", "wash_station", "rinse_station", "stack"]
            missing = [p for p in required_positions if p not in self.positions]
            
            if missing:
                raise Exception(f"Missing required positions: {', '.join(missing)}\n"
                              "Please calibrate all positions in Developer Mode first!")
            
            # *** DETECT CUP BEFORE ARM MOVEMENT ***
            cup_detected, detection_msg = self.detect_cup_before_pickup(confidence_threshold=0.5)
            if not cup_detected:
                raise Exception(f"Cup detection failed: {detection_msg}")
            
            # 1. Pick cup
            if not self.pick_cup():
                raise Exception("Pickup failed")
            
            # 2. Place at wash
            if not self.place_at_wash():
                raise Exception("Place at wash failed")
            
            # 3. Wash
            if not self.wash_cycle():
                raise Exception("Wash cycle failed")
            
            # 4. Pick from wash
            if not self.pick_from_wash():
                raise Exception("Pick from wash failed")
            
            # 5. Place at rinse
            if not self.place_at_rinse():
                raise Exception("Place at rinse failed")
            
            # 6. Rinse
            if not self.rinse_cycle():
                raise Exception("Rinse cycle failed")
            
            # 7. Place at stack
            if not self.place_at_stack():
                raise Exception("Place at stack failed")
            
            # Success
            cycle_time = time.time() - cycle_start
            self.cycle_times.append(cycle_time)
            self.washed_cups += 1
            self.state = SystemState.IDLE
            
            print("\n" + "="*60)
            print(f"✅ CUP #{self.washed_cups} COMPLETE - Time: {cycle_time:.1f}s")
            print("="*60)
            
            # Log cycle
            DataStorage.log_wash_cycle({
                "cup_number": self.washed_cups,
                "cycle_time": cycle_time,
                "wash_duration": self.wash_duration,
                "rinse_duration": self.rinse_duration,
                "success": True
            })
            
            return True
            
        except Exception as e:
            cycle_time = time.time() - cycle_start
            self.log_error(str(e))
            self.failed_cups += 1
            self.state = SystemState.ERROR
            
            print("\n" + "="*60)
            print(f"❌ CUP #{self.washed_cups + self.failed_cups} FAILED")
            print(f"   Error: {e}")
            print("="*60)
            
            # Log failed cycle
            DataStorage.log_wash_cycle({
                "cup_number": self.washed_cups + self.failed_cups,
                "cycle_time": cycle_time,
                "success": False,
                "error": str(e)
            })
            
            return False
    def execute_program(self, program_name: str) -> bool:
        """Execute a saved program by name"""
        from data.storage import DataStorage
    
        print(f"\n🎯 Loading program: {program_name}")
    
        # Load program
        program_data = DataStorage.load_program(program_name)
        if not program_data:
            self.log_error(f"Program '{program_name}' not found")
            return False
    
        steps = program_data.get("steps", [])
        if not steps:
            self.log_error(f"Program '{program_name}' has no steps")
            return False
    
        print(f"✓ Loaded {len(steps)} steps")
    
        # Execute each step
        for i, step in enumerate(steps):
            print(f"\n--- Step {i+1}/{len(steps)} ---")
        
            cmd = step.get("cmd", "G01")
        
            try:
                if cmd in ["G00", "G01"]:
                    # Movement
                    x = step.get("x", 0.0)
                    y = step.get("y", 0.0)
                    z = step.get("z", 0.0)
                    feedrate = step.get("feedrate", 100)
                
                    print(f"Moving: X={x:.1f}, Y={y:.1f}, Z={z:.1f}, F={feedrate}")
                
                    if cmd == "G00":
                        success, response = self.robot.move_point_to_point(x, y, z, feedrate)
                    else:
                        success, response = self.robot.move_linear(x, y, z, feedrate)
                
                    if not success:
                        self.log_error(f"Step {i+1} failed: {response}")
                        return False
                
                elif cmd == "GRIPPER":
                    angle = step.get("angle", 90)
                    print(f"Gripper: {angle}°")
                    self.robot.set_gripper_angle(angle)
                
                elif cmd == "PUMP_ON":
                    print("Pump ON")
                    self.robot.pump_on()
                
                elif cmd == "PUMP_OFF":
                    print("Pump OFF")
                    self.robot.pump_off()
                
                elif cmd == "WAIT":
                    pause = step.get("pause", 1.0)
                    print(f"Waiting {pause}s")
                    time.sleep(pause)
            
                # Apply pause ONLY if explicitly set in step
                pause = step.get("pause", 0.0)
                if pause > 0 and cmd != "WAIT":
                    print(f"Pause: {pause}s")
                    time.sleep(pause)
                
            except Exception as e:
                self.log_error(f"Step {i+1} error: {e}")
                return False
    
        print(f"\n✅ Program '{program_name}' complete!")
        return True


    def single_cup_cycle_with_program(self, program_name: str) -> bool:
        """Execute washing cycle using a saved program"""
        cycle_start = time.time()
    
        print("\n" + "="*60)
        print(f"🚀 STARTING CUP #{self.washed_cups + 1} CYCLE")
        print(f"   Using program: {program_name}")
        print("="*60)
    
        try:
            # *** DETECT CUP BEFORE ARM MOVEMENT ***
            cup_detected, detection_msg = self.detect_cup_before_pickup(confidence_threshold=0.8)  # Increased from 0.5 to reduce false positives
            if not cup_detected:
                raise Exception(f"Cup detection failed: {detection_msg}")
            
            # Execute the program
            if not self.execute_program(program_name):
                raise Exception(f"Program '{program_name}' failed")
        
            # Success
            cycle_time = time.time() - cycle_start
            self.cycle_times.append(cycle_time)
            self.washed_cups += 1
            self.state = SystemState.IDLE
        
            print("\n" + "="*60)
            print(f"✅ CUP #{self.washed_cups} COMPLETE - Time: {cycle_time:.1f}s")
            print("="*60)
        
            # Log cycle
            DataStorage.log_wash_cycle({
                "cup_number": self.washed_cups,
                "cycle_time": cycle_time,
                "program": program_name,
                "success": True
            })
        
            return True
        
        except Exception as e:
            cycle_time = time.time() - cycle_start
            self.log_error(str(e))
            self.failed_cups += 1
            self.state = SystemState.ERROR
        
            print("\n" + "="*60)
            print(f"❌ CUP #{self.washed_cups + self.failed_cups} FAILED")
            print(f"   Error: {e}")
            print("="*60)
        
            # Log failed cycle
            DataStorage.log_wash_cycle({
                "cup_number": self.washed_cups + self.failed_cups,
                "cycle_time": cycle_time,
                "program": program_name,
                "success": False,
                "error": str(e)
            })
        
            return False

    # ═══════════════════════════════════════════════════════════════
    # CONTROL
    # ════════════════
    def start_washing(self, mode: WashingMode, target_cups: int = 10):
        """Start washing operation"""
        self.washing_mode = mode
        self.target_cups = target_cups if mode == WashingMode.FIXED_COUNT else None
        self.is_running = True
        self.washed_cups = 0
        self.failed_cups = 0
        self.start_time = datetime.now()
        self.cycle_times = []
        
        print("\n" + "="*60)
        print(f"🚀 STARTING WASHING OPERATION")
        print(f"   Mode: {mode.value}")
        print(f"   Target: {target_cups if mode == WashingMode.FIXED_COUNT else '∞'}")
        print("="*60)
    
    def stop_washing(self):
        """Stop washing operation"""
        self.is_running = False
        self.robot.emergency_stop()
        self.wash_station.stop_washing()
        self.wash_station.stop_rinsing()
        self.state = SystemState.IDLE
        
        print("\n🛑 Washing stopped by user")
    
    def emergency_stop(self):
        """Emergency stop"""
        self.stop_washing()
        self.state = SystemState.EMERGENCY_STOP
        print("\n🚨 EMERGENCY STOP ACTIVATED")
    
    # ═══════════════════════════════════════════════════════════════
    # STATUS & LOGGING
    # ═══════════════════════════════════════════════════════════════
    
    def get_status(self) -> Dict:
        """Get complete system status"""
        elapsed_time = 0
        if self.start_time:
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        avg_cycle_time = sum(self.cycle_times) / len(self.cycle_times) if self.cycle_times else 0
        
        return {
            "state": self.state.value,
            "is_running": self.is_running,
            "washing_mode": self.washing_mode.value,
            "washed_cups": self.washed_cups,
            "failed_cups": self.failed_cups,
            "target_cups": self.target_cups,
            "elapsed_time": elapsed_time,
            "avg_cycle_time": avg_cycle_time,
            "arm_speed": self.arm_speed,
            "wash_duration": self.wash_duration,
            "rinse_duration": self.rinse_duration,
            "sensors": self.sensors.get_status_report(),
            "recent_errors": self.error_log[-5:] if self.error_log else [],
            "positions_calibrated": len(self.positions) > 0
        }
    
    def log_error(self, message: str):
        """Log error message"""
        error = {
            "message": message,
            "state": self.state.value,
            "timestamp": datetime.now().isoformat()
        }
        self.error_log.append(message)
        DataStorage.log_error(error)
        print(f"❌ ERROR: {message}")
