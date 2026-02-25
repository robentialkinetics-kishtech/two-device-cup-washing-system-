"""
robot.py
Robot Model - ZKBot Communication
"""
import serial
import time
from typing import Tuple, Dict, Optional
import re

class ZKBotController:
    """ZKBot robot arm controller"""
    
    def __init__(self, port: str = "COM3", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.connected = False
        
        # Current position tracking
        self.current_position = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0
        }
    
    def connect(self) -> Tuple[bool, str]:
        """Connect to robot"""
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=2.0,
                write_timeout=2.0
            )
            time.sleep(0.5)  # Reduced from 2 to 0.5 seconds for faster connection
            
            self.connected = True
            print(f"✓ Connected to ZKBot on {self.port}")
            return True, "Connected successfully"
            
        except serial.SerialException as e:
            self.connected = False
            error_msg = f"Connection failed: {str(e)}"
            print(f"❌ {error_msg}")
            return False, error_msg
        except Exception as e:
            self.connected = False
            error_msg = f"Unexpected error: {str(e)}"
            print(f"❌ {error_msg}")
            return False, error_msg
    
    def disconnect(self) -> bool:
        """Disconnect from robot"""
        try:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
            self.connected = False
            print("✓ Disconnected from ZKBot")
            return True
        except Exception as e:
            print(f"❌ Disconnect error: {e}")
            return False
    
    def send_command(self, command: str, wait_for_response: bool = True, timeout: float = 3.0) -> Tuple[bool, str]:
        """
        Send G-code command to robot using correct protocol format.
        
        Returns:
            Tuple[bool, str]: (success, response/error_message)
        """
        if not self.connected or not self.serial_connection:
            return False, "Not connected"
        
        try:
            # Send command exactly as formatted
            data = command.encode('utf-8')
            written = self.serial_connection.write(data)
            print(f"Sent: {command} | bytes: {written}")
            
            if not wait_for_response:
                return True, "ok"
            
            # Minimal wait for controller to process
            time.sleep(0.05)
            
            # Read response
            response = self.serial_connection.read(100)
            response_str = response.decode('utf-8', errors='ignore').strip().lower()
            print(f"Reply: {response}")
            
            # Parse response
            if "ok" in response_str:
                return True, response_str
            elif "error" in response_str:
                return False, response_str
            elif not response_str:
                # No response - assume success for movement commands
                return True, "ok (no response)"
            else:
                return True, response_str
            
        except Exception as e:
            error_msg = f"Send error: {str(e)}"
            print(f"❌ {error_msg}")
            return False, error_msg
    
    def home(self) -> Tuple[bool, str]:
        """Home the robot (G28)"""
        print("🏠 Homing robot...")
        cmd = "0x550xAA G28 0xAA0x55"
        success, response = self.send_command(cmd, timeout=10.0)
        
        if success:
            self.current_position = {"x": 0.0, "y": 0.0, "z": 0.0}
            print("✓ Homing complete")
        
        return success, response
    
    def move_point_to_point(self, x: float, y: float, z: float, feedrate: int = 100) -> Tuple[bool, str]:
        """
        Point-to-point movement (G00)
        
        Args:
            x, y, z: Target coordinates in mm
            feedrate: Movement speed
        """
        cmd = self.build_xyz_move_command(x, y, z, feedrate, speed_override=1.0, move_type="G00")
        success, response = self.send_command(cmd, timeout=5.0)
        
        if success:
            self.current_position = {"x": x, "y": y, "z": z}
        
        return success, response
    
    def move_linear(self, x: float, y: float, z: float, feedrate: int = 100) -> Tuple[bool, str]:
        """
        Linear movement (G01)
        
        Args:
            x, y, z: Target coordinates in mm
            feedrate: Movement speed
        """
        cmd = self.build_xyz_move_command(x, y, z, feedrate, speed_override=1.0, move_type="G01")
        success, response = self.send_command(cmd, timeout=5.0)
        
        if success:
            self.current_position = {"x": x, "y": y, "z": z}
        
        return success, response
    
    def move_offset(self, dx: float, dy: float, dz: float, feedrate: int = 100) -> Tuple[bool, str]:
        """
        Move relative to current position
        
        Args:
            dx, dy, dz: Offset in mm
            feedrate: Movement speed
        """
        new_x = self.current_position["x"] + dx
        new_y = self.current_position["y"] + dy
        new_z = self.current_position["z"] + dz
        
        return self.move_linear(new_x, new_y, new_z, feedrate)
    
    def set_gripper_angle(self, angle: int) -> Tuple[bool, str]:
        """
        Set gripper to specific angle (0-180)
        
        Args:
            angle: Gripper angle in degrees (0=closed, 180=open)
        """
        cmd = self.build_gripper_command(angle)
        return self.send_command(cmd)
    
    def gripper_open(self) -> Tuple[bool, str]:
        """Open gripper fully"""
        return self.set_gripper_angle(180)
    
    def gripper_close(self) -> Tuple[bool, str]:
        """Close gripper fully"""
        return self.set_gripper_angle(0)
    
    def reset_errors(self) -> Tuple[bool, str]:
        """Reset any robot errors"""
        cmd = "0x550xAA M999 0xAA0x55"
        return self.send_command(cmd)
    
    def check_estop(self) -> Tuple[bool, str]:
        """Check E-stop status"""
        cmd = "0x550xAA M122 0xAA0x55"
        return self.send_command(cmd)
    
    def pump_on(self) -> Tuple[bool, str]:
        """Activate vacuum pump"""
        cmd = "0x550xAA M03 0xAA0x55"
        return self.send_command(cmd)
    
    def pump_off(self) -> Tuple[bool, str]:
        """Deactivate vacuum pump"""
        cmd = "0x550xAA M05 0xAA0x55"
        return self.send_command(cmd)
    
    def emergency_stop(self) -> Tuple[bool, str]:
        """Emergency stop"""
        cmd = "0x550xAA M112 0xAA0x55"
        return self.send_command(cmd, wait_for_response=False)
    
    def build_gripper_command(self, angle: int) -> str:
        """
        Build a G06 command for DO-0 (4th axis gripper).
        Format: 0x550xAA G06 D7 S1 A<angle> 0xAA0x55

        Args:
            angle: Gripper angle (0-180 degrees)
        
        Returns:
            Formatted command string
        """
        angle = max(0, min(180, angle))  # Clamp to valid range
        gcode = f"G06 D7 S1 A{angle}"
        frame = f"0x550xAA {gcode} 0xAA0x55"
        print(f"FRAME: {frame}")
        return frame
    
    def build_xyz_move_command(self, x: float, y: float, z: float, feedrate: int = 100, 
                               speed_override: float = 1.0, move_type: str = "G01") -> str:
        """
        Build a G00/G01 XYZ move frame with speed override applied.
        Format: 0x550xAA G01 X... Y... Z... F... 0xAA0x55
        
        Args:
            x, y, z: Target coordinates in mm
            feedrate: Base movement speed (1-500 mm/min)
            speed_override: Speed multiplier (0.1 to 2.0), default 1.0 = 100%
            move_type: "G00" for rapid, "G01" for linear (default: G01)
        
        Returns:
            Formatted command string
        """
        # Validate move type
        move_type = move_type.upper()
        if move_type not in ["G00", "G01"]:
            move_type = "G01"
        
        # Build parts list
        parts = [move_type]
        parts.append(f"X{x}")
        parts.append(f"Y{y}")
        parts.append(f"Z{z}")
        
        # Apply speed override to feedrate (integer, 1-500 mm/min)
        effective_speed = int(feedrate * speed_override)
        effective_speed = max(1, min(500, effective_speed))  # Clamp to valid range
        parts.append(f"F{effective_speed}")
        
        # Build frame with spaces
        gcode = " ".join(parts)
        frame = f"0x550xAA {gcode} 0xAA0x55"
        print(f"FRAME: {frame} (Override: {speed_override*100:.0f}%)")
        return frame
    
    def get_position(self) -> Optional[Dict[str, float]]:
        """
        Get current position
        
        Returns:
            Dictionary with x, y, z coordinates or None if failed
        """
        success, response = self.send_command("0x550xAA P01 0xAA0x55")
        
        if success and response:
            # Parse response for coordinates
            # Expected format: "X:123.45 Y:67.89 Z:12.34"
            try:
                x_match = re.search(r'X[:\s]*([-\d.]+)', response)
                y_match = re.search(r'Y[:\s]*([-\d.]+)', response)
                z_match = re.search(r'Z[:\s]*([-\d.]+)', response)
                
                if x_match and y_match and z_match:
                    position = {
                        "x": float(x_match.group(1)),
                        "y": float(y_match.group(1)),
                        "z": float(z_match.group(1))
                    }
                    self.current_position = position
                    return position
            except Exception as e:
                print(f"⚠ Position parsing error: {e}")
        
        # Return cached position if query failed
        return self.current_position
    
    def emergency_stop(self) -> Tuple[bool, str]:
        """Emergency stop"""
        cmd = "0x550xAA M112 0xAA0x55"
        return self.send_command(cmd, wait_for_response=False)
    
    def __del__(self):
        """Cleanup on deletion"""
        if self.connected:
            self.disconnect()
