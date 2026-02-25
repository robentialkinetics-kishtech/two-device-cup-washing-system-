"""
#wash_station.py
Wash Station Controller
Manages brush motor, water pump, and washing logic
"""
import time
from typing import Optional

class WashStationController:
    """Control wash station hardware (motors, pumps, sensors)"""
    
    def __init__(self):
        self.brush_speed = 150  # 0-255 PWM
        self.water_flow = 100   # 0-255 PWM
        self.is_washing = False
        self.is_rinsing = False
        
        self.wash_duration = 10  # seconds
        self.rinse_duration = 5  # seconds
        
        self.total_wash_time = 0.0
        self.cycles_completed = 0
    
    def start_washing(self, wash_time: int = 10, brush_speed: int = 150):
        """Start washing cycle"""
        self.wash_duration = wash_time
        self.brush_speed = brush_speed
        self.is_washing = True
        
        print(f"🚿 Washing started - Duration: {wash_time}s, Brush: {brush_speed}")
        
        # TODO: Send PWM signals to actual motor controllers
        # Example: GPIO/Arduino/Motor Driver commands
        # self.set_brush_motor_pwm(brush_speed)
        # self.set_water_pump_pwm(self.water_flow)
        
        return True
    
    def stop_washing(self):
        """Stop washing cycle"""
        self.is_washing = False
        self.brush_speed = 0
        
        print("🛑 Washing stopped")
        
        # TODO: Stop motors
        # self.set_brush_motor_pwm(0)
        # self.set_water_pump_pwm(0)
        
        return True
    
    def start_rinsing(self, rinse_time: int = 5):
        """Start rinse cycle"""
        self.rinse_duration = rinse_time
        self.is_rinsing = True
        
        print(f"💧 Rinsing started - Duration: {rinse_time}s")
        
        # TODO: Control water pump for rinsing (no brush)
        # self.set_water_pump_pwm(self.water_flow)
        
        return True
    
    def stop_rinsing(self):
        """Stop rinse cycle"""
        self.is_rinsing = False
        
        print("💧 Rinsing stopped")
        
        # TODO: Stop water pump
        # self.set_water_pump_pwm(0)
        
        return True
    
    def set_brush_speed(self, speed: int):
        """Set brush motor speed (0-255)"""
        self.brush_speed = max(0, min(255, speed))
        
        if self.is_washing:
            # TODO: Update motor speed in real-time
            # self.set_brush_motor_pwm(self.brush_speed)
            pass
    
    def set_water_flow(self, flow: int):
        """Set water pump flow rate (0-255)"""
        self.water_flow = max(0, min(255, flow))
        
        if self.is_washing or self.is_rinsing:
            # TODO: Update pump speed
            # self.set_water_pump_pwm(self.water_flow)
            pass
    
    def execute_wash_cycle(self, duration: int) -> bool:
        """Complete washing cycle with timing"""
        self.start_washing(duration, self.brush_speed)
        time.sleep(duration)
        self.stop_washing()
        
        self.total_wash_time += duration
        self.cycles_completed += 1
        
        return True
    
    def execute_rinse_cycle(self, duration: int) -> bool:
        """Complete rinse cycle with timing"""
        self.start_rinsing(duration)
        time.sleep(duration)
        self.stop_rinsing()
        
        return True
    
    def get_status(self) -> dict:
        """Get wash station status"""
        return {
            "is_washing": self.is_washing,
            "is_rinsing": self.is_rinsing,
            "brush_speed": self.brush_speed,
            "water_flow": self.water_flow,
            "total_wash_time": self.total_wash_time,
            "cycles_completed": self.cycles_completed
        }
