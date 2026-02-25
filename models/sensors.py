"""
#sensors.py
Sensor System - Monitor all sensors
"""
from datetime import datetime
from config.constants import SensorStatus

class SensorSystem:
    """Monitor all system sensors"""
    
    def __init__(self):
        self.sensors = {
            "proximity_pickup": SensorStatus.OK,
            "proximity_wash": SensorStatus.OK,
            "proximity_rinse": SensorStatus.OK,
            "proximity_stack": SensorStatus.OK,
            "weight_sensor": SensorStatus.OK,
            "water_level": SensorStatus.OK,
            "brush_motor": SensorStatus.OK,
            "pump": SensorStatus.OK,
            "estop": SensorStatus.OK,
        }
        self.last_check = datetime.now()
        self.sensor_values = {}
    
    def check_sensor(self, sensor_name: str) -> SensorStatus:
        """Check individual sensor status"""
        # TODO: Read actual sensor values from GPIO/ADC/Serial
        # Example for different sensor types:
        
        # Digital sensors (proximity, limit switches)
        # value = GPIO.input(sensor_pin)
        # return SensorStatus.OK if value else SensorStatus.ERROR
        
        # Analog sensors (water level, weight)
        # value = ADC.read(sensor_channel)
        # return SensorStatus.OK if value > threshold else SensorStatus.WARNING
        
        # For now, return OK (simulation mode)
        return self.sensors.get(sensor_name, SensorStatus.ERROR)
    
    def check_all_sensors(self) -> bool:
        """Perform health check on all sensors"""
        all_ok = True
        
        for sensor_name in self.sensors:
            status = self.check_sensor(sensor_name)
            self.sensors[sensor_name] = status
            
            if status != SensorStatus.OK:
                all_ok = False
                print(f"⚠ Sensor issue: {sensor_name} - {status.value}")
        
        self.last_check = datetime.now()
        return all_ok
    
    def get_status_report(self) -> dict:
        """Get complete sensor status report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "all_ok": all(s == SensorStatus.OK for s in self.sensors.values()),
            "sensors": {k: v.value for k, v in self.sensors.items()},
            "last_check": self.last_check.isoformat()
        }
    
    def read_proximity(self, location: str) -> bool:
        """Read proximity sensor at specific location"""
        sensor_key = f"proximity_{location}"
        # TODO: Read actual sensor
        # return GPIO.input(PROXIMITY_PINS[location])
        return True  # Simulation
    
    def read_weight(self) -> float:
        """Read weight sensor value"""
        # TODO: Read from load cell/weight sensor
        # return ADC.read(WEIGHT_SENSOR_CHANNEL) * CALIBRATION_FACTOR
        return 0.0  # Simulation
    
    def read_water_level(self) -> float:
        """Read water level percentage"""
        # TODO: Read water level sensor
        # value = ADC.read(WATER_LEVEL_CHANNEL)
        # return (value / MAX_VALUE) * 100
        return 75.0  # Simulation (75% full)
    
    def check_estop(self) -> bool:
        """Check emergency stop status"""
        # TODO: Read E-stop button
        # return not GPIO.input(ESTOP_PIN)  # Active low
        return False  # Not pressed (simulation)
