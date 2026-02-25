"""
JSON data persistence layer - handles all file I/O operations
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

class DataStorage:
    """Manages JSON file operations for settings, programs, and logs"""
    
    @staticmethod
    def load_json(filepath: str, default: Dict = None) -> Dict:
        """Load JSON file with error handling"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"⚠ File not found: {filepath}, using defaults")
                if default:
                    DataStorage.save_json(filepath, default)
                return default or {}
        except Exception as e:
            print(f"✗ Error loading {filepath}: {e}")
            return default or {}
    
    @staticmethod
    def save_json(filepath: str, data: Dict) -> bool:
        """Save data to JSON file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"✗ Error saving {filepath}: {e}")
            return False
    
    @staticmethod
    def load_settings() -> Dict:
        """Load system settings"""
        from config.constants import SETTINGS_FILE
        default_settings = {
            "robot": {
                "port": "COM3",
                "baudrate": 9600,
                "arm_speed": 100,
                "wash_time": 10,
                "rinse_time": 5,
                "brush_speed": 150,
                "water_flow": 100
            },
            "user": {
                "username": "admin",
                "password_hash": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
                "role": "admin",
                "last_login": None,
                "session_active": False
            },
            "system": {
                "first_run": True,
                "calibrated": False,
                "total_cups_washed": 0,
                "total_runtime_hours": 0.0,
                "maintenance_due": False
            },
            "ui": {
                "last_mode": "single_cycle",
                "last_target_cups": 10,
                "theme": "dark"
            }
        }
        return DataStorage.load_json(SETTINGS_FILE, default_settings)
    
    @staticmethod
    def save_settings(settings: Dict) -> bool:
        """Save system settings"""
        from config.constants import SETTINGS_FILE
        return DataStorage.save_json(SETTINGS_FILE, settings)
    
    @staticmethod
    def load_calibration() -> Dict:
        """Load calibration data"""
        from config.constants import CALIBRATION_FILE
    
        default_calibration = {
            "positions": {},  # Empty - user must teach positions
            "offsets": {},
            "calibration_date": None,
            "calibrated_by": None,
            "notes": "Use Developer Mode to teach and save positions"
        }
    
        return DataStorage.load_json(CALIBRATION_FILE, default_calibration)

    
    @staticmethod
    def save_calibration(calibration: Dict) -> bool:
        """Save calibration data"""
        from config.constants import CALIBRATION_FILE
        calibration["calibration_date"] = datetime.now().isoformat()
        return DataStorage.save_json(CALIBRATION_FILE, calibration)
    
    @staticmethod
    def load_program(program_name: str) -> Optional[Dict]:
        """Load a specific program"""
        from config.constants import PROGRAMS_DIR
        filepath = os.path.join(PROGRAMS_DIR, f"{program_name}.json")
        return DataStorage.load_json(filepath)
    
    @staticmethod
    def save_program(program_name: str, program_data: Dict) -> bool:
        """Save a program"""
        from config.constants import PROGRAMS_DIR
        filepath = os.path.join(PROGRAMS_DIR, f"{program_name}.json")
        program_data["last_modified"] = datetime.now().isoformat()
        return DataStorage.save_json(filepath, program_data)
    
    @staticmethod
    def list_programs() -> List[str]:
        """List all available programs"""
        from config.constants import PROGRAMS_DIR
        try:
            if not os.path.exists(PROGRAMS_DIR):
                os.makedirs(PROGRAMS_DIR, exist_ok=True)
                return []
            
            programs = [f.replace('.json', '') for f in os.listdir(PROGRAMS_DIR) 
                       if f.endswith('.json')]
            return sorted(programs)
        except:
            return []
    
    @staticmethod
    def delete_program(program_name: str) -> bool:
        """Delete a program"""
        from config.constants import PROGRAMS_DIR
        filepath = os.path.join(PROGRAMS_DIR, f"{program_name}.json")
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
            return False
        except Exception as e:
            print(f"✗ Error deleting program: {e}")
            return False
    
    @staticmethod
    def log_wash_cycle(cycle_data: Dict) -> bool:
        """Log a washing cycle to history"""
        from config.constants import WASH_LOG_FILE
        
        # Load existing logs
        logs = DataStorage.load_json(WASH_LOG_FILE, {"cycles": []})
        
        # Add timestamp
        cycle_data["timestamp"] = datetime.now().isoformat()
        
        # Append new cycle
        logs["cycles"].append(cycle_data)
        
        # Keep only last 1000 cycles
        if len(logs["cycles"]) > 1000:
            logs["cycles"] = logs["cycles"][-1000:]
        
        return DataStorage.save_json(WASH_LOG_FILE, logs)
    
    @staticmethod
    def log_error(error_data: Dict) -> bool:
        """Log an error"""
        from config.constants import ERROR_LOG_FILE
        
        errors = DataStorage.load_json(ERROR_LOG_FILE, {"errors": []})
        error_data["timestamp"] = datetime.now().isoformat()
        errors["errors"].append(error_data)
        
        # Keep only last 500 errors
        if len(errors["errors"]) > 500:
            errors["errors"] = errors["errors"][-500:]
        
        return DataStorage.save_json(ERROR_LOG_FILE, errors)
