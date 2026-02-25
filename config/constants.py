"""
System-wide constants and enumerations
"""
from enum import Enum

# ============================================================================
# ENUMERATIONS
# ============================================================================

class WashingMode(Enum):
    """Washing operation modes"""
    SINGLE_CYCLE = "single_cycle"
    FIXED_COUNT = "fixed_count"
    INFINITE = "infinite"

class SystemState(Enum):
    """Robot state machine states"""
    IDLE = "idle"
    DETECTING = "detecting"
    MOVING_TO_PICKUP = "moving_to_pickup"
    PICKING_UP = "picking_up"
    MOVING_TO_WASH = "moving_to_wash"
    WASHING = "washing"
    MOVING_TO_RINSE = "moving_to_rinse"
    RINSING = "rinsing"
    MOVING_TO_STACK = "moving_to_stack"
    STACKING = "stacking"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"

class SensorStatus(Enum):
    """Sensor health status"""
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"

class UserRole(Enum):
    """User access levels"""
    USER = "user"
    DEVELOPER = "developer"
    ADMIN = "admin"

# ============================================================================
# ROBOT CONSTANTS
# ============================================================================

# Serial communication
DEFAULT_PORT = "COM3"
DEFAULT_BAUDRATE = 9600
SERIAL_TIMEOUT = 2

# Movement parameters
DEFAULT_ARM_SPEED = 100  # 1-500
MIN_ARM_SPEED = 1
MAX_ARM_SPEED = 500

# Workspace limits (like your old project)
WORKSPACE_LIMITS = {
    "X": {"min": -400, "max": 400},
    "Y": {"min": -400, "max": 400},
    "Z": {"min": -300, "max": 300}
}

# Home position
HOME_POSITION = {"x": 0, "y": 0, "z": 0}

# Wash parameters
DEFAULT_WASH_TIME = 10  # seconds
MIN_WASH_TIME = 3
MAX_WASH_TIME = 30

DEFAULT_RINSE_TIME = 5  # seconds
MIN_RINSE_TIME = 2
MAX_RINSE_TIME = 15

# Brush motor
DEFAULT_BRUSH_SPEED = 150  # 0-255 PWM
MIN_BRUSH_SPEED = 50
MAX_BRUSH_SPEED = 255

# Water pump
DEFAULT_WATER_FLOW = 100  # 0-255 PWM
MIN_WATER_FLOW = 30
MAX_WATER_FLOW = 255

# Speed override
SPEED_OVERRIDE_PERCENT = 50.0  # Default 50% for safe teaching

# ============================================================================
# UI CONSTANTS
# ============================================================================

# Window dimensions
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900

# Update intervals
UI_UPDATE_INTERVAL = 500  # milliseconds
SENSOR_CHECK_INTERVAL = 1000  # milliseconds

# Login
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "zkbot123"  # Will be hashed
SESSION_TIMEOUT = 3600  # seconds (1 hour)

# ============================================================================
# FILE PATHS
# ============================================================================

DATA_DIR = "data"
PROGRAMS_DIR = "data/programs"
LOGS_DIR = "data/logs"

SETTINGS_FILE = "config/settings.json"
CALIBRATION_FILE = "config/calibration.json"

WASH_LOG_FILE = "data/logs/wash_log.json"
ERROR_LOG_FILE = "data/logs/error_log.json"

DEFAULT_PROGRAM_FILE = "data/programs/default_program.json"

# ============================================================================
# COLORS (for UI styling)
# ============================================================================

COLOR_PRIMARY = "#2ecc71"
COLOR_DANGER = "#e74c3c"
COLOR_WARNING = "#f39c12"
COLOR_INFO = "#3498db"
COLOR_SUCCESS = "#27ae60"
COLOR_BG_DARK = "#1a1a1a"
COLOR_BG_MEDIUM = "#2c3e50"
COLOR_BG_LIGHT = "#34495e"
COLOR_TEXT = "#ecf0f1"
