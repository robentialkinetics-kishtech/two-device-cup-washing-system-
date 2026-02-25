"""
Input validation utilities
"""
import re
from typing import Optional

class Validators:
    """Input validation methods"""
    
    @staticmethod
    def validate_username(username: str) -> tuple[bool, str]:
        """Validate username format"""
        if not username:
            return False, "Username cannot be empty"
        
        if len(username) < 3:
            return False, "Username must be at least 3 characters"
        
        if len(username) > 20:
            return False, "Username must be less than 20 characters"
        
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            return False, "Username can only contain letters, numbers, and underscores"
        
        return True, "Valid"
    
    @staticmethod
    def validate_password(password: str) -> tuple[bool, str]:
        """Validate password strength"""
        if not password:
            return False, "Password cannot be empty"
        
        if len(password) < 6:
            return False, "Password must be at least 6 characters"
        
        return True, "Valid"
    
    @staticmethod
    def validate_coordinate(value: str) -> tuple[bool, float]:
        """Validate coordinate input"""
        try:
            coord = float(value)
            if -500 <= coord <= 500:  # Reasonable robot workspace
                return True, coord
            return False, 0.0
        except ValueError:
            return False, 0.0
    
    @staticmethod
    def validate_speed(speed: int) -> tuple[bool, int]:
        """Validate speed parameter"""
        if 1 <= speed <= 500:
            return True, speed
        return False, 100
    
    @staticmethod
    def validate_duration(duration: int) -> tuple[bool, int]:
        """Validate time duration"""
        if 1 <= duration <= 300:  # Max 5 minutes
            return True, duration
        return False, 10
    
    @staticmethod
    def validate_program_name(name: str) -> tuple[bool, str]:
        """Validate program name"""
        if not name:
            return False, "Program name cannot be empty"
        
        if len(name) > 50:
            return False, "Program name too long"
        
        if not re.match(r'^[a-zA-Z0-9_\-\s]+$', name):
            return False, "Program name contains invalid characters"
        
        return True, "Valid"
