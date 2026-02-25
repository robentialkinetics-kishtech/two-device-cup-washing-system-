"""
Time tracking and statistics utilities
"""
import time
from datetime import datetime, timedelta
from typing import List, Dict

class TimeTracker:
    """Track cycle times and calculate statistics"""
    
    def __init__(self):
        self.cycle_start = None
        self.operation_times = {}
        self.cycle_times = []
        
    def start_cycle(self):
        """Start timing a cycle"""
        self.cycle_start = time.time()
        self.operation_times = {}
    
    def start_operation(self, operation_name: str):
        """Start timing an operation"""
        self.operation_times[operation_name] = {
            "start": time.time(),
            "end": None,
            "duration": None
        }
    
    def end_operation(self, operation_name: str):
        """End timing an operation"""
        if operation_name in self.operation_times:
            self.operation_times[operation_name]["end"] = time.time()
            start = self.operation_times[operation_name]["start"]
            self.operation_times[operation_name]["duration"] = time.time() - start
    
    def end_cycle(self) -> float:
        """End cycle and return total time"""
        if self.cycle_start:
            cycle_time = time.time() - self.cycle_start
            self.cycle_times.append(cycle_time)
            return cycle_time
        return 0.0
    
    def get_average_cycle_time(self) -> float:
        """Get average cycle time"""
        if not self.cycle_times:
            return 0.0
        return sum(self.cycle_times) / len(self.cycle_times)
    
    def get_estimated_remaining_time(self, remaining_cups: int) -> float:
        """Estimate remaining time based on average"""
        avg = self.get_average_cycle_time()
        return avg * remaining_cups
    
    def get_cups_per_hour(self) -> float:
        """Calculate cups per hour rate"""
        avg = self.get_average_cycle_time()
        if avg > 0:
            return 3600 / avg
        return 0.0
    
    def format_time(self, seconds: float) -> str:
        """Format seconds to readable time string"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            "total_cycles": len(self.cycle_times),
            "average_cycle_time": self.get_average_cycle_time(),
            "min_cycle_time": min(self.cycle_times) if self.cycle_times else 0,
            "max_cycle_time": max(self.cycle_times) if self.cycle_times else 0,
            "cups_per_hour": self.get_cups_per_hour(),
            "total_time": sum(self.cycle_times)
        }
