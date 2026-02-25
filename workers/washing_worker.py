"""
Background worker thread for washing operations
"""
from PyQt5.QtCore import QThread, pyqtSignal
import time
from config.constants import WashingMode

class WashingWorker(QThread):
    """Background thread for washing cycles"""
    
    # Signals
    status_updated = pyqtSignal(dict)
    cycle_complete = pyqtSignal()
    cup_washed = pyqtSignal(int)  # Emits cup number
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int)  # Progress percentage
    
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.is_running = False
    
    def run(self):
        """Main washing loop"""
        self.is_running = True
        
        while self.is_running and self.controller.is_running:
            try:
                # Execute single cup cycle
                success = self.controller.single_cup_cycle()
                
                if success:
                    # Emit signals
                    self.cup_washed.emit(self.controller.washed_cups)
                    self.status_updated.emit(self.controller.get_status())
                    
                    # Calculate progress
                    if self.controller.target_cups:
                        progress = int((self.controller.washed_cups / self.controller.target_cups) * 100)
                        self.progress_updated.emit(progress)
                    
                    # Check if target reached
                    if self.controller.washing_mode == WashingMode.FIXED_COUNT:
                        if self.controller.washed_cups >= self.controller.target_cups:
                            self.controller.is_running = False
                            self.cycle_complete.emit()
                            break
                    
                    elif self.controller.washing_mode == WashingMode.SINGLE_CYCLE:
                        self.controller.is_running = False
                        self.cycle_complete.emit()
                        break
                
                else:
                    # Error occurred
                    error_msg = f"Cup {self.controller.washed_cups + self.controller.failed_cups} failed"
                    self.error_occurred.emit(error_msg)
                    self.status_updated.emit(self.controller.get_status())
                
                # Small delay between cycles
                time.sleep(0.5)
                
            except Exception as e:
                self.error_occurred.emit(str(e))
                self.controller.is_running = False
                break
        
        # Final status update
        self.status_updated.emit(self.controller.get_status())
    
    def stop(self):
        """Stop washing thread"""
        self.is_running = False
        self.controller.stop_washing()
        self.wait()  # Wait for thread to finish
