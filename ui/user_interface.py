"""
User Interface - Main washing control page
WITH PROGRAM SELECTION SUPPORT
"""
import os
import sys
import time
import cv2
import numpy as np
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QSpinBox, QComboBox, QSlider,
                             QProgressBar, QGroupBox, QGridLayout, QTextEdit,
                             QFrame, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QImage, QPixmap
from config.constants import WashingMode
from utils.time_tracker import TimeTracker
from workers.washing_worker import WashingWorker
from data.storage import DataStorage


class UserInterface(QWidget):
    """Main user washing control interface"""
    
    # Signals
    logout_requested = pyqtSignal()
    developer_mode_requested = pyqtSignal()
    
    def __init__(self, controller, username="User"):
        super().__init__()
        self.controller = controller
        self.username = username
        self.worker = None
        self.time_tracker = TimeTracker()
        
        # Load available programs
        self.available_programs = DataStorage.list_programs()
        self.selected_program = None
        
        if self.available_programs:
            self.selected_program = self.available_programs[0]  # Default to first program
            print(f"✓ Loaded {len(self.available_programs)} programs")
        else:
            print("⚠ No programs found - create one in Developer Mode")
        
        self.initUI()
        self.setup_timers()
        
        # Start camera thread automatically
        QTimer.singleShot(1000, self.start_camera_thread_auto)
    
    def initUI(self):
        """Initialize user interface"""
        main_layout = QVBoxLayout()
        
        # ═══════════════════════════════════════════════════════════
        # TOP BAR
        # ═══════════════════════════════════════════════════════════
        
        top_bar = self.create_top_bar()
        main_layout.addWidget(top_bar)
        
        # ═══════════════════════════════════════════════════════════
        # CAMERA PREVIEW (NEW!)
        # ═══════════════════════════════════════════════════════════
        
        camera_panel = self.create_camera_preview_panel()
        main_layout.addWidget(camera_panel)
        
        # ═══════════════════════════════════════════════════════════
        # CONTENT AREA
        # ═══════════════════════════════════════════════════════════
        
        content_layout = QHBoxLayout()
        
        # Left panel - Status
        left_panel = self.create_status_panel()
        content_layout.addWidget(left_panel, 2)
        
        # Center panel - Controls
        center_panel = self.create_control_panel()
        content_layout.addWidget(center_panel, 3)
        
        # Right panel - Statistics
        right_panel = self.create_statistics_panel()
        content_layout.addWidget(right_panel, 2)
        
        main_layout.addLayout(content_layout)
        
        self.setLayout(main_layout)
    
    def create_top_bar(self):
        """Create top navigation bar"""
        top_bar = QFrame()
        top_bar.setStyleSheet("""
            QFrame {
                background-color: #161b22;
                border-bottom: 2px solid #30363d;
                padding: 10px;
            }
        """)
        
        layout = QHBoxLayout()
        
        # Title
        title = QLabel("🤖 Cup Washing System")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #58a6ff;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # User info
        user_label = QLabel(f"👤 {self.username}")
        user_label.setStyleSheet("color: #8b949e; margin-right: 15px;")
        layout.addWidget(user_label)
        
        # Developer mode button
        dev_btn = QPushButton("⚙️ Developer")
        dev_btn.clicked.connect(self.developer_mode_requested.emit)
        layout.addWidget(dev_btn)
        
        # Logout button
        logout_btn = QPushButton("🚪 Logout")
        logout_btn.clicked.connect(self.on_logout)
        layout.addWidget(logout_btn)
        
        top_bar.setLayout(layout)
        return top_bar
    
    def create_camera_preview_panel(self):
        """Create camera preview panel for cup detection visualization"""
        panel = QGroupBox("📷 CAMERA PREVIEW")
        layout = QHBoxLayout()
        
        # Camera feed label
        self.camera_label = QLabel()
        self.camera_label.setMinimumHeight(200)
        self.camera_label.setMinimumWidth(300)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("""
            background-color: #0d1117;
            border: 2px solid #30363d;
            border-radius: 6px;
        """)
        self.camera_label.setText("📷 Camera initializing...")
        layout.addWidget(self.camera_label, 3)
        
        # Camera info panel
        info_panel = QFrame()
        info_panel.setStyleSheet("background-color: #161b22; border-radius: 6px; padding: 10px;")
        info_layout = QVBoxLayout()
        
        info_layout.addWidget(QLabel("📊 Detection Status:", font=QFont("Arial", 10, QFont.Bold)))
        
        self.cup_status_label = QLabel("🔍 Scanning...")
        self.cup_status_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        info_layout.addWidget(self.cup_status_label)
        
        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setStyleSheet("color: #58a6ff; font-size: 11px;")
        info_layout.addWidget(self.confidence_label)
        
        self.stable_count_label = QLabel("Stable Frames: 0/8")
        self.stable_count_label.setStyleSheet("color: #79c0ff; font-size: 11px;")
        info_layout.addWidget(self.stable_count_label)
        
        info_layout.addSpacing(15)
        
        # Camera control buttons
        cam_btn_layout = QVBoxLayout()
        
        self.camera_toggle_btn = QPushButton("📷 Start Camera")
        self.camera_toggle_btn.setMinimumHeight(40)
        self.camera_toggle_btn.clicked.connect(self.toggle_camera)
        cam_btn_layout.addWidget(self.camera_toggle_btn)
        
        info_layout.addLayout(cam_btn_layout)
        info_layout.addStretch()
        
        info_frame = QWidget()
        info_frame.setLayout(info_layout)
        layout.addWidget(info_frame, 1)
        
        panel.setLayout(layout)
        self.camera_panel = panel
        
        # Camera thread for background capture
        self.camera_thread = None
        self.camera_running = False
        
        return panel
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.camera_running:
            self.start_camera_thread()
            self.camera_toggle_btn.setText("📷 Stop Camera")
            self.camera_toggle_btn.setStyleSheet("background-color: #da3633; color: white;")
        else:
            self.stop_camera_thread()
            self.camera_toggle_btn.setText("📷 Start Camera")
            self.camera_toggle_btn.setStyleSheet("")
    
    def start_camera_thread_auto(self):
        """Auto-start camera thread on UI load"""
        self.start_camera_thread()
    
    def start_camera_thread(self):
        """Start background camera thread"""
        if self.camera_running:
            return
        
        try:
            self.camera_running = True
            self.camera_thread = CameraThread(self.controller)
            self.camera_thread.frame_ready.connect(self.on_camera_frame_ready)
            self.camera_thread.detection_updated.connect(self.on_detection_updated)
            self.camera_thread.start()
            self.add_log("✓ Camera thread started")
        except Exception as e:
            self.camera_running = False
            self.add_log(f"❌ Camera start failed: {e}")
            self.camera_label.setText(f"📷 Camera Error\n{str(e)[:50]}")
    
    def stop_camera_thread(self):
        """Stop background camera thread"""
        if self.camera_thread:
            self.camera_running = False
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_thread = None
        self.camera_label.setText("📷 Camera stopped")
        self.add_log("⏹ Camera stopped")
    
    def on_camera_frame_ready(self, pixmap):
        """Handle new camera frame"""
        if pixmap:
            self.camera_label.setPixmap(pixmap)
    
    def on_detection_updated(self, detection_info):
        """Handle detection status update"""
        if detection_info["cup_detected"]:
            self.cup_status_label.setText(f"✓ Cup Detected")
            self.cup_status_label.setStyleSheet("color: #3fb950; font-size: 11px;")
        else:
            self.cup_status_label.setText(f"✗ No Cup Detected")
            self.cup_status_label.setStyleSheet("color: #f85149; font-size: 11px;")
        
        confidence = detection_info.get("confidence", 0)
        self.confidence_label.setText(f"Confidence: {confidence:.2f}")
        
        stable_count = detection_info.get("stable_count", 0)
        self.stable_count_label.setText(f"Stable Frames: {stable_count}/8")
    
    def create_top_bar(self):
        """Create top navigation bar"""
        top_bar = QFrame()
        top_bar.setStyleSheet("""
            QFrame {
                background-color: #161b22;
                border-bottom: 2px solid #30363d;
                padding: 10px;
            }
        """)
        
        layout = QHBoxLayout()
        
        # Title
        title = QLabel("🤖 Cup Washing System")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #58a6ff;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # User info
        user_label = QLabel(f"👤 {self.username}")
        user_label.setStyleSheet("color: #8b949e; margin-right: 15px;")
        layout.addWidget(user_label)
        
        # Developer mode button
        dev_btn = QPushButton("⚙️ Developer")
        dev_btn.clicked.connect(self.developer_mode_requested.emit)
        layout.addWidget(dev_btn)
        
        # Logout button
        logout_btn = QPushButton("🚪 Logout")
        logout_btn.clicked.connect(self.on_logout)
        layout.addWidget(logout_btn)
        
        top_bar.setLayout(layout)
        return top_bar
    
    def create_status_panel(self):
        """Create status monitoring panel"""
        panel = QGroupBox("📊 SYSTEM STATUS")
        layout = QVBoxLayout()
        
        # Current state
        layout.addWidget(QLabel("Current State:"))
        self.state_label = QLabel("IDLE")
        self.state_label.setObjectName("statusLabel")
        self.state_label.setAlignment(Qt.AlignCenter)
        self.state_label.setStyleSheet("""
            background-color: #238636;
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 15px;
            border-radius: 8px;
        """)
        layout.addWidget(self.state_label)
        
        # Progress bar
        layout.addWidget(QLabel("Progress:", font=QFont("Arial", 10, QFont.Bold)))
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(30)
        layout.addWidget(self.progress_bar)
        
        # Statistics grid
        stats_grid = QGridLayout()
        
        stats_grid.addWidget(QLabel("Washed:"), 0, 0)
        self.washed_label = QLabel("0")
        self.washed_label.setStyleSheet("color: #3fb950; font-size: 18px; font-weight: bold;")
        stats_grid.addWidget(self.washed_label, 0, 1)
        
        stats_grid.addWidget(QLabel("Failed:"), 1, 0)
        self.failed_label = QLabel("0")
        self.failed_label.setStyleSheet("color: #f85149; font-size: 18px; font-weight: bold;")
        stats_grid.addWidget(self.failed_label, 1, 1)
        
        stats_grid.addWidget(QLabel("Target:"), 2, 0)
        self.target_label = QLabel("0")
        self.target_label.setStyleSheet("color: #58a6ff; font-size: 18px; font-weight: bold;")
        stats_grid.addWidget(self.target_label, 2, 1)
        
        layout.addLayout(stats_grid)
        
        # Time info
        layout.addWidget(QLabel("Time Information:", font=QFont("Arial", 10, QFont.Bold)))
        
        time_frame = QFrame()
        time_frame.setStyleSheet("background-color: #161b22; border-radius: 6px; padding: 10px;")
        time_layout = QVBoxLayout()
        
        self.elapsed_time_label = QLabel("Elapsed: 0s")
        self.avg_cycle_label = QLabel("Avg Cycle: 0s")
        self.remaining_time_label = QLabel("Remaining: --")
        
        time_layout.addWidget(self.elapsed_time_label)
        time_layout.addWidget(self.avg_cycle_label)
        time_layout.addWidget(self.remaining_time_label)
        
        time_frame.setLayout(time_layout)
        layout.addWidget(time_frame)
        
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def create_control_panel(self):
        """Create control panel"""
        panel = QGroupBox("⚙️ WASHING CONTROLS")
        layout = QVBoxLayout()
        
        # ─────────────────────────────────────────────────────────
        # PROGRAM SELECTION (NEW!)
        # ─────────────────────────────────────────────────────────
        
        program_group = QGroupBox("📋 Washing Program")
        program_layout = QVBoxLayout()
        
        # Program selector
        program_select_layout = QHBoxLayout()
        program_select_layout.addWidget(QLabel("Program:"))
        
        self.program_selector = QComboBox()
        if self.available_programs:
            self.program_selector.addItems(self.available_programs)
            if self.selected_program:
                self.program_selector.setCurrentText(self.selected_program)
        else:
            self.program_selector.addItem("No programs available")
            self.program_selector.setEnabled(False)
        
        self.program_selector.currentTextChanged.connect(self.on_program_changed)
        program_select_layout.addWidget(self.program_selector)
        
        # Refresh button
        refresh_programs_btn = QPushButton("🔄")
        refresh_programs_btn.setMaximumWidth(40)
        refresh_programs_btn.setToolTip("Refresh program list")
        refresh_programs_btn.clicked.connect(self.refresh_programs)
        program_select_layout.addWidget(refresh_programs_btn)
        
        program_layout.addLayout(program_select_layout)
        
        # Program info
        self.program_info_label = QLabel("No program selected")
        self.program_info_label.setStyleSheet("color: #8b949e; font-size: 10px; font-style: italic;")
        program_layout.addWidget(self.program_info_label)
        
        program_group.setLayout(program_layout)
        layout.addWidget(program_group)
        
        # ─────────────────────────────────────────────────────────
        # MODE SELECTION
        # ─────────────────────────────────────────────────────────
        
        mode_group = QGroupBox("🔄 Washing Mode")
        mode_layout = QVBoxLayout()
        
        mode_layout.addWidget(QLabel("Select Operation Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Single Cycle (1 cup)",
            "Fixed Count (Custom)",
            "Infinite (Continuous)"
        ])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        
        # Target cups
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Target Cups:"))
        self.target_cups_spin = QSpinBox()
        self.target_cups_spin.setRange(1, 1000)
        self.target_cups_spin.setValue(10)
        self.target_cups_spin.setSuffix(" cups")
        target_layout.addWidget(self.target_cups_spin)
        mode_layout.addLayout(target_layout)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # ─────────────────────────────────────────────────────────
        # SPEED SETTINGS
        # ─────────────────────────────────────────────────────────
        
        speed_group = QGroupBox("⚡ Speed Settings")
        speed_layout = QGridLayout()
        
        # Arm speed
        speed_layout.addWidget(QLabel("Arm Speed:"), 0, 0)
        self.arm_speed_slider = QSlider(Qt.Horizontal)
        self.arm_speed_slider.setRange(50, 300)
        self.arm_speed_slider.setValue(100)
        self.arm_speed_slider.valueChanged.connect(self.on_arm_speed_changed)
        speed_layout.addWidget(self.arm_speed_slider, 0, 1)
        self.arm_speed_label = QLabel("100")
        speed_layout.addWidget(self.arm_speed_label, 0, 2)
        
        # Wash time
        speed_layout.addWidget(QLabel("Wash Time:"), 1, 0)
        self.wash_time_slider = QSlider(Qt.Horizontal)
        self.wash_time_slider.setRange(3, 30)
        self.wash_time_slider.setValue(10)
        self.wash_time_slider.valueChanged.connect(self.on_wash_time_changed)
        speed_layout.addWidget(self.wash_time_slider, 1, 1)
        self.wash_time_label = QLabel("10s")
        speed_layout.addWidget(self.wash_time_label, 1, 2)
        
        # Rinse time
        speed_layout.addWidget(QLabel("Rinse Time:"), 2, 0)
        self.rinse_time_slider = QSlider(Qt.Horizontal)
        self.rinse_time_slider.setRange(2, 15)
        self.rinse_time_slider.setValue(5)
        self.rinse_time_slider.valueChanged.connect(self.on_rinse_time_changed)
        speed_layout.addWidget(self.rinse_time_slider, 2, 1)
        self.rinse_time_label = QLabel("5s")
        speed_layout.addWidget(self.rinse_time_label, 2, 2)
        
        speed_group.setLayout(speed_layout)
        layout.addWidget(speed_group)
        
        # ─────────────────────────────────────────────────────────
        # ACTION BUTTONS
        # ─────────────────────────────────────────────────────────
        
        buttons_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("▶ START WASHING")
        self.start_btn.setObjectName("startButton")
        self.start_btn.setMinimumHeight(60)
        self.start_btn.clicked.connect(self.on_start_washing)
        buttons_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹ STOP")
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.setMinimumHeight(50)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.on_stop_washing)
        buttons_layout.addWidget(self.stop_btn)
        
        self.emergency_btn = QPushButton("🚨 EMERGENCY STOP")
        self.emergency_btn.setObjectName("emergencyButton")
        self.emergency_btn.setMinimumHeight(50)
        self.emergency_btn.clicked.connect(self.on_emergency_stop)
        buttons_layout.addWidget(self.emergency_btn)
        
        layout.addLayout(buttons_layout)
        
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def create_statistics_panel(self):
        """Create statistics panel"""
        panel = QGroupBox("📈 STATISTICS & LOG")
        layout = QVBoxLayout()
        
        # Rate display
        rate_frame = QFrame()
        rate_frame.setStyleSheet("background-color: #161b22; border-radius: 6px; padding: 15px;")
        rate_layout = QVBoxLayout()
        
        rate_title = QLabel("Production Rate")
        rate_title.setStyleSheet("color: #8b949e; font-size: 12px;")
        rate_layout.addWidget(rate_title)
        
        self.rate_label = QLabel("0")
        self.rate_label.setStyleSheet("color: #58a6ff; font-size: 32px; font-weight: bold;")
        self.rate_label.setAlignment(Qt.AlignCenter)
        rate_layout.addWidget(self.rate_label)
        
        rate_unit = QLabel("cups/hour")
        rate_unit.setStyleSheet("color: #8b949e; font-size: 12px;")
        rate_unit.setAlignment(Qt.AlignCenter)
        rate_layout.addWidget(rate_unit)
        
        rate_frame.setLayout(rate_layout)
        layout.addWidget(rate_frame)
        
        # Activity log
        layout.addWidget(QLabel("Activity Log:", font=QFont("Arial", 10, QFont.Bold)))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(300)
        layout.addWidget(self.log_text)
        
        # Clear log button
        clear_btn = QPushButton("🗑️ Clear Log")
        clear_btn.clicked.connect(self.log_text.clear)
        layout.addWidget(clear_btn)
        
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel

    # ═══════════════════════════════════════════════════════════════
    # PROGRAM MANAGEMENT (NEW!)
    # ═══════════════════════════════════════════════════════════════
    
    def on_program_changed(self, program_name):
        """Handle program selection change"""
        if program_name and program_name != "No programs available":
            self.selected_program = program_name
            
            # Load program info
            program_data = DataStorage.load_program(program_name)
            if program_data:
                steps_count = len(program_data.get("steps", []))
                description = program_data.get("description", "")
                self.program_info_label.setText(f"{steps_count} steps - {description}")
            else:
                self.program_info_label.setText("No program info available")
            
            self.add_log(f"✓ Selected program: {program_name}")
            print(f"✓ Selected program: {program_name}")
        else:
            self.selected_program = None
            self.program_info_label.setText("No program selected")
    
    def refresh_programs(self):
        """Refresh list of available programs"""
        self.available_programs = DataStorage.list_programs()
        self.program_selector.clear()
        
        if self.available_programs:
            self.program_selector.addItems(self.available_programs)
            self.program_selector.setEnabled(True)
            
            # Select first program
            self.selected_program = self.available_programs[0]
            self.program_selector.setCurrentText(self.selected_program)
            
            self.add_log(f"✓ Loaded {len(self.available_programs)} programs")
            print(f"✓ Loaded {len(self.available_programs)} programs")
        else:
            self.program_selector.addItem("No programs available")
            self.program_selector.setEnabled(False)
            self.selected_program = None
            self.program_info_label.setText("Create programs in Developer Mode")
            
            self.add_log("⚠ No programs found")
            print("⚠ No programs found - create one in Developer Mode")

    # ═══════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ═══════════════════════════════════════════════════════════════
    
    def on_mode_changed(self, index):
        """Handle mode change"""
        self.target_cups_spin.setEnabled(index == 1)  # Enable only for Fixed Count
    
    def on_arm_speed_changed(self, value):
        """Handle arm speed change"""
        self.arm_speed_label.setText(str(value))
        self.controller.arm_speed = value
    
    def on_wash_time_changed(self, value):
        """Handle wash time change"""
        self.wash_time_label.setText(f"{value}s")
        self.controller.wash_duration = value
    
    def on_rinse_time_changed(self, value):
        """Handle rinse time change"""
        self.rinse_time_label.setText(f"{value}s")
        self.controller.rinse_duration = value
    
    def on_start_washing(self):
        """Start washing operation"""
        # Check if program is selected
        if not self.selected_program:
            QMessageBox.warning(self, "No Program",
                              "Please select a washing program first!\n\n"
                              "Go to Developer Mode to create a program.")
            return
        
        # Verify program exists
        program_data = DataStorage.load_program(self.selected_program)
        if not program_data:
            QMessageBox.critical(self, "Error",
                               f"Program '{self.selected_program}' not found!\n"
                               "Please refresh the program list.")
            return
        
        # Get mode
        mode_index = self.mode_combo.currentIndex()
        modes = [WashingMode.SINGLE_CYCLE, WashingMode.FIXED_COUNT, WashingMode.INFINITE]
        mode = modes[mode_index]
        
        target = self.target_cups_spin.value()
        
        # Confirm
        if mode == WashingMode.INFINITE:
            reply = QMessageBox.question(self, "Confirm",
                                        f"Start infinite washing cycle?\n"
                                        f"Using program: {self.selected_program}",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
        
        # Start controller
        self.controller.start_washing(mode=mode, target_cups=target)
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.target_label.setText(str(target) if mode == WashingMode.FIXED_COUNT else "∞")
        
        # Start worker thread with program
        self.worker = WashingWorkerWithProgram(self.controller, self.selected_program)
        self.worker.status_updated.connect(self.on_status_update)
        self.worker.cup_washed.connect(self.on_cup_washed)
        self.worker.cycle_complete.connect(self.on_cycle_complete)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.start()
        
        # Start time tracking
        self.time_tracker.start_cycle()
        
        self.add_log(f"✓ Washing started with program: {self.selected_program}")
    
    def on_stop_washing(self):
        """Stop washing operation"""
        reply = QMessageBox.question(self, "Confirm",
                                    "Stop washing operation?",
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.worker:
                self.worker.stop()
            
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            
            self.add_log("⏹ Washing stopped by user")
    
    def on_emergency_stop(self):
        """Emergency stop"""
        self.controller.emergency_stop()
        if self.worker:
            self.worker.stop()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.state_label.setText("EMERGENCY STOP")
        self.state_label.setStyleSheet("""
            background-color: #da3633;
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 15px;
            border-radius: 8px;
        """)
        
        self.add_log("🚨 EMERGENCY STOP ACTIVATED")
        
        QMessageBox.warning(self, "Emergency Stop", "System has been emergency stopped!")
    
    def on_status_update(self, status):
        """Handle status update from worker"""
        # Update state
        state = status["state"].upper().replace("_", " ")
        self.state_label.setText(state)
        
        # Update colors based on state
        if "error" in status["state"].lower():
            color = "#da3633"
        elif "washing" in status["state"].lower() or "rinsing" in status["state"].lower():
            color = "#f0883e"
        elif "moving" in status["state"].lower():
            color = "#58a6ff"
        elif "idle" in status["state"].lower():
            color = "#238636"
        else:
            color = "#6e7681"
        
        self.state_label.setStyleSheet(f"""
            background-color: {color};
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 15px;
            border-radius: 8px;
        """)
        
        # Update counters
        self.washed_label.setText(str(status["washed_cups"]))
        self.failed_label.setText(str(status["failed_cups"]))
        
        # Update time
        elapsed = status.get("elapsed_time", 0)
        avg_cycle = status.get("avg_cycle_time", 0)
        self.elapsed_time_label.setText(f"Elapsed: {self.time_tracker.format_time(elapsed)}")
        self.avg_cycle_label.setText(f"Avg Cycle: {self.time_tracker.format_time(avg_cycle)}")
        
        # Calculate remaining time
        if status["target_cups"] and status["washed_cups"] < status["target_cups"]:
            remaining_cups = status["target_cups"] - status["washed_cups"]
            remaining_time = avg_cycle * remaining_cups
            self.remaining_time_label.setText(f"Remaining: {self.time_tracker.format_time(remaining_time)}")
        else:
            self.remaining_time_label.setText("Remaining: --")
        
        # Update rate
        if avg_cycle > 0:
            rate = 3600 / avg_cycle
            self.rate_label.setText(f"{rate:.1f}")
    
    def on_cup_washed(self, cup_number):
        """Handle cup washed event"""
        self.add_log(f"✓ Cup #{cup_number} washed successfully")
    
    def on_cycle_complete(self):
        """Handle cycle completion"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        
        self.add_log("✓ Washing cycle complete!")
        
        QMessageBox.information(self, "Complete", 
                               f"Washing cycle complete!\n\n"
                               f"Total cups washed: {self.controller.washed_cups}\n"
                               f"Failed: {self.controller.failed_cups}")
    
    def on_error(self, error_msg):
        """Handle error"""
        self.add_log(f"❌ ERROR: {error_msg}")
        
        # Show message box for cup detection errors
        if "cup detection" in error_msg.lower() or "no cup" in error_msg.lower():
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            
            QMessageBox.warning(self, "Cup Not Found", 
                               f"⚠ Cup Detection Failed\n\n"
                               f"{error_msg}\n\n"
                               f"Please place a cup in the pickup area and try again.",
                               QMessageBox.Ok)
    
    def on_logout(self):
        """Handle logout"""
        reply = QMessageBox.question(self, "Logout",
                                    "Are you sure you want to logout?",
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.controller.is_running:
                QMessageBox.warning(self, "Warning", 
                                  "Please stop washing operation before logging out")
                return
            
            self.logout_requested.emit()
    
    def add_log(self, message):
        """Add message to activity log"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def setup_timers(self):
        """Setup update timers"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(500)  # Update every 500ms
    
    def update_display(self):
        """Periodic display update"""
        if self.controller.is_running:
            # Force status update
            status = self.controller.get_status()
            self.on_status_update(status)


# ═══════════════════════════════════════════════════════════════
# WASHING WORKER WITH PROGRAM SUPPORT
# ═══════════════════════════════════════════════════════════════

class WashingWorkerWithProgram(QThread):
    """Worker thread for washing operations using saved programs"""
    
    status_updated = pyqtSignal(dict)
    cup_washed = pyqtSignal(int)
    cycle_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, controller, program_name):
        super().__init__()
        self.controller = controller
        self.program_name = program_name
        self.running = True
    
    def run(self):
        """Main worker loop"""
        try:
            while self.running and self.controller.is_running:
                # Check if target reached
                if (self.controller.washing_mode == WashingMode.FIXED_COUNT and 
                    self.controller.washed_cups >= self.controller.target_cups):
                    break
                
                # Execute program
                success = self.controller.single_cup_cycle_with_program(self.program_name)
                
                if not success:
                    self.error_occurred.emit("Cup cycle failed!")
                    break
                
                # Update status
                status = self.controller.get_status()
                self.status_updated.emit(status)
                self.cup_washed.emit(self.controller.washed_cups)
                
                # Update progress
                if self.controller.target_cups:
                    progress = int((self.controller.washed_cups / self.controller.target_cups) * 100)
                    self.progress_updated.emit(progress)
                
                time.sleep(0.5)
            
            # Complete
            self.controller.is_running = False
            self.cycle_complete.emit()
            
        except Exception as e:
            self.error_occurred.emit(str(e))
            self.controller.is_running = False
    
    def stop(self):
        """Stop worker"""
        self.running = False
        self.controller.stop_washing()


# ═══════════════════════════════════════════════════════════════
# CAMERA THREAD FOR LIVE PREVIEW
# ═══════════════════════════════════════════════════════════════

class CameraThread(QThread):
    """Background thread for camera capture and detection"""
    
    frame_ready = pyqtSignal(object)  # Emits QPixmap
    detection_updated = pyqtSignal(dict)  # Emits detection info
    
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.running = True
    
    def run(self):
        """Main camera loop"""
        try:
            # Start camera if not already running
            # Try web camera (index 0) first, then fallback to others
            if not self.controller.vision.is_running:
                camera_started = False
                # Try indices in order: 0 (web cam), 1 (laptop), 2 (USB)
                for camera_id in [0, 1, 2]:
                    print(f"[Camera Thread] Trying camera index {camera_id}...")
                    if self.controller.vision.start_camera(camera_id=camera_id):
                        camera_started = True
                        print(f"[Camera Thread] ✓ Using camera index {camera_id}")
                        break
                
                if not camera_started:
                    print("[Camera Thread] ⚠ No camera available - running in preview mode")
                    return
            
            while self.running:
                try:
                    # Capture frame
                    frame = self.controller.vision.capture_frame()
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    
                    # Check for cup with stability
                    cup_detected, stable_count = self.controller.vision.detect_cup_stable(frame)
                    
                    # Get annotated frame for display
                    display_frame = self.controller.vision.annotate_frame(frame, show_stable_count=True)
                    
                    # Resize for display if needed
                    h, w = display_frame.shape[:2]
                    if w > 400:
                        scale = 400 / w
                        display_frame = cv2.resize(display_frame, (int(w * scale), int(h * scale)))
                    
                    # Convert to RGB for PyQt
                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    
                    # Emit frame
                    self.frame_ready.emit(pixmap)
                    
                    # Emit detection info
                    cup_pos = self.controller.vision.get_cup_position(frame)
                    detection_info = {
                        "cup_detected": cup_pos is not None,
                        "confidence": cup_pos.get("confidence", 0) if cup_pos else 0,
                        "stable_count": stable_count
                    }
                    self.detection_updated.emit(detection_info)
                    
                    time.sleep(0.05)  # ~20 FPS
                except Exception as frame_error:
                    print(f"⚠ Frame processing error: {frame_error}")
                    time.sleep(0.1)
        
        except Exception as e:
            print(f"❌ Camera thread error: {e}")
    
    def stop(self):
        """Stop camera thread"""
        self.running = False
        if self.controller.vision:
            self.controller.vision.stop_camera()

