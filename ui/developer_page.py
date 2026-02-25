"""
#developer_page.py
Developer Page - Program editor and position calibration
OPTIMIZED COMPACT LAYOUT - PRODUCTION VERSION
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QGroupBox, QLineEdit, QComboBox, QSpinBox,
                             QDoubleSpinBox, QMessageBox, QFileDialog,
                             QFrame, QGridLayout, QTextEdit, QListWidget,
                             QSplitter, QScrollArea, QInputDialog)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor
from data.storage import DataStorage
import json
import time
from datetime import datetime

class DeveloperPage(QWidget):
    """Developer mode - Program editor and calibration"""
    
    # Signals
    back_to_user_mode = pyqtSignal()
    
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.current_program = []
        self.current_program_name = "new_program"
        self.calibration = DataStorage.load_calibration()
        
        self.initUI()
        self.load_programs_list()
    
    def initUI(self):
        """Initialize developer UI - COMPACT VERSION"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # ═══════════════════════════════════════════════════════════
        # COMPACT TOP BAR
        # ═══════════════════════════════════════════════════════════
        
        top_bar = self.create_compact_top_bar()
        main_layout.addWidget(top_bar)
        
        # ═══════════════════════════════════════════════════════════
        # CONTENT - THREE PANELS
        # ═══════════════════════════════════════════════════════════
        
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Program list
        left_panel = self.create_program_list_panel()
        splitter.addWidget(left_panel)
        
        # Center: Step editor
        center_panel = self.create_step_editor_panel()
        splitter.addWidget(center_panel)
        
        # Right: Position calibration
        right_panel = self.create_calibration_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([230, 650, 320])
        
        main_layout.addWidget(splitter)
        
        self.setLayout(main_layout)
    
    def create_compact_top_bar(self):
        """Create COMPACT top navigation bar"""
        top_bar = QFrame()
        top_bar.setStyleSheet("""
            QFrame {
                background-color: #161b22;
                border-bottom: 1px solid #30363d;
            }
        """)
        top_bar.setMaximumHeight(45)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("⚙️ Developer Mode")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setStyleSheet("color: #f0883e;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Back button
        back_btn = QPushButton("◀ User Mode")
        back_btn.setMaximumWidth(120)
        back_btn.clicked.connect(self.back_to_user_mode.emit)
        layout.addWidget(back_btn)
        
        top_bar.setLayout(layout)
        return top_bar
    
    def create_program_list_panel(self):
        """Create program list panel - COMPACT"""
        panel = QGroupBox("📁 Programs")
        panel.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 10px; }")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 10, 5, 5)
        layout.setSpacing(5)
        
        # Program list
        self.program_list = QListWidget()
        self.program_list.itemClicked.connect(self.on_program_selected)
        layout.addWidget(self.program_list)
        
        # Buttons - COMPACT
        new_btn = QPushButton("➕ New")
        new_btn.clicked.connect(self.on_new_program)
        layout.addWidget(new_btn)
        
        load_btn = QPushButton("📂 Load")
        load_btn.clicked.connect(self.on_load_program)
        layout.addWidget(load_btn)
        
        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self.on_save_program)
        layout.addWidget(save_btn)
        
        delete_btn = QPushButton("🗑️ Delete")
        delete_btn.setStyleSheet("background-color: #da3633;")
        delete_btn.clicked.connect(self.on_delete_program)
        layout.addWidget(delete_btn)
        
        # Export/Import in one row
        io_layout = QHBoxLayout()
        export_btn = QPushButton("📤 Export")
        export_btn.clicked.connect(self.on_export_program)
        io_layout.addWidget(export_btn)
        
        import_btn = QPushButton("📥 Import")
        import_btn.clicked.connect(self.on_import_program)
        io_layout.addWidget(import_btn)
        
        layout.addLayout(io_layout)
        
        panel.setLayout(layout)
        return panel
    
    def create_step_editor_panel(self):
        """Create step editor panel - COMPACT"""
        panel = QGroupBox("🔧 Step Editor")
        panel.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 10px; }")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 10, 5, 5)
        layout.setSpacing(5)
        
        # Program name - inline
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Program:"))
        self.program_name_input = QLineEdit("new_program")
        name_layout.addWidget(self.program_name_input)
        layout.addLayout(name_layout)
        
        # Step table - COMPACT
        self.step_table = QTableWidget()
        self.step_table.setColumnCount(9)
        self.step_table.setHorizontalHeaderLabels([
            "Step", "Cmd", "X", "Y", "Z", "Feed", "Angle", "Pause", "✓"
        ])
        self.step_table.horizontalHeader().setStretchLastSection(True)
        self.step_table.itemClicked.connect(self.on_step_selected)
        self.step_table.setMaximumHeight(200)
        layout.addWidget(self.step_table)
        
        # Step editor form - COMPACT GRID
        editor_group = QGroupBox("Add/Edit Step")
        editor_layout = QGridLayout()
        editor_layout.setSpacing(5)
        
        # Row 0: Command
        editor_layout.addWidget(QLabel("Cmd:"), 0, 0)
        self.cmd_combo = QComboBox()
        self.cmd_combo.addItems(["G00 (Rapid)", "G01 (Linear)", "GRIPPER", "PUMP_ON", "PUMP_OFF", "WAIT"])
        self.cmd_combo.currentIndexChanged.connect(self.on_command_changed)
        editor_layout.addWidget(self.cmd_combo, 0, 1, 1, 3)
        
        # Row 1: X, Y
        editor_layout.addWidget(QLabel("X:"), 1, 0)
        self.x_input = QDoubleSpinBox()
        self.x_input.setRange(-500, 500)
        self.x_input.setDecimals(1)
        self.x_input.setSuffix("mm")
        editor_layout.addWidget(self.x_input, 1, 1)
        
        editor_layout.addWidget(QLabel("Y:"), 1, 2)
        self.y_input = QDoubleSpinBox()
        self.y_input.setRange(-500, 500)
        self.y_input.setDecimals(1)
        self.y_input.setSuffix("mm")
        editor_layout.addWidget(self.y_input, 1, 3)
        
        # Row 2: Z, Feed
        editor_layout.addWidget(QLabel("Z:"), 2, 0)
        self.z_input = QDoubleSpinBox()
        self.z_input.setRange(-500, 500)
        self.z_input.setDecimals(1)
        self.z_input.setSuffix("mm")
        editor_layout.addWidget(self.z_input, 2, 1)
        
        editor_layout.addWidget(QLabel("Feed:"), 2, 2)
        self.feedrate_input = QSpinBox()
        self.feedrate_input.setRange(1, 500)
        self.feedrate_input.setValue(100)
        editor_layout.addWidget(self.feedrate_input, 2, 3)
        
        # Row 3: Gripper, Pause
        editor_layout.addWidget(QLabel("Gripper:"), 3, 0)
        self.gripper_angle_input = QSpinBox()
        self.gripper_angle_input.setRange(0, 180)
        self.gripper_angle_input.setValue(90)
        self.gripper_angle_input.setSuffix("°")
        self.gripper_angle_input.setEnabled(False)
        editor_layout.addWidget(self.gripper_angle_input, 3, 1)
        
        editor_layout.addWidget(QLabel("Pause:"), 3, 2)
        self.pause_input = QDoubleSpinBox()
        self.pause_input.setRange(0, 10)
        self.pause_input.setDecimals(1)
        self.pause_input.setSuffix("s")
        editor_layout.addWidget(self.pause_input, 3, 3)
        
        # Row 4: Buttons
        btn_layout = QHBoxLayout()
        
        add_btn = QPushButton("➕ Add")
        add_btn.clicked.connect(self.on_add_step)
        btn_layout.addWidget(add_btn)
        
        update_btn = QPushButton("✏️ Update")
        update_btn.clicked.connect(self.on_update_step)
        btn_layout.addWidget(update_btn)
        
        delete_step_btn = QPushButton("🗑️ Delete")
        delete_step_btn.clicked.connect(self.on_delete_step)
        btn_layout.addWidget(delete_step_btn)
        
        editor_layout.addLayout(btn_layout, 4, 0, 1, 4)
        
        editor_group.setLayout(editor_layout)
        layout.addWidget(editor_group)
        
        # Control buttons - COMPACT
        control_layout = QHBoxLayout()
        
        clear_btn = QPushButton("🗑️ Clear")
        clear_btn.clicked.connect(self.on_clear_program)
        control_layout.addWidget(clear_btn)
        
        debug_btn = QPushButton("🔍 Debug")
        debug_btn.clicked.connect(self.debug_show_program)
        control_layout.addWidget(debug_btn)
        
        test_btn = QPushButton("▶️ Test Program")
        test_btn.setStyleSheet("background-color: #238636; font-weight: bold;")
        test_btn.clicked.connect(self.on_test_program)
        control_layout.addWidget(test_btn)
        
        layout.addLayout(control_layout)
        
        panel.setLayout(layout)
        return panel
    
    def create_calibration_panel(self):
        """Create position calibration panel - COMPACT SCROLLABLE"""
        panel = QGroupBox("📍 Position Calibration")
        panel.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 10px; }")
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 10, 5, 5)
        main_layout.setSpacing(5)
        
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Content widget
        content = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        # Current position display - COMPACT
        self.current_pos_label = QLabel("X: 0.0  Y: 0.0  Z: 0.0")
        self.current_pos_label.setStyleSheet("""
            background-color: #161b22; 
            color: #58a6ff; 
            font-family: monospace; 
            font-size: 11px;
            padding: 5px;
            border-radius: 3px;
        """)
        layout.addWidget(self.current_pos_label)
        
        # Manual control - COMPACT
        jog_group = QGroupBox("Manual Control")
        jog_layout = QGridLayout()
        jog_layout.setSpacing(3)
        
        # Jog distance
        jog_layout.addWidget(QLabel("Step:"), 0, 0)
        self.jog_distance = QComboBox()
        self.jog_distance.addItems(["0.1", "1", "5", "10", "50"])
        self.jog_distance.setCurrentText("5")
        jog_layout.addWidget(self.jog_distance, 0, 1, 1, 2)
        
        # Compact jog buttons
        x_minus_btn = QPushButton("X-")
        x_minus_btn.setMaximumWidth(60)
        x_minus_btn.clicked.connect(lambda: self.jog_axis("x", -1))
        jog_layout.addWidget(x_minus_btn, 1, 0)
        
        x_plus_btn = QPushButton("X+")
        x_plus_btn.setMaximumWidth(60)
        x_plus_btn.clicked.connect(lambda: self.jog_axis("x", 1))
        jog_layout.addWidget(x_plus_btn, 1, 2)
        
        y_minus_btn = QPushButton("Y-")
        y_minus_btn.setMaximumWidth(60)
        y_minus_btn.clicked.connect(lambda: self.jog_axis("y", -1))
        jog_layout.addWidget(y_minus_btn, 2, 0)
        
        y_plus_btn = QPushButton("Y+")
        y_plus_btn.setMaximumWidth(60)
        y_plus_btn.clicked.connect(lambda: self.jog_axis("y", 1))
        jog_layout.addWidget(y_plus_btn, 2, 2)
        
        z_minus_btn = QPushButton("Z-")
        z_minus_btn.setMaximumWidth(60)
        z_minus_btn.clicked.connect(lambda: self.jog_axis("z", -1))
        jog_layout.addWidget(z_minus_btn, 3, 0)
        
        z_plus_btn = QPushButton("Z+")
        z_plus_btn.setMaximumWidth(60)
        z_plus_btn.clicked.connect(lambda: self.jog_axis("z", 1))
        jog_layout.addWidget(z_plus_btn, 3, 2)
        
        jog_group.setLayout(jog_layout)
        layout.addWidget(jog_group)
        
        # Gripper control - COMPACT
        gripper_group = QGroupBox("🤏 Gripper")
        gripper_layout = QVBoxLayout()
        gripper_layout.setSpacing(3)
        
        angle_layout = QHBoxLayout()
        self.manual_gripper_angle = QSpinBox()
        self.manual_gripper_angle.setRange(0, 180)
        self.manual_gripper_angle.setValue(90)
        self.manual_gripper_angle.setSuffix("°")
        angle_layout.addWidget(self.manual_gripper_angle)
        
        set_gripper_btn = QPushButton("Set")
        set_gripper_btn.clicked.connect(self.on_set_gripper)
        angle_layout.addWidget(set_gripper_btn)
        
        gripper_layout.addLayout(angle_layout)
        
        # Quick buttons - COMPACT
        quick_layout = QHBoxLayout()
        
        close_btn = QPushButton("0°")
        close_btn.clicked.connect(lambda: self.quick_gripper(0))
        quick_layout.addWidget(close_btn)
        
        half_btn = QPushButton("90°")
        half_btn.clicked.connect(lambda: self.quick_gripper(90))
        quick_layout.addWidget(half_btn)
        
        open_btn = QPushButton("180°")
        open_btn.clicked.connect(lambda: self.quick_gripper(180))
        quick_layout.addWidget(open_btn)
        
        gripper_layout.addLayout(quick_layout)
        
        gripper_group.setLayout(gripper_layout)
        layout.addWidget(gripper_group)
        
        # Home button
        home_btn = QPushButton("🏠 Home")
        home_btn.clicked.connect(self.on_home_robot)
        layout.addWidget(home_btn)
        
        # Position save - COMPACT
        pos_group = QGroupBox("Save Position")
        pos_layout = QVBoxLayout()
        pos_layout.setSpacing(3)
        
        self.position_name_input = QLineEdit()
        self.position_name_input.setPlaceholderText("Position name")
        pos_layout.addWidget(self.position_name_input)
        
        save_pos_btn = QPushButton("💾 Save Current")
        save_pos_btn.clicked.connect(self.on_save_position)
        pos_layout.addWidget(save_pos_btn)
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        # Saved positions list - COMPACT
        layout.addWidget(QLabel("Saved Positions:", font=QFont("Arial", 9, QFont.Bold)))
        
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(4)
        self.positions_table.setHorizontalHeaderLabels(["Name", "X", "Y", "Z"])
        self.positions_table.setMinimumHeight(120)
        self.positions_table.setMaximumHeight(200)
        layout.addWidget(self.positions_table)
        
        # Load positions
        self.load_positions_table()
        
        # Buttons for positions - COMPACT GRID
        pos_btn_layout = QGridLayout()
        pos_btn_layout.setSpacing(3)
        
        goto_btn = QPushButton("➡️ Go")
        goto_btn.setToolTip("Go to selected position")
        goto_btn.clicked.connect(self.on_goto_position)
        pos_btn_layout.addWidget(goto_btn, 0, 0)
        
        update_pos_btn = QPushButton("🔄 Refresh")
        update_pos_btn.setToolTip("Update position display")
        update_pos_btn.clicked.connect(self.update_current_position)
        pos_btn_layout.addWidget(update_pos_btn, 0, 1)
        
        overwrite_btn = QPushButton("✏️ Overwrite")
        overwrite_btn.setToolTip("Overwrite selected position with current position")
        overwrite_btn.clicked.connect(self.on_overwrite_position)
        pos_btn_layout.addWidget(overwrite_btn, 1, 0)
        
        delete_pos_btn = QPushButton("🗑️ Delete")
        delete_pos_btn.setToolTip("Delete selected position")
        delete_pos_btn.setStyleSheet("background-color: #da3633;")
        delete_pos_btn.clicked.connect(self.on_delete_position)
        pos_btn_layout.addWidget(delete_pos_btn, 1, 1)
        
        layout.addLayout(pos_btn_layout)
        
        # Set content widget
        content.setLayout(layout)
        scroll.setWidget(content)
        
        # Add scroll to main layout
        main_layout.addWidget(scroll)
        
        panel.setLayout(main_layout)
        return panel
    
    # ═══════════════════════════════════════════════════════════════
    # PROGRAM MANAGEMENT
    # ═══════════════════════════════════════════════════════════════
    
    def load_programs_list(self):
        """Load list of programs"""
        self.program_list.clear()
        programs = DataStorage.list_programs()
        for program in programs:
            self.program_list.addItem(program)
    
    def on_program_selected(self, item):
        """Handle program selection"""
        program_name = item.text()
        self.load_program(program_name)
    
    def load_program(self, program_name: str):
        """Load program from file"""
        program_data = DataStorage.load_program(program_name)
        if program_data:
            self.current_program_name = program_name
            self.program_name_input.setText(program_name)
            self.current_program = program_data.get("steps", [])
            self.refresh_step_table()
            
            print(f"✓ Program '{program_name}' loaded with {len(self.current_program)} steps")
    
    def on_new_program(self):
        """Create new program"""
        self.current_program = []
        self.current_program_name = "new_program"
        self.program_name_input.setText("new_program")
        self.refresh_step_table()
        print("✓ New program created")
    
    def on_save_program(self):
        """Save current program"""
        program_name = self.program_name_input.text().strip()
        
        if not program_name:
            QMessageBox.warning(self, "Error", "Please enter a program name")
            return
        
        if not self.current_program:
            QMessageBox.warning(self, "Warning", "Program has no steps!")
            return
        
        program_data = {
            "name": program_name,
            "description": f"Program created on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "steps": self.current_program,
            "last_modified": datetime.now().isoformat()
        }
        
        if DataStorage.save_program(program_name, program_data):
            self.current_program_name = program_name
            self.load_programs_list()
            print(f"✓ Program '{program_name}' saved")
            QMessageBox.information(self, "Success", f"Program '{program_name}' saved")
        else:
            QMessageBox.critical(self, "Error", "Failed to save program")
    
    def on_load_program(self):
        """Load selected program"""
        selected = self.program_list.currentItem()
        if selected:
            self.load_program(selected.text())
        else:
            QMessageBox.warning(self, "Warning", "Please select a program")
    
    def on_delete_program(self):
        """Delete selected program"""
        selected = self.program_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select a program")
            return
        
        program_name = selected.text()
        
        reply = QMessageBox.question(self, "Confirm Delete",
                                     f"Delete program '{program_name}'?",
                                     QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            if DataStorage.delete_program(program_name):
                self.load_programs_list()
                print(f"✓ Program '{program_name}' deleted")
                QMessageBox.information(self, "Success", "Program deleted")
            else:
                QMessageBox.critical(self, "Error", "Failed to delete program")
    
    def on_export_program(self):
        """Export program to JSON file"""
        if not self.current_program:
            QMessageBox.warning(self, "Warning", "No program to export")
            return
        
        filename, _ = QFileDialog.getSaveFileName(self, "Export Program", 
                                                  f"{self.current_program_name}.json",
                                                  "JSON Files (*.json)")
        if filename:
            program_data = {
                "name": self.current_program_name,
                "steps": self.current_program,
                "exported": datetime.now().isoformat()
            }
            
            try:
                with open(filename, 'w') as f:
                    json.dump(program_data, f, indent=2)
                print(f"✓ Program exported to {filename}")
                QMessageBox.information(self, "Success", "Program exported")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {e}")
    
    def on_import_program(self):
        """Import program from JSON file"""
        filename, _ = QFileDialog.getOpenFileName(self, "Import Program",
                                                  "", "JSON Files (*.json)")
        if filename:
            try:
                with open(filename, 'r') as f:
                    program_data = json.load(f)
                
                self.current_program = program_data.get("steps", [])
                self.current_program_name = program_data.get("name", "imported_program")
                self.program_name_input.setText(self.current_program_name)
                self.refresh_step_table()
                
                print(f"✓ Program imported from {filename}")
                QMessageBox.information(self, "Success", "Program imported")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Import failed: {e}")
    
    # ═══════════════════════════════════════════════════════════════
    # STEP EDITING
    # ═══════════════════════════════════════════════════════════════
    
    def on_command_changed(self, index):
        """Handle command type change"""
        cmd = self.cmd_combo.currentText()
        
        if "PUMP" in cmd or "WAIT" in cmd:
            self.x_input.setEnabled(False)
            self.y_input.setEnabled(False)
            self.z_input.setEnabled(False)
            self.feedrate_input.setEnabled(False)
            self.gripper_angle_input.setEnabled(False)
            
        elif "GRIPPER" in cmd:
            self.x_input.setEnabled(False)
            self.y_input.setEnabled(False)
            self.z_input.setEnabled(False)
            self.feedrate_input.setEnabled(False)
            self.gripper_angle_input.setEnabled(True)
            
        else:
            self.x_input.setEnabled(True)
            self.y_input.setEnabled(True)
            self.z_input.setEnabled(True)
            self.feedrate_input.setEnabled(True)
            self.gripper_angle_input.setEnabled(False)
    
    def on_step_selected(self, item):
        """Handle step table row selection"""
        row = item.row()
        if row < len(self.current_program):
            step = self.current_program[row]
            
            cmd = step.get("cmd", "G01")
            if cmd == "G00":
                self.cmd_combo.setCurrentText("G00 (Rapid)")
            elif cmd == "G01":
                self.cmd_combo.setCurrentText("G01 (Linear)")
            elif cmd == "GRIPPER":
                self.cmd_combo.setCurrentText("GRIPPER")
            elif cmd == "PUMP_ON":
                self.cmd_combo.setCurrentText("PUMP_ON")
            elif cmd == "PUMP_OFF":
                self.cmd_combo.setCurrentText("PUMP_OFF")
            
            self.x_input.setValue(step.get("x", 0.0))
            self.y_input.setValue(step.get("y", 0.0))
            self.z_input.setValue(step.get("z", 0.0))
            self.feedrate_input.setValue(step.get("feedrate", 100))
            self.gripper_angle_input.setValue(step.get("angle", 90))
            self.pause_input.setValue(step.get("pause", 0.0))
    
    def on_add_step(self):
        """Add new step to program"""
        step = self.get_step_from_inputs()
        self.current_program.append(step)
        self.refresh_step_table()
        print(f"✓ Added step {len(self.current_program)}: {step}")
    
    def on_update_step(self):
        """Update selected step"""
        selected_row = self.step_table.currentRow()
        if selected_row >= 0 and selected_row < len(self.current_program):
            step = self.get_step_from_inputs()
            self.current_program[selected_row] = step
            self.refresh_step_table()
            print(f"✓ Updated step {selected_row + 1}: {step}")
        else:
            QMessageBox.warning(self, "Warning", "Please select a step to update")
    
    def on_delete_step(self):
        """Delete selected step"""
        selected_row = self.step_table.currentRow()
        if selected_row >= 0 and selected_row < len(self.current_program):
            deleted_step = self.current_program[selected_row]
            del self.current_program[selected_row]
            self.refresh_step_table()
            print(f"✓ Deleted step {selected_row + 1}: {deleted_step}")
        else:
            QMessageBox.warning(self, "Warning", "Please select a step to delete")
    
    def on_clear_program(self):
        """Clear all steps"""
        reply = QMessageBox.question(self, "Confirm", 
                                    "Clear all steps?",
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.current_program = []
            self.refresh_step_table()
            print("✓ All steps cleared")
    
    def get_step_from_inputs(self) -> dict:
        """Get step data from input fields"""
        cmd_text = self.cmd_combo.currentText()
        
        if "G00" in cmd_text:
            cmd = "G00"
        elif "G01" in cmd_text:
            cmd = "G01"
        elif "GRIPPER" in cmd_text:
            cmd = "GRIPPER"
        elif "PUMP_ON" in cmd_text:
            cmd = "PUMP_ON"
        elif "PUMP_OFF" in cmd_text:
            cmd = "PUMP_OFF"
        else:
            cmd = "WAIT"
        
        step = {
            "cmd": cmd,
            "x": self.x_input.value(),
            "y": self.y_input.value(),
            "z": self.z_input.value(),
            "feedrate": self.feedrate_input.value(),
            "angle": self.gripper_angle_input.value(),
            "pause": self.pause_input.value()
        }
        
        return step
    
    def refresh_step_table(self):
        """Refresh step table display"""
        self.step_table.setRowCount(len(self.current_program))
        
        for i, step in enumerate(self.current_program):
            self.step_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.step_table.setItem(i, 1, QTableWidgetItem(step.get("cmd", "G01")))
            self.step_table.setItem(i, 2, QTableWidgetItem(f"{step.get('x', 0):.1f}"))
            self.step_table.setItem(i, 3, QTableWidgetItem(f"{step.get('y', 0):.1f}"))
            self.step_table.setItem(i, 4, QTableWidgetItem(f"{step.get('z', 0):.1f}"))
            self.step_table.setItem(i, 5, QTableWidgetItem(str(step.get('feedrate', 100))))
            self.step_table.setItem(i, 6, QTableWidgetItem(f"{step.get('angle', 90)}°"))
            self.step_table.setItem(i, 7, QTableWidgetItem(f"{step.get('pause', 0):.1f}"))
    
    def debug_show_program(self):
        """Debug: Show current program in console"""
        print("\n" + "="*60)
        print("🔍 DEBUG: Current Program Data")
        print("="*60)
        print(f"Program name: {self.program_name_input.text()}")
        print(f"Number of steps: {len(self.current_program)}")
        print("\nSteps:")
        for i, step in enumerate(self.current_program):
            print(f"  Step {i+1}: {step}")
        print("="*60)
    
    def on_test_program(self):
        """Test execute current program"""
        if not self.current_program:
            QMessageBox.warning(self, "Warning", "No steps to execute")
            return
        
        reply = QMessageBox.question(self, "Confirm",
                                    f"Execute program with {len(self.current_program)} steps?\n\n"
                                    "Make sure the robot has clear workspace!",
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            try:
                print("\n" + "="*60)
                print(f"🧪 TESTING PROGRAM: {len(self.current_program)} steps")
                print("="*60)
                
                for i, step in enumerate(self.current_program):
                    print(f"\n--- Step {i+1}/{len(self.current_program)} ---")
                    print(f"Command: {step}")
                    
                    cmd = step.get("cmd", "G01")
                    
                    if cmd in ["G00", "G01"]:
                        # Movement command
                        x = step.get("x", 0.0)
                        y = step.get("y", 0.0)
                        z = step.get("z", 0.0)
                        feedrate = step.get("feedrate", 100)
                        
                        print(f"Moving to: X={x:.2f}, Y={y:.2f}, Z={z:.2f}, F={feedrate}")
                        
                        if cmd == "G00":
                            success, response = self.controller.robot.move_point_to_point(
                                x, y, z, feedrate
                            )
                        else:
                            success, response = self.controller.robot.move_linear(
                                x, y, z, feedrate
                            )
                        
                        if not success:
                            error_msg = f"Step {i+1} movement failed: {response}"
                            print(f"❌ {error_msg}")
                            QMessageBox.critical(self, "Error", error_msg)
                            return
                        
                        print(f"✓ Movement complete")
                        time.sleep(1.0)
                        
                    elif cmd == "GRIPPER":
                        # Gripper with angle control
                        angle = step.get("angle", 90)
                        print(f"Setting gripper to {angle}°...")
    
                        cmd_str = f"0x550xAA G06 D7 S1 A{angle} 0xAA0x55"
                        success, response = self.controller.robot.send_command(cmd_str)
    
                        # FIXED: Accept both "ok" and successful send
                        if success or "ok" in response.lower():
                            print(f"✓ Gripper set to {angle}°")
                            time.sleep(0.5)
                        else:
                            # Don't fail on gripper - just warn
                            print(f"⚠ Gripper response: {response}")
                            time.sleep(0.5)

                        
                    elif cmd == "PUMP_ON":
                        print("Activating pump...")
                        self.controller.robot.pump_on()
                        time.sleep(0.5)
                        
                    elif cmd == "PUMP_OFF":
                        print("Deactivating pump...")
                        self.controller.robot.pump_off()
                        time.sleep(0.5)
                    
                    elif cmd == "WAIT":
                        pause = step.get("pause", 1.0)
                        print(f"Waiting for {pause}s...")
                        time.sleep(pause)
                    
                    # Step pause (additional delay after command)
                    pause = step.get("pause", 0.0)
                    if pause > 0 and cmd != "WAIT":
                        print(f"Pausing for {pause}s...")
                        time.sleep(pause)
                
                print("\n" + "="*60)
                print("✅ PROGRAM TEST COMPLETE!")
                print("="*60)
                
                QMessageBox.information(self, "Success", 
                                      f"Program executed successfully!\n\n"
                                      f"Steps completed: {len(self.current_program)}")
            
            except Exception as e:
                error_msg = f"Test failed: {e}"
                print(f"\n❌ {error_msg}")
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "Error", f"Execution failed:\n{e}")
    
    # ═══════════════════════════════════════════════════════════════
    # POSITION CALIBRATION
    # ═══════════════════════════════════════════════════════════════
    
    def jog_axis(self, axis: str, direction: int):
        """Jog robot axis"""
        distance = float(self.jog_distance.currentText()) * direction
        
        print(f"Jogging {axis.upper()}{'+' if direction > 0 else ''}{distance:.1f}mm")
        
        try:
            if axis == "x":
                success, _ = self.controller.robot.move_offset(distance, 0, 0, feedrate=50)
            elif axis == "y":
                success, _ = self.controller.robot.move_offset(0, distance, 0, feedrate=50)
            elif axis == "z":
                success, _ = self.controller.robot.move_offset(0, 0, distance, feedrate=50)
            
            time.sleep(0.3)
            self.update_current_position()
        except Exception as e:
            print(f"❌ Jog error: {e}")
            QMessageBox.warning(self, "Error", f"Jog failed: {e}")
    
    def on_set_gripper(self):
        """Set gripper to specified angle"""
        angle = self.manual_gripper_angle.value()
        print(f"Setting gripper to {angle}°...")
        
        try:
            cmd_str = f"0x550xAA G06 D7 S1 A{angle} 0xAA0x55"
            success, response = self.controller.robot.send_command(cmd_str)
            
            if "ok" in response.lower():
                print(f"✓ Gripper set to {angle}°")
            else:
                print(f"⚠ Gripper response: {response}")
        except Exception as e:
            print(f"❌ Gripper error: {e}")
            QMessageBox.warning(self, "Error", f"Failed to set gripper: {e}")
    
    def quick_gripper(self, angle: int):
        """Quick gripper preset"""
        self.manual_gripper_angle.setValue(angle)
        self.on_set_gripper()
    
    def on_home_robot(self):
        """Home the robot"""
        try:
            print("🏠 Homing robot...")
            success, response = self.controller.robot.home()
            if success:
                print("✓ Robot homed")
                QMessageBox.information(self, "Success", "Robot homed")
                self.update_current_position()
            else:
                print(f"❌ Homing failed: {response}")
                QMessageBox.critical(self, "Error", f"Homing failed: {response}")
        except Exception as e:
            print(f"❌ Homing error: {e}")
            QMessageBox.critical(self, "Error", f"Homing error: {e}")
    
    def update_current_position(self):
        """Update current position display"""
        try:
            pos = self.controller.robot.get_position()
            if pos:
                self.current_pos_label.setText(
                    f"X: {pos['x']:.2f}  Y: {pos['y']:.2f}  Z: {pos['z']:.2f}"
                )
            else:
                pos = self.controller.robot.current_position
                self.current_pos_label.setText(
                    f"X: {pos['x']:.2f}  Y: {pos['y']:.2f}  Z: {pos['z']:.2f}"
                )
        except Exception as e:
            print(f"⚠ Position update error: {e}")
    
    def on_save_position(self):
        """Save current position"""
        pos_name = self.position_name_input.text().strip()
        
        if not pos_name:
            QMessageBox.warning(self, "Warning", "Please enter a position name")
            return
        
        # Check if position already exists
        if pos_name in self.calibration.get("positions", {}):
            reply = QMessageBox.question(self, "Confirm Overwrite",
                                        f"Position '{pos_name}' already exists.\n"
                                        "Overwrite it with current position?",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
        
        pos = self.controller.robot.current_position
        
        print(f"💾 Saving position '{pos_name}': X={pos['x']:.2f}, Y={pos['y']:.2f}, Z={pos['z']:.2f}")
        
        self.calibration["positions"][pos_name] = {
            "x": pos["x"],
            "y": pos["y"],
            "z": pos["z"]
        }
        
        if DataStorage.save_calibration(self.calibration):
            # IMPORTANT: Reload positions in controller
            self.controller.calibration = self.calibration
            self.controller.positions = self.calibration.get("positions", {})
            self.controller.reload_positions()
            
            self.load_positions_table()
            self.position_name_input.clear()
            
            print(f"✓ Position '{pos_name}' saved")
            QMessageBox.information(self, "Success", 
                                  f"Position '{pos_name}' saved!\n"
                                  f"X={pos['x']:.1f}, Y={pos['y']:.1f}, Z={pos['z']:.1f}")
        else:
            QMessageBox.critical(self, "Error", "Failed to save position")
    
    def load_positions_table(self):
        """Load positions into table"""
        # Reload from file
        self.calibration = DataStorage.load_calibration()
        positions = self.calibration.get("positions", {})
        
        self.positions_table.setRowCount(len(positions))
        
        for i, (name, pos) in enumerate(positions.items()):
            self.positions_table.setItem(i, 0, QTableWidgetItem(name))
            self.positions_table.setItem(i, 1, QTableWidgetItem(f"{pos['x']:.1f}"))
            self.positions_table.setItem(i, 2, QTableWidgetItem(f"{pos['y']:.1f}"))
            self.positions_table.setItem(i, 3, QTableWidgetItem(f"{pos['z']:.1f}"))
    
    def on_goto_position(self):
        """Go to selected position"""
        selected_row = self.positions_table.currentRow()
        
        if selected_row < 0:
            QMessageBox.warning(self, "Warning", "Please select a position")
            return
        
        pos_name = self.positions_table.item(selected_row, 0).text()
        
        print(f"Going to position '{pos_name}'...")
        
        try:
            success = self.controller.move_to(pos_name)
            if success:
                time.sleep(0.5)
                self.update_current_position()
                print(f"✓ Moved to '{pos_name}'")
            else:
                print(f"❌ Failed to move to '{pos_name}'")
                QMessageBox.critical(self, "Error", f"Failed to move to '{pos_name}'")
        except Exception as e:
            print(f"❌ Move error: {e}")
            QMessageBox.critical(self, "Error", f"Move error: {e}")
    
    def on_overwrite_position(self):
        """Overwrite selected position with current robot position"""
        selected_row = self.positions_table.currentRow()
        
        if selected_row < 0:
            QMessageBox.warning(self, "Warning", "Please select a position to overwrite")
            return
        
        pos_name = self.positions_table.item(selected_row, 0).text()
        
        reply = QMessageBox.question(self, "Confirm Overwrite",
                                    f"Overwrite position '{pos_name}' with current position?\n\n"
                                    f"This will replace the saved coordinates.",
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            pos = self.controller.robot.current_position
            
            print(f"✏️ Overwriting position '{pos_name}': X={pos['x']:.2f}, Y={pos['y']:.2f}, Z={pos['z']:.2f}")
            
            # Update position
            self.calibration["positions"][pos_name] = {
                "x": pos["x"],
                "y": pos["y"],
                "z": pos["z"]
            }
            
            if DataStorage.save_calibration(self.calibration):
                # Reload positions in controller
                self.controller.calibration = self.calibration
                self.controller.positions = self.calibration.get("positions", {})
                self.controller.reload_positions()
                
                self.load_positions_table()
                
                print(f"✓ Position '{pos_name}' updated")
                QMessageBox.information(self, "Success", 
                                      f"Position '{pos_name}' updated!\n"
                                      f"New coordinates:\n"
                                      f"X={pos['x']:.1f}, Y={pos['y']:.1f}, Z={pos['z']:.1f}")
            else:
                QMessageBox.critical(self, "Error", "Failed to update position")
    
    def on_delete_position(self):
        """Delete selected position"""
        selected_row = self.positions_table.currentRow()
        
        if selected_row < 0:
            QMessageBox.warning(self, "Warning", "Please select a position to delete")
            return
        
        pos_name = self.positions_table.item(selected_row, 0).text()
        
        reply = QMessageBox.question(self, "Confirm Delete",
                                    f"Delete position '{pos_name}'?\n\n"
                                    f"This cannot be undone.",
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            print(f"🗑️ Deleting position '{pos_name}'")
            
            # Remove from calibration
            if pos_name in self.calibration["positions"]:
                del self.calibration["positions"][pos_name]
                
                if DataStorage.save_calibration(self.calibration):
                    # Reload positions in controller
                    self.controller.calibration = self.calibration
                    self.controller.positions = self.calibration.get("positions", {})
                    self.controller.reload_positions()
                    
                    self.load_positions_table()
                    
                    print(f"✓ Position '{pos_name}' deleted")
                    QMessageBox.information(self, "Success", f"Position '{pos_name}' deleted")
                else:
                    QMessageBox.critical(self, "Error", "Failed to delete position")
            else:
                QMessageBox.warning(self, "Warning", "Position not found")
