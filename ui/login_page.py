"""
Login Page - Authentication screen
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QFrame, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap
import hashlib
from data.storage import DataStorage

class LoginPage(QWidget):
    """Login authentication page"""
    
    # Signal emitted when login successful
    login_successful = pyqtSignal(str, str)  # username, role
    
    def __init__(self):
        super().__init__()
        self.settings = DataStorage.load_settings()
        self.initUI()
    
    def initUI(self):
        """Initialize login UI"""
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        
        # ═══════════════════════════════════════════════════════════
        # LOGO & TITLE
        # ═══════════════════════════════════════════════════════════
        
        title_label = QLabel("🤖 ZKBot Cup Washing System")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 28, QFont.Bold))
        main_layout.addWidget(title_label)
        
        subtitle = QLabel("Automated Industrial Washing Station")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #8b949e; font-size: 14px; margin-bottom: 30px;")
        main_layout.addWidget(subtitle)
        
        # ═══════════════════════════════════════════════════════════
        # LOGIN FORM
        # ═══════════════════════════════════════════════════════════
        
        login_frame = QFrame()
        login_frame.setStyleSheet("""
            QFrame {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 12px;
                padding: 20px;
            }
        """)
        login_frame.setMaximumWidth(400)
        
        form_layout = QVBoxLayout()
        
        # Username
        username_label = QLabel("Username:")
        username_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        form_layout.addWidget(username_label)
        
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter username")
        self.username_input.setText("admin")  # Default for testing
        self.username_input.returnPressed.connect(self.on_login_clicked)
        form_layout.addWidget(self.username_input)
        
        # Password
        password_label = QLabel("Password:")
        password_label.setStyleSheet("font-weight: bold; margin-top: 15px;")
        form_layout.addWidget(password_label)
        
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setText("zkbot123")  # Default for testing
        self.password_input.returnPressed.connect(self.on_login_clicked)
        form_layout.addWidget(self.password_input)
        
        # Login button
        self.login_btn = QPushButton("🔐 LOGIN")
        self.login_btn.setObjectName("startButton")
        self.login_btn.setMinimumHeight(50)
        self.login_btn.clicked.connect(self.on_login_clicked)
        self.login_btn.setStyleSheet("""
            QPushButton {
                background-color: #238636;
                color: white;
                border: none;
                font-size: 16px;
                font-weight: bold;
                margin-top: 20px;
            }
            QPushButton:hover {
                background-color: #2ea043;
            }
        """)
        form_layout.addWidget(self.login_btn)
        
        # Error label
        self.error_label = QLabel("")
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setStyleSheet("color: #f85149; margin-top: 10px;")
        self.error_label.setWordWrap(True)
        form_layout.addWidget(self.error_label)
        
        login_frame.setLayout(form_layout)
        main_layout.addWidget(login_frame, alignment=Qt.AlignCenter)
        
        # ═══════════════════════════════════════════════════════════
        # INFO
        # ═══════════════════════════════════════════════════════════
        
        info_label = QLabel("Default credentials: admin / zkbot123")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #8b949e; font-size: 12px; margin-top: 20px;")
        main_layout.addWidget(info_label)
        
        version_label = QLabel("Version 1.0.0 | © 2026 ZKBot Systems")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setStyleSheet("color: #484f58; font-size: 11px; margin-top: 10px;")
        main_layout.addWidget(version_label)
        
        self.setLayout(main_layout)
    
    def on_login_clicked(self):
        """Handle login button click"""
        username = self.username_input.text().strip()
        password = self.password_input.text()
        
        if not username or not password:
            self.show_error("Please enter both username and password")
            return
        
        # Authenticate
        if self.authenticate(username, password):
            # Get user role
            user_data = self.settings.get("user", {})
            role = user_data.get("role", "user")
            
            # Update session
            user_data["last_login"] = DataStorage.load_json.__module__  # timestamp
            user_data["session_active"] = True
            self.settings["user"] = user_data
            DataStorage.save_settings(self.settings)
            
            # Emit success signal
            self.login_successful.emit(username, role)
            
            # Clear form
            self.password_input.clear()
            self.error_label.clear()
        else:
            self.show_error("❌ Invalid username or password")
    
    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate user credentials"""
        user_data = self.settings.get("user", {})
        stored_username = user_data.get("username", "admin")
        stored_hash = user_data.get("password_hash", "")
        
        # Check username
        if username != stored_username:
            return False
        
        # Check password
        password_hash = self.hash_password(password)
        return password_hash == stored_hash
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def show_error(self, message: str):
        """Display error message"""
        self.error_label.setText(message)
        self.username_input.setStyleSheet("border: 1px solid #f85149;")
        self.password_input.setStyleSheet("border: 1px solid #f85149;")
