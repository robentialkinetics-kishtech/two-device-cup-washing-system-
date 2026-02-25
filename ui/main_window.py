# main_window.py
"""
Main Window - Page manager and navigation
"""
from PyQt5.QtWidgets import (QMainWindow, QStackedWidget, QMessageBox,
                             QStatusBar)
from PyQt5.QtCore import Qt
from ui.login_page import LoginPage
from ui.user_interface import UserInterface
from ui.developer_page import DeveloperPage
from ui.styles import get_dark_stylesheet
from models.controller import CupWashingController

class MainWindow(QMainWindow):
    """Main application window with page management"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize controller
        self.controller = CupWashingController()
        
        # Current user
        self.current_user = None
        self.current_role = None
        
        self.initUI()
        
        # Initialize system
        if not self.controller.initialize():
            QMessageBox.critical(self, "Error", 
                               "Failed to initialize robot system!\n"
                               "Check connections and try again.")
    
    def initUI(self):
        """Initialize main window UI"""
        self.setWindowTitle("ZKBot Cup Washing System v1.0")
        self.setGeometry(100, 100, 1400, 900)
        
        # Apply dark theme
        self.setStyleSheet(get_dark_stylesheet())
        
        # ═══════════════════════════════════════════════════════════
        # STACKED WIDGET FOR PAGE SWITCHING
        # ═══════════════════════════════════════════════════════════
        
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # ═══════════════════════════════════════════════════════════
        # CREATE PAGES
        # ═══════════════════════════════════════════════════════════
        
        # Login page
        self.login_page = LoginPage()
        self.login_page.login_successful.connect(self.on_login_success)
        self.stacked_widget.addWidget(self.login_page)
        
        # User interface page
        self.user_interface = UserInterface(self.controller)
        self.user_interface.logout_requested.connect(self.on_logout)
        self.user_interface.developer_mode_requested.connect(self.show_developer_page)
        self.stacked_widget.addWidget(self.user_interface)
        
        # Developer page
        self.developer_page = DeveloperPage(self.controller)
        self.developer_page.back_to_user_mode.connect(self.show_user_interface)
        self.stacked_widget.addWidget(self.developer_page)
        
        # ═══════════════════════════════════════════════════════════
        # STATUS BAR
        # ═══════════════════════════════════════════════════════════
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("System ready - Please login")
        
        # Start with login page
        self.stacked_widget.setCurrentWidget(self.login_page)
    
    def on_login_success(self, username: str, role: str):
        """Handle successful login"""
        self.current_user = username
        self.current_role = role
        
        self.status_bar.showMessage(f"Logged in as {username} ({role})")
        
        # Update user interface with username
        self.user_interface.username = username
        
        # Show user interface
        self.show_user_interface()
    
    def on_logout(self):
        """Handle logout"""
        self.current_user = None
        self.current_role = None
        
        # Stop any running operations
        if self.controller.is_running:
            self.controller.stop_washing()
        
        self.status_bar.showMessage("Logged out - Please login")
        self.stacked_widget.setCurrentWidget(self.login_page)
    
    def show_user_interface(self):
        """Show user interface page"""
        self.stacked_widget.setCurrentWidget(self.user_interface)
        self.status_bar.showMessage(f"User Mode - {self.current_user}")
    
    def show_developer_page(self):
        """Show developer page"""
        if self.current_role not in ["developer", "admin"]:
            QMessageBox.warning(self, "Access Denied",
                              "You don't have permission to access Developer Mode")
            return
        
        self.stacked_widget.setCurrentWidget(self.developer_page)
        self.status_bar.showMessage(f"Developer Mode - {self.current_user}")
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.controller.is_running:
            reply = QMessageBox.question(self, "Confirm Exit",
                                        "Washing operation is running!\n"
                                        "Are you sure you want to exit?",
                                        QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.No:
                event.ignore()
                return
        
        # Shutdown system
        self.controller.shutdown()
        event.accept()
