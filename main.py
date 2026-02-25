"""
#main.py
ZKBot Cup Washing System - Main Entry Point
Automated Industrial Washing Station

Author: ZKBot Systems
Version: 1.0.0
Date: 2026-01-18
"""

import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    """Main application entry point"""
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("ZKBot Cup Washing System")
    app.setOrganizationName("ZKBot Systems")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
