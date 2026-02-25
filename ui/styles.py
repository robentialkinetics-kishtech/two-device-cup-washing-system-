"""
UI Stylesheet and theming
"""

def get_dark_stylesheet():
    """Modern dark theme stylesheet"""
    return """
    /* ============================================
       MAIN WINDOW & GLOBAL STYLES
       ============================================ */
    QMainWindow {
        background-color: #0d1117;
    }
    
    QWidget {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 13px;
    }
    
    /* ============================================
       BUTTONS
       ============================================ */
    QPushButton {
        background-color: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        min-height: 32px;
    }
    
    QPushButton:hover {
        background-color: #30363d;
        border-color: #58a6ff;
    }
    
    QPushButton:pressed {
        background-color: #161b22;
    }
    
    QPushButton:disabled {
        background-color: #161b22;
        color: #484f58;
        border-color: #21262d;
    }
    
    /* Primary Button (Start) */
    QPushButton#startButton {
        background-color: #238636;
        color: white;
        border: none;
        font-size: 15px;
        font-weight: bold;
        min-height: 45px;
    }
    
    QPushButton#startButton:hover {
        background-color: #2ea043;
    }
    
    QPushButton#startButton:pressed {
        background-color: #1f7a2e;
    }
    
    /* Danger Button (Stop/Emergency) */
    QPushButton#stopButton, QPushButton#emergencyButton {
        background-color: #da3633;
        color: white;
        border: none;
        font-weight: bold;
    }
    
    QPushButton#stopButton:hover, QPushButton#emergencyButton:hover {
        background-color: #f85149;
    }
    
    /* ============================================
       INPUT FIELDS
       ============================================ */
    QLineEdit, QSpinBox, QDoubleSpinBox {
        background-color: #0d1117;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 8px 12px;
        min-height: 32px;
    }
    
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
        border-color: #58a6ff;
        outline: none;
    }
    
    QLineEdit:disabled, QSpinBox:disabled {
        background-color: #161b22;
        color: #484f58;
    }
    
    /* ============================================
       COMBO BOX
       ============================================ */
    QComboBox {
        background-color: #0d1117;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 6px 12px;
        min-height: 32px;
    }
    
    QComboBox:hover {
        border-color: #58a6ff;
    }
    
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
    
    QComboBox::down-arrow {
        image: url(none);
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 5px solid #8b949e;
        margin-right: 5px;
    }
    
    QComboBox QAbstractItemView {
        background-color: #161b22;
        color: #c9d1d9;
        border: 1px solid #30363d;
        selection-background-color: #1f6feb;
    }
    
    /* ============================================
       SLIDERS
       ============================================ */
    QSlider::groove:horizontal {
        border: 1px solid #30363d;
        background: #21262d;
        height: 6px;
        border-radius: 3px;
    }
    
    QSlider::handle:horizontal {
        background: #58a6ff;
        border: 2px solid #1f6feb;
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }
    
    QSlider::handle:horizontal:hover {
        background: #79c0ff;
    }
    
    QSlider::sub-page:horizontal {
        background: #1f6feb;
        border-radius: 3px;
    }
    
    /* ============================================
       PROGRESS BAR
       ============================================ */
    QProgressBar {
        border: 1px solid #30363d;
        border-radius: 6px;
        background: #161b22;
        height: 24px;
        text-align: center;
        color: #c9d1d9;
    }
    
    QProgressBar::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                   stop:0 #1f6feb, stop:1 #58a6ff);
        border-radius: 5px;
    }
    
    /* ============================================
       GROUP BOX
       ============================================ */
    QGroupBox {
        color: #c9d1d9;
        border: 2px solid #30363d;
        border-radius: 8px;
        margin-top: 12px;
        padding-top: 16px;
        font-weight: 600;
        font-size: 14px;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 0 5px;
        background-color: #0d1117;
    }
    
    /* ============================================
       LABELS
       ============================================ */
    QLabel {
        color: #c9d1d9;
        background: transparent;
    }
    
    QLabel#titleLabel {
        font-size: 24px;
        font-weight: bold;
        color: #58a6ff;
    }
    
    QLabel#statusLabel {
        font-size: 16px;
        font-weight: bold;
        padding: 12px;
        border-radius: 6px;
        border: 2px solid #30363d;
    }
    
    QLabel#errorLabel {
        color: #f85149;
    }
    
    QLabel#successLabel {
        color: #3fb950;
    }
    
    /* ============================================
       TABLE WIDGET
       ============================================ */
    QTableWidget {
        background-color: #0d1117;
        alternate-background-color: #161b22;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
        gridline-color: #21262d;
    }
    
    QTableWidget::item {
        padding: 5px;
    }
    
    QTableWidget::item:selected {
        background-color: #1f6feb;
    }
    
    QHeaderView::section {
        background-color: #161b22;
        color: #c9d1d9;
        padding: 8px;
        border: none;
        border-bottom: 1px solid #30363d;
        font-weight: 600;
    }
    
    /* ============================================
       TEXT EDIT / TEXT BROWSER
       ============================================ */
    QTextEdit, QTextBrowser, QPlainTextEdit {
        background-color: #0d1117;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 8px;
    }
    
    /* ============================================
       SCROLL BAR
       ============================================ */
    QScrollBar:vertical {
        background: #161b22;
        width: 12px;
        border-radius: 6px;
    }
    
    QScrollBar::handle:vertical {
        background: #30363d;
        border-radius: 6px;
        min-height: 20px;
    }
    
    QScrollBar::handle:vertical:hover {
        background: #484f58;
    }
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    
    QScrollBar:horizontal {
        background: #161b22;
        height: 12px;
        border-radius: 6px;
    }
    
    QScrollBar::handle:horizontal {
        background: #30363d;
        border-radius: 6px;
        min-width: 20px;
    }
    
    /* ============================================
       TAB WIDGET
       ============================================ */
    QTabWidget::pane {
        border: 1px solid #30363d;
        border-radius: 6px;
        background: #0d1117;
    }
    
    QTabBar::tab {
        background: #161b22;
        color: #8b949e;
        border: 1px solid #30363d;
        padding: 10px 20px;
        margin-right: 2px;
    }
    
    QTabBar::tab:selected {
        background: #0d1117;
        color: #58a6ff;
        border-bottom-color: #0d1117;
    }
    
    QTabBar::tab:hover:!selected {
        background: #21262d;
    }
    
    /* ============================================
       STATUS BAR
       ============================================ */
    QStatusBar {
        background: #161b22;
        color: #8b949e;
        border-top: 1px solid #30363d;
    }
    """
