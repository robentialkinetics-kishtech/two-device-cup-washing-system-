#!/usr/bin/env python3
"""
Quick import test to verify all modules load correctly
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

try:
    print("✓ Importing PyQt5...")
    from PyQt5.QtWidgets import QApplication
    
    print("✓ Importing cv2...")
    import cv2
    
    print("✓ Importing ultralytics...")
    from ultralytics import YOLO
    
    print("✓ Importing models...")
    from models.vision import VisionSystem
    from models.controller import CupWashingController
    
    print("✓ Importing UI...")
    from ui.main_window import MainWindow
    
    print("\n✅ All imports successful!")
    
except Exception as e:
    print(f"\n❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
