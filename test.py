import serial
import time

# Connect
ser = serial.Serial('COM3', 115200, timeout=2)
time.sleep(2)

# Test different formats
test_commands = [
    "G00 X20 Y0 Z0 F10",           # Standard
    "G0 X20 Y0 Z0 F10",            # Short form
    "G00X20Y0Z0F10",               # No spaces
    "G00 X20.0 Y0.0 Z0.0 F10",     # Decimals
    "$J=G00 X20 Y0 Z0 F10",        # GRBL style
    "G00 X20 Y0 Z0",                # No feedrate
]

for i, cmd in enumerate(test_commands):
    print(f"\n{i+1}. Testing: {cmd}")
    ser.reset_input_buffer()
    ser.write(f"{cmd}\n".encode())
    time.sleep(0.5)
    
    response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
    print(f"   Response: {response.strip() if response else '(no response)'}")
    time.sleep(1)

ser.close()
