# Robot Alarm Troubleshooting Guide

## Problem: Arm Alarms But Doesn't Move

If your robot gives an alarm sound but doesn't move, here are the most common causes and fixes:

## Common Causes

### 1. **Robot Not Homed** ⚠️ MOST COMMON
**Symptom**: Robot beeps/alarms but doesn't move

**Solution**: Robot MUST be homed before any movement
```python
robot.home()  # This is critical!
```

**Why**: Most robots require homing to establish reference position before movement commands.

### 2. **E-Stop Engaged**
**Symptom**: Robot in error/alarm state

**Solution**: 
- Check physical E-stop button (should be released)
- Check E-stop status: `robot.check_estop()`
- Reset errors: `robot.reset_errors()`

### 3. **Robot in Error State**
**Symptom**: Previous error not cleared

**Solution**:
```python
robot.reset_errors()  # Clear any previous errors
time.sleep(0.5)       # Wait for reset
```

### 4. **Coordinates Out of Limits**
**Symptom**: Movement command rejected

**Solution**: Check workspace limits
- X: -400 to 400 mm
- Y: -400 to 400 mm  
- Z: -300 to 300 mm

Your target (20, -20, -20) is within limits, so this is likely not the issue.

### 5. **Baudrate Mismatch**
**Symptom**: Commands not received properly

**Solution**: Verify baudrate matches robot controller
- Default in code: 115200
- Check robot controller settings

## Diagnostic Steps

### Step 1: Run Diagnostics
```bash
python test_robot_diagnostics.py
```

This will:
- Test connection
- Check E-stop
- Reset errors
- Home robot
- Test small movement
- Test target position

### Step 2: Check Initialization Sequence

The correct sequence is:
1. **Connect** to robot
2. **Reset errors** (clear any previous errors)
3. **Check E-stop** (verify safety)
4. **Home robot** (CRITICAL - establishes reference)
5. **Then** send movement commands

### Step 3: Verify Camera Index

Your code uses camera index **1** (external camera), not 0 (laptop camera):

```python
CAM_INDEX = 1  # External camera
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
```

If camera doesn't open, try:
- `CAM_INDEX = 0` (laptop camera)
- `CAM_INDEX = 2` (another USB camera)

## Updated Test Script

I've created `test_vision_roi_background_sub.py` which:

✅ Uses camera index 1 (your external camera)  
✅ Uses background subtraction (like your code)  
✅ Properly initializes robot (home, reset errors, check E-stop)  
✅ Better error messages  
✅ Manual movement test ('m' key)

## Quick Fix Checklist

Before running vision test:

- [ ] Robot is powered on
- [ ] USB cable connected
- [ ] E-stop button is released (not pressed)
- [ ] Robot is homed (run `robot.home()`)
- [ ] No previous errors (run `robot.reset_errors()`)
- [ ] Camera index correct (1 for external, 0 for laptop)
- [ ] Background captured (press 'c' with empty ROI)

## Testing Order

1. **First**: Run `test_robot_diagnostics.py`
   - This will verify robot connection and movement
   - If this fails, fix robot issues first

2. **Then**: Run `test_vision_roi_background_sub.py`
   - This tests vision + robot movement together
   - Press 'c' to capture background first
   - Place cup in ROI to trigger movement

## Expected Behavior

When cup is detected:
1. Detection count increases (shown on screen)
2. After 8 consecutive detections → "✅ Cup detected" message
3. Robot moves to X=20, Y=-20, Z=-20
4. Cooldown period (3 seconds) before next detection

## If Still Not Working

1. **Check robot controller display** - it may show specific error codes
2. **Try manual movement** - Press 'm' in test script to test movement without vision
3. **Check serial communication** - Verify commands are being sent:
   ```python
   # Add debug prints in robot.py send_command()
   print(f"Sent: {command}")
   print(f"Response: {response}")
   ```
4. **Verify command format** - Check if robot expects different G-code format
5. **Test with simpler command** - Try moving just 1mm in X direction first

## Command Format

Your robot uses this frame format:
```
0x550xAA <G-code> 0xAA0x55
```

Example movement command:
```
0x550xAA G00 X20 Y-20 Z-20 F100 0xAA0x55
```

If robot still alarms, the controller might:
- Expect different spacing
- Need different command format
- Have safety limits enabled
- Require additional initialization

## Next Steps

1. Run diagnostics script first
2. If diagnostics pass, run vision test
3. Check console output for specific error messages
4. Adjust ROI and detection parameters as needed
