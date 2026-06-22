import time
import sys
import select
from gpiozero import DigitalInputDevice

# 📌 Configuration
SENSOR_PIN = 18
ACCUMULATION = 0.10
HOLD_BASELINE = 0.50    # 🔒 The absolute floor the system locks onto

if 'sensor' in locals():
    try: sensor.close()
    except: pass

print("Initializing Latching Memory PIR Engine...")
sensor = DigitalInputDevice(SENSOR_PIN, pull_up=False)

motion_intensity = 0.0
has_detected_presence = False

print("\n--- Running Latching Lock Tracker ---")
print("1. Wave your hand to activate the presence lock.")
print("2. The system will hold midway permanently, even if the sensor goes blind.")
print("3. Press ENTER in the terminal when you want to simulate the person leaving.")
print("-" * 75)

try:
    while True:
        raw_state = sensor.value
        
        # Check if the user pressed ENTER to manually clear the session
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()  # Flush the input
            print("\n[MANUAL OVERRIDE] Resetting environment to empty...")
            motion_intensity = 0.0
            has_detected_presence = False
            time.sleep(1)
            continue

        # 1. Capture the initial entry event
        if raw_state == 1:
            has_detected_presence = True

        # 2. Advanced Latching Logic
        if raw_state == 1:
            # Active movement: Scale up toward maximum intensity
            motion_intensity = min(motion_intensity + ACCUMULATION, 1.0)
            status = "MOVING"
        elif has_detected_presence:
            # Hardware pin dropped to 0, but software latch prevents it from resetting to IDLE!
            # It will decay slightly down to the baseline floor and park there indefinitely.
            if motion_intensity > HOLD_BASELINE:
                motion_intensity = max(motion_intensity - 0.05, HOLD_BASELINE)
            else:
                motion_intensity = HOLD_BASELINE
            status = "PRESENCE LOCKED"
        else:
            motion_intensity = 0.0
            status = "IDLE"
            
        # Visual text bar representation
        bar_length = int(motion_intensity * 20)
        graph_bar = "█" * bar_length
        
        print(f"Pin: {raw_state} | Score: {motion_intensity:.2f} | State: {status:<16} | {graph_bar}", end="\r")
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nTerminating loop process.")
finally:
    sensor.close()
    print("System Idle.")