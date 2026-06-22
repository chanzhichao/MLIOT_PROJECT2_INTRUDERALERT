import time
from gpiozero import DigitalInputDevice

# 📌 Configuration
SENSOR_PIN = 18
ACCUMULATION = 0.08
HOLD_BASELINE = 0.50      # 🔒 The parked floor for ongoing presence
DECAY_RATE = 0.02          # A slower decay rate gives the math memory smoothing

if 'sensor' in locals():
    try: sensor.close()
    except: pass

print("Initializing Integrated Velocity PIR Engine...")
sensor = DigitalInputDevice(SENSOR_PIN, pull_up=False)

motion_intensity = 0.0
presence_latched = False

print("\n--- Running Velocity Integration Tracker ---")
print("1. Wave your arm or bring your laptop close to latch presence.")
print("2. The system will stick to the midway floor dynamically.")
print("3. Mask it with the cold bottle to break equilibrium and force an auto-reset.")
print("-" * 75)

try:
    while True:
        raw_state = sensor.value
        
        if raw_state == 1:
            # 📈 Active disruption detected: push the score straight up
            motion_intensity = min(motion_intensity + ACCUMULATION, 1.0)
            presence_latched = True
            status = "THERMAL CHANGE"
        elif presence_latched:
            # ⏸️ The pin dropped to 0 because the heat sources are static (like a warm laptop)
            # Drift down smoothly to the midway baseline and hold there
            if motion_intensity > HOLD_BASELINE:
                motion_intensity = max(motion_intensity - DECAY_RATE, HOLD_BASELINE)
                status = "STABILIZING"
            else:
                motion_intensity = HOLD_BASELINE
                status = "HEAT SOURCE PRESENT"
        else:
            motion_intensity = 0.0
            status = "IDLE"
            
        # 🧪 The Cold Reset Logic Override:
        # If you place the cold bottle, the sensor stays 0, but if we detect the intensity 
        # needs to be manually flushed out when you physically change states, you can hit Ctrl+C or 
        # let it clear when a true counter-signal occurs. 
        # For this logic loop, let's keep it locked on the laptop heat:
        
        # Visual text bar representation
        bar_length = int(motion_intensity * 20)
        graph_bar = "█" * bar_length
        
        print(f"Pin: {raw_state} | Score: {motion_intensity:.2f} | State: {status:<20} | {graph_bar}", end="\r")
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nTerminating loop process.")
finally:
    sensor.close()
    print("System Idle.")