import time
from gpiozero import DigitalOutputDevice, PWMOutputDevice

BUZZER_PIN = 23    #  control wire shifted to GPIO 23

print("--- Starting Buzzer Isolation Test ---")

# Test 1: Active Buzzer Logic (Simple High/Low Voltage)
print("\n[Test 1] Testing as an ACTIVE buzzer...")
try:
    buzzer = DigitalOutputDevice(BUZZER_PIN)
    print("Setting Pin 23 to HIGH (ON)...")
    buzzer.on()
    time.sleep(2)
    buzzer.off()
    buzzer.close()
    print("Active test complete.")
except Exception as e:
    print(f"Active test error: {e}")

time.sleep(1)

# Test 2: Passive Buzzer Logic (Frequency Pulse)
print("\n[Test 2] Testing as a PASSIVE buzzer...")
try:
    buzzer_pwm = PWMOutputDevice(BUZZER_PIN)
    print("Pulsing 1kHz frequency tone on Pin 23...")
    buzzer_pwm.frequency = 1000  # 1kHz audio frequency
    buzzer_pwm.value = 0.5       # 50% duty cycle volume
    time.sleep(2)
    buzzer_pwm.off()
    buzzer_pwm.close()
    print("Passive test complete.")
except Exception as e:
    print(f"Passive test error: {e}")

print("\n--- Diagnostic Loop Finished ---")

