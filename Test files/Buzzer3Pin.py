import os
import time
import RPi.GPIO as GPIO

BUZZER_PIN = 24

print("--- Ultimate Raw GPIO Passive Test ---")

# Clean up any lingering locks
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

try:
    print("Initializing hardware PWM frequency emulation...")
    # Emulate a raw square wave manually by toggling the pin fast
    # This bypasses all library abstractions entirely
    for _ in range(500):  # Run for a brief moment
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(0.001)  # 1 millisecond high
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        time.sleep(0.001)  # 1 millisecond low
        
    print("✅ Raw toggle complete. Did you hear a buzz or click?")

except Exception as e:
    print(f"Execution Error: {e}")

finally:
    GPIO.cleanup()
    print("Pins cleaned and reset.")