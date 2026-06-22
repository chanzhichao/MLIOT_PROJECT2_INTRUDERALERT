import time
from gpiozero import LED, DigitalOutputDevice, PWMOutputDevice, DigitalInputDevice

# 📌 Core Component Pins
LED_PIN = 17       
BUZZER_PIN = 23    
SENSOR_PIN = 18    

print("Initializing components...")
led = LED(LED_PIN)
sensor = DigitalInputDevice(SENSOR_PIN, pull_up=False, active_state=False)

print("\n--- Phase 1: LED Test ---")
led.on()
time.sleep(1)
led.off()
print("LED Phase complete.")

print("\n--- Phase 2: Buzzer Test ---")
try:
    # Testing active buzzer logic on Pin 23
    buzzer = DigitalOutputDevice(BUZZER_PIN)
    print("Sending ON signal to buzzer...")
    buzzer.on()
    time.sleep(1)
    buzzer.off()
    buzzer.close()
except Exception as e:
    print(f"Buzzer test error: {e}")

print("\n--- Phase 3: Live Sensor Loop (10 Seconds) ---")
print("Wave your hand in front of the sensor now...")
end_time = time.time() + 10

while time.time() < end_time:
    if sensor.is_active:
        print("Sensor Triggered!")
    time.sleep(0.1)

print("\nAll diagnostic phases complete!")