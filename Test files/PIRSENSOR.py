import time
from gpiozero import DigitalInputDevice

SENSOR_PIN = 18  # Green signal wire plugged into GPIO 18

print("Initializing PIR Sensor on GPIO 18...")
# We initialize without internal pull-ups because PIR modules have their own power regulation
sensor = DigitalInputDevice(SENSOR_PIN, pull_up=False)

print("Starting live sensor readouts for 15 seconds.")
print("Wave your hand in front of the sensor to test...")
print("-" * 40)

end_time = time.time() + 15

while time.time() < end_time:
    # Print the raw digital state: 1 (High) or 0 (Low)
    print(f"Sensor Raw Value: {sensor.value} | Triggered: {sensor.is_active}")
    time.sleep(0.2)

print("-" * 40)
print("Sensor test loop complete.")