import time
from gpiozero import DigitalInputDevice
import paho.mqtt.client as mqtt

# 📌 Configuration
SENSOR_PIN = 18
MQTT_BROKER = "localhost"  # Since the broker is running on the Pi
MQTT_TOPIC = "home/sensor/pir"

print("Initializing PIR Sensor and MQTT Publisher...")
sensor = DigitalInputDevice(SENSOR_PIN, pull_up=False)

# Setup MQTT Client
client = mqtt.Client()
client.connect(MQTT_BROKER, 1883, 60)
client.loop_start()

print("Publisher ready! Sending live data when motion is detected...")
last_state = -1

try:
    while True:
        current_state = sensor.value # 1 for motion, 0 for clear
        
        # Only send a message when the state actually changes
        if current_state != last_state:
            payload = "MOTION" if current_state == 1 else "CLEAR"
            print(f"Broadcast Update: {payload}")
            client.publish(MQTT_TOPIC, payload)
            last_state = current_state
            
        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nStopping Publisher cleanly.")
    client.loop_stop()
    client.disconnect()