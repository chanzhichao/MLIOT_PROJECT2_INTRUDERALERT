import time
import numpy as np
import pyaudio
import paho.mqtt.client as mqtt
from gpiozero import DigitalInputDevice, LED

# 📌 Hardware Configuration
SENSOR_PIN = 18
LED_PIN = 17
DEVICE_INDEX = 1
RATE = 48000# Adjusted to your working hardware rate
CHUNK = 1024

# 📌 MQTT Configuration
MQTT_BROKER = "localhost"
MQTT_FUSION_TOPIC = "home/analytics/fusion"

# 📌 Sensor Fusion Parameters (Pre-ML Rule Setup)
WEIGHT_PIR = 0.6      # Give motion a 60% weight
WEIGHT_AUDIO = 0.4    # Give audio a 40% weight
AUDIO_MAX_EXPECTED = 20000  # Cap raw audio amplitude for normalization
ALARM_THRESHOLD = 0.5 # Trigger if combined index crosses 50%

# Safety hardware reset
for var in ['sensor', 'led']:
    if var in locals():
        try: exec(f"{var}.close()")
        except: pass

print("Initializing hardware and network clients...")
sensor = DigitalInputDevice(SENSOR_PIN, pull_up=False)
led = LED(LED_PIN)

# Network client setup
client = mqtt.Client()
client.connect(MQTT_BROKER, 1883, 60)
client.loop_start()

# Audio Stream Setup
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                input_device_index=DEVICE_INDEX, frames_per_buffer=CHUNK)

print("\n--- Sensor Fusion Core Active ---")
print("Monitoring environment... Running for 30 seconds.")
print("-" * 50)

try:
    end_time = time.time() + 30
    while time.time() < end_time:
        # 1. Gather Audio Data
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        peak_audio = float(np.max(np.abs(audio_data)))
        
        # Normalize audio input between 0.0 and 1.0
        normalized_audio = min(peak_audio / AUDIO_MAX_EXPECTED, 1.0)
        
        # 2. Gather PIR Motion Data (0 or 1)
        motion_state = float(sensor.value)
        
        # 3. Perform Sensor Fusion Calculation
        activity_index = (WEIGHT_PIR * motion_state) + (WEIGHT_AUDIO * normalized_audio)
        
        # 4. Threshold Logic Assessment
        if activity_index >= ALARM_THRESHOLD:
            status = "HIGH_ALERT"
            led.on()  # Light up indicating confirmed threat event
        else:
            status = "NORMAL"
            led.off()
            
        # Log to local console window
        print(f"PIR: {motion_state} | Mic: {normalized_audio:.2f} | Fusion Index: {activity_index:.2f} | Status: {status}")
        
        # 5. Broadcast Analytics Over the Network
        client.publish(MQTT_FUSION_TOPIC, f"INDEX:{activity_index:.2f}|STATUS:{status}")
        
        time.sleep(0.1)

finally:
    print("\nShutting down Fusion pipeline gracefully...")
    stream.stop_stream()
    stream.close()
    p.terminate()
    client.loop_stop()
    client.disconnect()
    sensor.close()
    led.close()
    print("System Idle.")