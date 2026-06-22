import time
import numpy as np
import pyaudio
import paho.mqtt.client as mqtt
from gpiozero import DigitalInputDevice, LED

# 📌 Hardware Configuration
SENSOR_PIN = 18
LED_PIN = 17
DEVICE_INDEX = 1
RATE = 48000  
CHUNK = 1024

# 📌 MQTT Configuration
MQTT_BROKER = "localhost"
MQTT_ML_TOPIC = "home/analytics/ml_inference"

"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
# 📌 DEPLOYED ML MODEL MATRICES 
MODEL_PARAMS = {
    "WEIGHTS": np.array([2.5, 1.5]),    # W[0] = Engineered PIR Weight, W[1] = Audio Weight
    "BIAS": -1.2,                       # B = Base bias penalty
    "AUDIO_GATE_THRESHOLD": 0.08,       # Hard lockout threshold for quiet rooms
    "CLASSES": {0: "NORMAL", 1: "ANOMALY_DETECTED"}
}
"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
# Safety hardware reset
for var in ['sensor', 'led']:
    if var in locals():
        try: exec(f"{var}.close()")
        except: pass

print("Loading Production-Ready Sensor Fusion Engine...")
sensor = DigitalInputDevice(SENSOR_PIN, pull_up=False)
led = LED(LED_PIN)

client = mqtt.Client()
client.connect(MQTT_BROKER, 1883, 60)
client.loop_start()

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                input_device_index=DEVICE_INDEX, frames_per_buffer=CHUNK)

# 🕒 Advanced Feature Trackers
motion_intensity = 0.0
presence_latched = False
DECAY_RATE = 0.02

print("\n--- Unified Vector Fusion Pipeline Live ---")
print("Monitoring fused environment matrix... Loops continuously.")
print("-" * 75)

try:
    while True:
        # 1. Feature Extraction: Audio
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        raw_audio = float(np.max(np.abs(audio_data)))
        # 🚀 Change this line inside the loop:
        # OLD: x_audio = min(raw_audio / 20000.0, 1.0)
        x_audio = min(raw_audio / 4000.0, 1.0)  # 📈 Amplifies the signal by 5x!
        
        # 2. Feature Extraction: Engineered PIR State
        raw_pir = sensor.value
        if raw_pir == 1:
            motion_intensity = min(motion_intensity + 0.08, 1.0)
            presence_latched = True
        elif presence_latched:
            # Drop smoothly to the midway floor and park if a heat source is lingering
            if motion_intensity > 0.50:
                motion_intensity = max(motion_intensity - DECAY_RATE, 0.50)
            else:
                motion_intensity = 0.50
        
        x_pir = motion_intensity
        
        # 3. Apply Non-Linear Audio Lockout Gate
        if x_audio < MODEL_PARAMS["AUDIO_GATE_THRESHOLD"]:
            # If the room goes dead quiet, we force the vectors to clear out
            # This completely neutralizes the warm laptop latch when you leave!
            x_audio = 0.0
            x_pir = 0.0
            motion_intensity = 0.0
            presence_latched = False
            
        # 4. Construct Feature Vector (X)
        X = np.array([x_pir, x_audio])
        
        # 5. Core ML Math: Matrix Dot Product (W·X + B)
        W = MODEL_PARAMS["WEIGHTS"]
        B = MODEL_PARAMS["BIAS"]
        logit = np.dot(W, X) + B
        
        # 6. Activation Function
        prediction_index = 1 if logit >= 0.0 else 0
        current_class = MODEL_PARAMS["CLASSES"][prediction_index]
        
        # 7. Actuator Output
        if prediction_index == 1:
            led.on()
        else:
            led.off()
            
        # 🏁 Updated Print with Terminal Line-Clear String (\033[K)
        print(f"X: [{X[0]:.2f}, {X[1]:.2f}] | Logit: {logit:+.2f} | Output: {current_class}\033[K", end="\r")
        
        # 8. Network Broadcast to MQTT Broker
        client.publish(MQTT_ML_TOPIC, f"LOGIT:{logit:.2f}|CLASS:{current_class}")
        
        time.sleep(0.05)

        # 7. Actuator Output
        if prediction_index == 1:
            led.on()
            # 🕒 Generate a clean human-readable timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # 📜 Print a permanent row ONLY for anomalies (wiping out any end="\r" leftover text)
            print(f"[{timestamp}] ⚠️ ANOMALY DETECTED | X: [{X[0]:.2f}, {X[1]:.2f}] | Logit: {logit:+.2f}\033[K")
        else:
            led.off()
            
        # 8. Network Broadcast to MQTT Broker
        client.publish(MQTT_ML_TOPIC, f"LOGIT:{logit:.2f}|CLASS:{current_class}")
        
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nShutting down fusion engine pipeline...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    client.loop_stop()
    client.disconnect()
    sensor.close()
    led.close()
    print("System Standby.")