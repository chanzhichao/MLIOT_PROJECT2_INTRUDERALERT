import time
import numpy as np
import pyaudio
import paho.mqtt.client as mqtt
from gpiozero import DigitalInputDevice, LED

'''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
PLACEHOLDER = {
    "WEIGHTS": np.array([2.0, 1.5]),    # W[0] = PIR weight, W[1] = Audio weight
    "BIAS": -1.0,                       # B = Base bias penalty
    "AUDIO_GATE_THRESHOLD": 0.15,       # Hard squashing threshold for "quiet room"
    "CLASSES": {0: "NORMAL", 1: "ANOMALY_DETECTED"}
}
"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"


# 📌 Hardware Configuration
SENSOR_PIN = 18
LED_PIN = 17
DEVICE_INDEX = 1
RATE = 48000  
CHUNK = 1024

# 📌 MQTT Configuration
MQTT_BROKER = "localhost"
MQTT_ML_TOPIC = "home/analytics/ml_inference"

# 📌 DEPLOYED ML MODEL MATRICES (Consistent Placeholder Parameters)
# Feature Index Map: Index 0 = PIR Motion, Index 1 = Audio Amplitude
MODEL_PARAMS = PLACEHOLDER

# Safety hardware reset
for var in ['sensor', 'led']:
    if var in locals():
        try: exec(f"{var}.close()")
        except: pass

print("Deploying ML Model Matrices onto Edge Processor...")
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

print("\n--- Matrix Inference Engine Active ---")
print("Running dot-product calculations... Loops for 30 seconds.")
print("-" * 70)

try:
    end_time = time.time() + 30
    while time.time() < end_time:
        # 1. Feature Extraction (Audio)
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        raw_audio = float(np.max(np.abs(audio_data)))
        x_audio = min(raw_audio / 20000.0, 1.0)
        
        # 2. Feature Extraction (PIR)
        x_pir = float(sensor.value)
        
        # 3. Apply Non-Linear Input Gate (Forces Audio to 0 if room is quiet)
        if x_audio < MODEL_PARAMS["AUDIO_GATE_THRESHOLD"]:
            x_audio = 0.0
            x_pir = 0.0  # Zero out motion too since mic quiet = absolute lockout
            
        # 4. Construct Feature Vector (X)
        X = np.array([x_pir, x_audio])
        
        # 5. Core ML Math: Dot Product of Weights and Features plus Bias (W·X + B)
        W = MODEL_PARAMS["WEIGHTS"]
        B = MODEL_PARAMS["BIAS"]
        logit = np.dot(W, X) + B
        
        # 6. Activation Function (Heaviside Step Function)
        prediction_index = 1 if logit >= 0.0 else 0
        current_class = MODEL_PARAMS["CLASSES"][prediction_index]
        
        # 7. Actuator Output Execution
        if prediction_index == 1:
            led.on()
        else:
            led.off()
            
        # Log vector arrays and model status
        print(f"X Vector: {X} | W Vector: {W} | Logit: {logit:+.2f} | Class: {current_class}")
        
        # 8. Network Broadcast
        client.publish(MQTT_ML_TOPIC, f"LOGIT:{logit:.2f}|CLASS:{current_class}")
        
        time.sleep(0.1)

finally:
    print("\nPowering down Matrix Engine...")
    stream.stop_stream()
    stream.close()
    p.terminate()
    client.loop_stop()
    client.disconnect()
    sensor.close()
    led.close()
    print("System Standby.")