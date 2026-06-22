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

# 📌 Deployed ML Model Matrices
MODEL_PARAMS = {
    "WEIGHTS": np.array([2.5, 1.5]),    # Increased PIR weight now that it's continuous
    "BIAS": -1.2,                       
    "AUDIO_GATE_THRESHOLD": 0.15,       
    "CLASSES": {0: "NORMAL", 1: "ANOMALY_DETECTED"}
}

# Safety hardware reset
for var in ['sensor', 'led']:
    if var in locals():
        try: exec(f"{var}.close()")
        except: pass

print("Deploying Advanced Feature Engine...")
sensor = DigitalInputDevice(SENSOR_PIN, pull_up=False)
led = LED(LED_PIN)

client = mqtt.Client()
client.connect(MQTT_BROKER, 1883, 60)
client.loop_start()

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                input_device_index=DEVICE_INDEX, frames_per_buffer=CHUNK)

# 🕒 Feature Engineering Trackers
motion_intensity = 0.0
DECAY_RATE = 0.08  # How fast the motion memory fades out per loop iteration

print("\n--- Matrix Inference Engine (Continuous PIR) ---")
print("-" * 75)

try:
    end_time = time.time() + 30
    while time.time() < end_time:
        # 1. Feature Extraction (Audio)
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        raw_audio = float(np.max(np.abs(audio_data)))
        x_audio = min(raw_audio / 20000.0, 1.0)
        
        # 2. Advanced Feature Engineering (PIR Accumulator)
        if sensor.value == 1:
            # Add intensity if motion is actively happening, capped at 1.0
            motion_intensity = min(motion_intensity + 0.25, 1.0)
        else:
            # Let the intensity fade away if the room goes still
            motion_intensity = max(motion_intensity - DECAY_RATE, 0.0)
            
        x_pir = motion_intensity
        
        # 3. Apply Strict Non-Linear Input Gate
        if x_audio < MODEL_PARAMS["AUDIO_GATE_THRESHOLD"]:
            x_audio = 0.0
            x_pir = 0.0  # Safe lockout
            
        # 4. Construct Feature Vector (X)
        X = np.array([x_pir, x_audio])
        
        # 5. ML Dot Product Math (W·X + B)
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
            
        # Notice how X Vector now displays fractions for the PIR sensor!
        print(f"X Vector: [{X[0]:.2f}, {X[1]:.2f}] | Logit: {logit:+.2f} | Class: {current_class}")
        
        client.publish(MQTT_ML_TOPIC, f"LOGIT:{logit:.2f}|CLASS:{current_class}")
        time.sleep(0.1)

finally:
    print("\nPowering down Engine...")
    stream.stop_stream()
    stream.close()
    p.terminate()
    client.loop_stop()
    client.disconnect()
    sensor.close()
    led.close()