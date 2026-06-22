import cv2
import numpy as np
import onnxruntime as ort
import ai_edge_litert.interpreter as litert
import time
import json
import paho.mqtt.client as mqtt
from gpiozero import DigitalInputDevice  # Lightweight way to read any PIR sensor

# ==========================================
# 1. HARDWARE & MQTT SETTINGS
# ==========================================
CAMERA_INDEX = 0      
PIR_SENSOR_PIN = 18   

# MQTT Broker Configuration
MQTT_BROKER = "test.mosquitto.org"  # Public broker for testing
MQTT_PORT = 1883
MQTT_TOPIC = "home/security/intruder"

# ==========================================
# 2. INITIALIZE MQTT CLIENT
# ==========================================
print("Connecting to MQTT Broker...")
mqtt_client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)

try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()  # Starts background thread to handle network traffic
    print(f"✅ Successfully connected to MQTT broker: {MQTT_BROKER}")
except Exception as e:
    print(f"❌ Failed to connect to MQTT broker: {e}. Running in offline mode.")

# ==========================================
# 3. INITIALIZE AI MODELS FROM STORAGE
# ==========================================
camera_session = ort.InferenceSession("Models/yolo11n.onnx")
camera_input_name = camera_session.get_inputs()[0].name

audio_interpreter = litert.Interpreter(model_path="Models/mobilenetv2_audio.tflite")
audio_interpreter.allocate_tensors()
audio_input_details = audio_interpreter.get_input_details()
audio_output_details = audio_interpreter.get_output_details()

# Ensure V4L2 backend is specified to prevent Linux frame-buffer lag
video_capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

# ==========================================
# 4. HELPER PREPROCESSING FUNCTIONS & PIR DECAY
# ==========================================
pir_hardware = DigitalInputDevice(PIR_SENSOR_PIN)

# Decay tracking variables
pir_state = False
last_motion_time = 0
DECAY_DURATION = 2.0  # Threshold snaps back exactly 2.0s after motion stops

def read_pir_sensor(pin):
    global pir_state, last_motion_time
    if pir_hardware.value == 1:
        pir_state = True
        last_motion_time = time.time()  # Reset our timer anchor to 'now'
    else:
        if pir_state and (time.time() - last_motion_time > DECAY_DURATION):
            pir_state = False  # Time's up! Snap back to False.
    return pir_state

def preprocess_frame(frame):
    # YOLO11 explicitly expects 640x640 dimensions
    resized = cv2.resize(frame, (640, 640))
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = np.float32(rgb_img) / 255.0
    blob = np.transpose(blob, (2, 0, 1))  # Format shape to [Channels, Height, Width]
    blob = np.expand_dims(blob, axis=0)   # Add batch dimension
    return blob

# ==========================================
# 5. 4D VECTOR SENSOR FUSION CONFIGURATION
# ==========================================
BASE_THRESHOLD = 0.70     
SENSITIVITY_BOOST = 0.25  

# Independent Vector Weights (The Intelligent OR Math Layer)
W_VISION = 1.00            # Dim 1: Human Visual Identification
W_AUDIO_CLASS = 0.50       # Dim 2: Audio Model Confidence (Scream Profile)
W_AUDIO_SPIKE = 0.30       # Dim 3: Audio DSP Envelope Sudden Transient Shift

# Audio Tracking Settings
background_audio_level = 0.10  
ALPHA = 0.01                   
SPIKE_SENSITIVITY_MULTIPLIER = 2.0  

# Performance Optimization Trackers
frame_counter = 0
vision_score = 0.0  # Persistent score matrix memory space across skipped frames

# ==========================================
# 6. EXECUTION LOOP
# ==========================================
try:
    print("Security System successfully armed. Monitoring streams...")
    while True:
        # Physical Gatekeeper Tracking (Dim 4)
        pir_active = read_pir_sensor(PIR_SENSOR_PIN)
        
        ret, frame = video_capture.read()
        if not ret:
            print("Warning: Camera stream disconnected.")
            continue

        # --- FRAME SKIPPING OPTIMIZATION ---
        frame_counter += 1
        if frame_counter % 5 == 0:  # Only run heavy vision inference every 5th frame
            input_data = preprocess_frame(frame)
            vision_output = camera_session.run(None, {camera_input_name: input_data})
            
            raw_predictions = vision_output[0][0] 
            class_confidences = raw_predictions[4:, :]
            vision_score = float(np.max(class_confidences)) 
            frame_counter = 0  # Flush loop counter
        
        # --- AUDIO INFERENCE & DSP VECTOR ANALYSIS ---
        expected_audio_shape = audio_input_details[0]['shape']
        
        # Simulation Layer: Emulating raw threat probabilities
        if np.random.random() > 0.95:
            print("\n🔊 [SIMULATION] Injecting a sudden loud screaming spike!")
            raw_audio_score = 0.95  
        else:
            raw_audio_score = np.random.uniform(0.10, 0.15)

        # Vector Extraction
        audio_class_score = raw_audio_score 
        raw_delta = raw_audio_score - background_audio_level
        audio_spike_score = max(0.0, raw_delta * SPIKE_SENSITIVITY_MULTIPLIER)
        audio_spike_score = min(1.0, audio_spike_score)  

        # Adapt Ambient Environmental Sound Tracking
        background_audio_level = (ALPHA * raw_audio_score) + ((1.0 - ALPHA) * background_audio_level)

        # --- 4D VECTOR SENSOR FUSION DECISION MATH ---
        fused_ai_score = (W_VISION * vision_score) + \
                         (W_AUDIO_CLASS * audio_class_score) + \
                         (W_AUDIO_SPIKE * audio_spike_score)
        
        if pir_active:
            current_threshold = BASE_THRESHOLD - SENSITIVITY_BOOST
        else:
            current_threshold = BASE_THRESHOLD

        # --- ALERT VERIFICATION LANE ---
        if fused_ai_score > current_threshold:
            print("\n🚨 EXHIBITION ANOMALY TRIGGERED! (True Positive Matrix) 🚨")
            
            # Mathematical Vector Contribution Auditing
            vision_contrib = vision_score * W_VISION
            audio_class_contrib = audio_class_score * W_AUDIO_CLASS
            audio_spike_contrib = audio_spike_score * W_AUDIO_SPIKE
            
            total_contrib = vision_contrib + audio_class_contrib + audio_spike_contrib
            
            # Safely avoid DivisionByZero if system glitches out empty
            if total_contrib > 0:
                vision_pct = (vision_contrib / total_contrib) * 100
                audio_class_pct = (audio_class_contrib / total_contrib) * 100
                audio_spike_pct = (audio_spike_contrib / total_contrib) * 100
            else:
                vision_pct = audio_class_pct = audio_spike_pct = 0.0

            print("=" * 60)
            print(f"4D FUSED MATRIX SCORE: {fused_ai_score:.2f} (Required Limit: {current_threshold:.2f})")
            print(f"PIR Boundary Ring    : {'BREACHED (-' + str(SENSITIVITY_BOOST) + ' Lowered Barrier)' if pir_active else 'SECURE'}")
            print("-" * 60)
            print(f"🎥 Dim 1: Human Classification : Raw: {vision_score:.2f} -> Weighted: {vision_contrib:.2f} ({vision_pct:.1f}%)")
            print(f"🔊 Dim 2: Semantic Sound Class : Raw: {audio_class_score:.2f} -> Weighted: {audio_class_contrib:.2f} ({audio_class_pct:.1f}%)")
            print(f"⚡ Dim 3: Acoustic Spike Delta : Final: {audio_spike_score:.2f} (Raw: {raw_delta:.2f}) -> Weighted: {audio_spike_contrib:.2f} ({audio_spike_pct:.1f}%)")
            print(f"🌐 Ambient Environment Floor : Dynamic Base Level: {background_audio_level:.2f}")
            print("=" * 60)
            
            # Image Capture Action
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"exhibition_incident_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Incident Evidence Captured: {filename}")
            
            # MQTT Telemetry Construction
            payload = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "ALERT_CRITICAL",
                "exhibition_zone": "Fragile Art Display A",
                "fusion_score": round(fused_ai_score, 2),
                "threshold_used": round(current_threshold, 2),
                "pir_state": int(pir_active),
                "vision_raw_human": round(vision_score, 2),
                "vision_contribution": round(vision_contrib, 2),
                "audio_classification_raw": round(audio_class_score, 2),
                "audio_classification_contribution": round(audio_class_contrib, 2),
                "audio_spike_delta": round(audio_spike_score, 2),
                "audio_spike_contribution": round(audio_spike_contrib, 2),
                "ambient_background_level": round(background_audio_level, 2),
                "evidence_snapshot": filename
            }
            
            # Dispatch Telemetry Frame
            mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
            print(f"📡 4D Telemetry Payload Dispatched to topic '{MQTT_TOPIC}'\n")
            
            time.sleep(5)  # Anti-flicker delay to keep logs readable before resuming telemetry

        # Micro-sleep to allow underlying CPU kernels to balance multi-threading tasks
        time.sleep(0.01)

finally:
    print("\nReleasing system resources gracefully...")
    video_capture.release()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("🔄 Hardware safely returned to OS state.")