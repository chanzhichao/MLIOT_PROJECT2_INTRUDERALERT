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
    mqtt_client.loop_start() # Starts background thread to handle network traffic
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

video_capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

# ==========================================
# 4. HELPER PREPROCESSING FUNCTIONS & PIR DECAY
# ==========================================
# Initialize the PIR sensor hardware pin
pir_hardware = DigitalInputDevice(PIR_SENSOR_PIN)

# Decay tracking variables
pir_state = False
last_motion_time = 0
DECAY_DURATION = 2.0  # Threshold snaps back to 0.75 exactly 2.0s after motion stops

def read_pir_sensor(pin):
    global pir_state, last_motion_time
    
    # Check if the PIR sensor physically detects heat/motion right now
    if pir_hardware.value == 1:
        pir_state = True
        last_motion_time = time.time()  # Reset our timer anchor to 'now'
    else:
        # If hardware stopped moving, check if our fast decay timer has run out
        if pir_state and (time.time() - last_motion_time > DECAY_DURATION):
            pir_state = False  # Time's up! Snap back to False.
            
    return pir_state

def preprocess_frame(frame):
    # YOLO11 explicitly expects 640x640 dimensions
    resized = cv2.resize(frame, (640, 640))
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = np.float32(rgb_img) / 255.0
    blob = np.transpose(blob, (2, 0, 1)) # Format shape to [Channels, Height, Width]
    blob = np.expand_dims(blob, axis=0)  # Add batch dimension [1, Channels, Height, Width]
    return blob

# ==========================================
# 5. SENSOR FUSION CONFIGURATION
# ==========================================
BASE_THRESHOLD = 0.75     
SENSITIVITY_BOOST = 0.20  

W_VISION = 0.25            
W_AUDIO = 0.8             

# ==========================================
# 6. EXECUTION LOOP
# ==========================================
try:
    print("Security System successfully armed. Monitoring streams...")
    while True:
        pir_active = read_pir_sensor(PIR_SENSOR_PIN)
        
        ret, frame = video_capture.read()
        if not ret:
            print("Warning: Camera stream disconnected.")
            continue

        # --- Inference 1: Vision (YOLO11 ONNX) ---
        input_data = preprocess_frame(frame)
        vision_output = camera_session.run(None, {camera_input_name: input_data})
        
        raw_predictions = vision_output[0][0] 
        class_confidences = raw_predictions[4:, :]
        vision_score = float(np.max(class_confidences)) 

        # --- Inference 2: Audio (MobileNetV2 LiteRT) ---
        expected_audio_shape = audio_input_details[0]['shape']
        audio_data = np.random.rand(*expected_audio_shape).astype(np.float32) 
        
        audio_interpreter.set_tensor(audio_input_details[0]['index'], audio_data)
        audio_interpreter.invoke()
        audio_output = audio_interpreter.get_tensor(audio_output_details[0]['index'])
        audio_score = float(np.max(audio_output)) 

        # --- Sensor Fusion Decision Math ---
        fused_ai_score = (W_VISION * vision_score) + (W_AUDIO * audio_score)
        
        if pir_active:
            current_threshold = BASE_THRESHOLD - SENSITIVITY_BOOST
        else:
            current_threshold = BASE_THRESHOLD

        print(f"AI Fusion Score: {fused_ai_score:.2f} | Active Threshold: {current_threshold:.2f} (PIR: {pir_active})")

        # --- Alert Action, Image Capture, & MQTT Publish ---
        if fused_ai_score > current_threshold:
            print("\n🚨 INTRUDER CONFIRMED! (True Positive Breakdown) 🚨")
            
            # 1. Calculate weighted mathematical contributions
            vision_contribution = vision_score * W_VISION
            audio_contribution = audio_score * W_AUDIO
            
            total_weights = W_VISION + W_AUDIO
            vision_pct = (vision_contribution / total_weights) * 100
            audio_pct = (audio_contribution / total_weights) * 100

            # 2. Print a detailed math audit trail to the console
            print("=" * 50)
            print(f"FINAL FUSION SCORE : {fused_ai_score:.2f} (Required Threshold: {current_threshold:.2f})")
            print(f"PIR Sensor Status  : {'ACTIVE (Lowered threshold by ' + str(SENSITIVITY_BOOST) + ')' if pir_active else 'INACTIVE'}")
            print("-" * 50)
            print(f"🎥 Vision (YOLO11) : Raw Prob: {vision_score:.2f} -> Contributed: {vision_contribution:.2f} ({vision_pct:.1f}% of AI decision)")
            print(f"🔊 Audio (MobNet)  : Raw Prob: {audio_score:.2f} -> Contributed: {audio_contribution:.2f} ({audio_pct:.1f}% of AI decision)")
            print("=" * 50)
            
            # 3. Capture snapshot
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"intruder_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Snapshot saved as: {filename}")
            
            # 4. Construct JSON Payload for MQTT
            payload = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "ALERT",
                "fusion_score": round(fused_ai_score, 2),
                "threshold_used": round(current_threshold, 2),
                "pir_state": int(pir_active),
                "vision_raw": round(vision_score, 2),
                "vision_contribution": round(vision_contribution, 2),
                "audio_raw": round(audio_score, 2),
                "audio_contribution": round(audio_contribution, 2),
                "snapshot_file": filename
            }
            
            # Publish JSON string to MQTT broker
            mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
            print(f"📡 MQTT Alert Sent to topic '{MQTT_TOPIC}'\n")
            
            time.sleep(5) # Cooldown period to easily read console data before resuming

        time.sleep(0.1)

finally:
    video_capture.release()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()