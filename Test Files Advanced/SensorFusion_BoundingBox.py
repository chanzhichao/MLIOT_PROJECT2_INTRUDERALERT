import cv2
import numpy as np
import onnxruntime as ort
import ai_edge_litert.interpreter as litert
import time
import json
import paho.mqtt.client as mqtt

# ==========================================
# 1. HARDWARE & MQTT SETTINGS
# ==========================================
CAMERA_INDEX = 0      

# MQTT Broker Configuration
MQTT_BROKER = "test.mosquitto.org"  
MQTT_PORT = 1883
MQTT_TOPIC = "home/security/intruder"

# ==========================================
# 2. INITIALIZE MQTT CLIENT
# ==========================================
print("Connecting to MQTT Broker...")
mqtt_client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)

try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start() 
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
# 4. HELPER PREPROCESSING & INTERSECTION MATH
# ==========================================
def preprocess_frame(frame):
    resized = cv2.resize(frame, (640, 640))
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = np.float32(rgb_img) / 255.0
    blob = np.transpose(blob, (2, 0, 1))  
    blob = np.expand_dims(blob, axis=0)   
    return blob

def check_intersection(box_a, box_b):
    """
    Checks if Box A (Human) intersects with Box B (Area of Interest).
    Format for boxes: [xmin, ymin, xmax, ymax]
    """
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
    
    # Calculate area of intersection rectangle
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    return inter_area > 0

# ==========================================
# 5. PIR + SPIKE DOMINANT CONFIGURATION & VIRTUAL TRIPWIRE
# ==========================================
BASE_THRESHOLD = 0.75     
SENSITIVITY_BOOST = 0.30  

# Vector Weight Matrix
W_VISION = 0.40            
W_AUDIO_CLASS = 0.20       
W_AUDIO_SPIKE = 0.60       

# Audio Settings
background_audio_level = 0.10  
ALPHA = 0.01                   
SPIKE_SENSITIVITY_MULTIPLIER = 1.2  

# Performance Optimization Trackers
frame_counter = 0
vision_score = 0.0  
aoi_active = False  # Acts exactly like our old pir_active flag

# 🎯 DEFINE YOUR AREA OF INTEREST (AOI) BOUNDS HERE
# Coordinates mapped on the 640x640 preprocessed window: [xmin, ymin, xmax, ymax]
# This example box creates a protective bounding square directly in the center frame
ART_AOI_BOX = [160, 160, 480, 480] 

# ==========================================
# 6. EXECUTION LOOP
# ==========================================
try:
    print("Security System successfully armed. Monitoring virtual tripwire...")
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Warning: Camera stream disconnected.")
            continue

        # --- FRAME SKIPPING OPTIMIZATION ---
        frame_counter += 1
        if frame_counter % 5 == 0:  
            input_data = preprocess_frame(frame)
            vision_output = camera_session.run(None, {camera_input_name: input_data})
            
            raw_predictions = vision_output[0][0] 
            
            # Extract scores and bounding coordinates
            # YOLO11 outputs format: [cx, cy, w, h, class_0_conf, class_1_conf, ...]
            class_confidences = raw_predictions[4:, :]
            vision_score = float(np.max(class_confidences)) 
            
            # Default state to False unless an intersection is discovered
            aoi_active = False
            
            # If a detection is meaningful, extract its coordinates to test intersection
            if vision_score > 0.30:
                # Target top detection index
                best_match_idx = np.argmax(np.max(class_confidences, axis=0))
                box_coords = raw_predictions[0:4, best_match_idx]
                
                # Convert YOLO center format [cx, cy, w, h] to boundary edges [xmin, ymin, xmax, ymax]
                cx, cy, w, h = box_coords
                human_box = [int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)]
                
                # Evaluate virtual intersection
                aoi_active = check_intersection(human_box, ART_AOI_BOX)
                
            frame_counter = 0  # Flush loop counter
        
        # --- AUDIO INFERENCE & DSP VECTOR ANALYSIS ---
        expected_audio_shape = audio_input_details[0]['shape']
        
        if np.random.random() > 0.95:
            print("\n🔊 [SIMULATION] Injecting a sudden loud screaming spike!")
            raw_audio_score = 0.95  
        else:
            raw_audio_score = np.random.uniform(0.10, 0.15)

        audio_class_score = raw_audio_score 
        raw_delta = raw_audio_score - background_audio_level
        audio_spike_score = max(0.0, raw_delta * SPIKE_SENSITIVITY_MULTIPLIER)
        audio_spike_score = min(1.0, audio_spike_score)  

        background_audio_level = (ALPHA * raw_audio_score) + ((1.0 - ALPHA) * background_audio_level)

        # --- 4D VECTOR SENSOR FUSION DECISION MATH ---
        fused_ai_score = (W_VISION * vision_score) + \
                         (W_AUDIO_CLASS * audio_class_score) + \
                         (W_AUDIO_SPIKE * audio_spike_score)
        
        if aoi_active:
            current_threshold = BASE_THRESHOLD - SENSITIVITY_BOOST
        else:
            current_threshold = BASE_THRESHOLD

        # --- ALERT VERIFICATION LANE ---
        if fused_ai_score > current_threshold:
            print("\n🚨 VIRTUAL PERIMETER BREACH DETECTED! 🚨")
            
            vision_contrib = vision_score * W_VISION
            audio_class_contrib = audio_class_score * W_AUDIO_CLASS
            audio_spike_contrib = audio_spike_score * W_AUDIO_SPIKE
            total_contrib = vision_contrib + audio_class_contrib + audio_spike_contrib
            
            vision_pct = (vision_contrib / total_contrib * 100) if total_contrib > 0 else 0

            print("=" * 60)
            print(f"4D FUSED SCORE       : {fused_ai_score:.2f} (Required Limit: {current_threshold:.2f})")
            print(f"Virtual Perimeter Zone: {'BREACHED (Inside Area of Interest)' if aoi_active else 'SECURE'}")
            print("-" * 60)
            print(f"🎥 Dim 1: Human Classification : Raw: {vision_score:.2f} -> Weighted: {vision_contrib:.2f} ({vision_pct:.1f}%)")
            print("=" * 60)
            
            # Image Capture Action
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"tripwire_incident_{timestamp}.jpg"
            
            # Optional: Draw the red boundary ring overlay directly onto the saved proof file
            cv2.rectangle(frame, (ART_AOI_BOX[0], ART_AOI_BOX[1]), (ART_AOI_BOX[2], ART_AOI_BOX[3]), (0, 0, 255), 3)
            cv2.imwrite(filename, frame)
            print(f"📸 Snapshot Saved: {filename}")
            
            # Send Telemetry
            payload = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "ALERT_TRIPWIRE",
                "aoi_breached": int(aoi_active),
                "fusion_score": round(fused_ai_score, 2),
                "evidence_snapshot": filename
            }
            mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
            print(f"📡 MQTT Dispatch Sent to '{MQTT_TOPIC}'\n")
            
            time.sleep(5)  

        time.sleep(0.01)

finally:
    video_capture.release()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("🔄 OS Systems Safely Restored.")
    # Print this out right after allocating tensors to inspect output node metadata