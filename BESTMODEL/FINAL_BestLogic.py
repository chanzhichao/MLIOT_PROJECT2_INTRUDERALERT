import time
import datetime
import collections
import json
import numpy as np
import pyaudio
import cv2
import onnxruntime as ort
import ai_edge_litert.interpreter as litert
import RPi.GPIO as GPIO
import paho.mqtt.client as mqtt  

# ==========================================
# 1. HARDWARE CONFIGURATION
# ==========================================
BUZZER_PIN = 23  
LED_PIN = 17
PIR_PIN = 18  
CAMERA_INDEX = 0

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(LED_PIN, GPIO.OUT)       
GPIO.setup(PIR_PIN, GPIO.IN)

GPIO.output(BUZZER_PIN, GPIO.LOW) 
GPIO.output(LED_PIN, GPIO.LOW)

# ==========================================
# 2. MQTT NETWORK TELEMETRY CONFIGURATION
# ==========================================
MQTT_BROKER = "broker.hivemq.com"  
MQTT_PORT = 1883
MQTT_TOPIC = "home/security/fusion_engine"

print("Connecting to MQTT Broker...")
mqtt_client = mqtt.Client()
try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()  
    print(f"✅ MQTT Connected to {MQTT_BROKER} on topic: {MQTT_TOPIC}")
except Exception as e:
    print(f"⚠️ MQTT Connection failed ({e}). Running in local hardware mode only.")

# ==========================================
# 3. AUDIO & SLIDING WINDOW CONFIGURATION
# ==========================================
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000  
CHUNK = 2048  

ONE_SECOND_TOTAL_SAMPLES = 16000
audio_rolling_buffer = collections.deque(maxlen=ONE_SECOND_TOTAL_SAMPLES)
abnormal_history = collections.deque(maxlen=50)

AUDIO_LABELS = ["class1", "class2", "class3", "class4", "class5", "class6", "class7"]

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK)

# ==========================================
# 4. LOAD AI MODELS (CORRECT SIDE-BY-SIDE PATHS)
# ==========================================
print("Loading MobileNetV2 Audio TFLite model...")
model_path = "/home/user/iot_project/BESTMODEL/Models/mobilenetv2_audio.tflite"
audio_interpreter = litert.Interpreter(model_path=model_path)
audio_interpreter.allocate_tensors()
audio_input_details = audio_interpreter.get_input_details()
audio_output_details = audio_interpreter.get_output_details()
expected_audio_shape = audio_input_details[0]['shape']

print("Loading YOLO11 Camera ONNX model...")
camera_session = ort.InferenceSession("/home/user/iot_project/BESTMODEL/Models/yolo11n.onnx")
camera_input_name = camera_session.get_inputs()[0].name

video_capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ==========================================
# 5. MATH HOOKS & TARGET DEFINITION
# ==========================================
ART_AOI_BOX = [160, 160, 480, 480] 
last_capture_time = 0
CAPTURE_COOLDOWN = 3.0  

# 🚀 PIR Software Decay Variables
PIR_DECAY_TIME = 3.0       # Seconds to hold the presence state active
pir_expiration_time = 0.0   # Timestamp tracking when the decay window drops

def preprocess_frame(frame):
    resized = cv2.resize(frame, (640, 640))
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = np.float32(rgb_img) / 255.0
    blob = np.transpose(blob, (2, 0, 1))  
    blob = np.expand_dims(blob, axis=0)   
    return blob

def check_intersection(box_a, box_b):
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    return inter_area > 0

print("🏠 High-Performance Fused Security Engine Active & Calibrating...")
time.sleep(3)
print("⚡ Monitoring Guard Active.")

try:
    while True:
        # 🟢 STEP 1: PIR DETECTION WITH SOFTWARE DECAY
        raw_pir = GPIO.input(PIR_PIN)
        current_time = time.time()

        if raw_pir == 1:
            pir_expiration_time = current_time + PIR_DECAY_TIME

        pir_active = current_time < pir_expiration_time
        pir_string = "YES" if pir_active else "NO"

        # 🟢 STEP 2: AUDIO MIC CAPTURE
        try:
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            signal_samples = np.frombuffer(raw_data, dtype=np.int16)
        except IOError:
            continue  

        rms = np.sqrt(np.mean(signal_samples.astype(np.float32)**2)) if len(signal_samples) > 0 else 0
        current_db = 20 * np.log10(rms) if rms > 0 else 0

        downsampled_samples = signal_samples[::3].astype(np.float32) / 32768.0
        audio_rolling_buffer.extend(downsampled_samples)

        if len(audio_rolling_buffer) < ONE_SECOND_TOTAL_SAMPLES:
            continue

        full_second_snapshot = np.array(audio_rolling_buffer)
        audio_tensor_input = np.zeros(expected_audio_shape, dtype=np.float32)
        filled_length = min(len(full_second_snapshot), audio_tensor_input.size)
        audio_tensor_input.flat[:filled_length] = full_second_snapshot[:filled_length]

        audio_interpreter.set_tensor(audio_input_details[0]['index'], audio_tensor_input)
        audio_interpreter.invoke()
        raw_logits = audio_interpreter.get_tensor(audio_output_details[0]['index'])[0]

        exp_logits = np.exp(raw_logits - np.max(raw_logits))
        audio_probabilities = exp_logits / np.sum(exp_logits)

        normal_pool = audio_probabilities[0] + audio_probabilities[5] + audio_probabilities[6]
        abnormal_pool = 1.0 - normal_pool
        
        is_anomaly = False
        if len(abnormal_history) >= 30:
            current_mean = np.mean(abnormal_history)
            current_std = max(np.std(abnormal_history), 0.005)
            temp_z = (abnormal_pool - current_mean) / current_std
            if temp_z > 1.2 and abnormal_pool > 0.01:
                is_anomaly = True

        if not is_anomaly:
            abnormal_history.append(abnormal_pool)

        z_score = 0.0
        std_abnormal = 0.005
        top1_name, top2_name = "NONE", "NONE"
        top1_prob, top2_prob = 0.0, 0.0

        if len(abnormal_history) >= 30:
            mean_abnormal = np.mean(abnormal_history)
            std_abnormal = max(np.std(abnormal_history), 0.005)
            z_score = (abnormal_pool - mean_abnormal) / std_abnormal

            sorted_indices = list(np.argsort(audio_probabilities))
            if 5 in sorted_indices:
                sorted_indices.remove(5)
            
            top1_idx = sorted_indices[-1]
            top2_idx = sorted_indices[-2]
            top1_name = AUDIO_LABELS[top1_idx]
            top1_prob = audio_probabilities[top1_idx] * 100
            top2_name = AUDIO_LABELS[top2_idx]
            top2_prob = audio_probabilities[top2_idx] * 100

        # 🟢 STEP 3: RUN CAMERA SCANNING CORE
        ret, frame = video_capture.read()
        cam_confidence = 0.0
        aoi_active = False

        if ret:
            annotated_frame = cv2.resize(frame, (640, 640))
            input_data = preprocess_frame(frame)
            vision_output = camera_session.run(None, {camera_input_name: input_data})
            raw_predictions = vision_output[0][0] 
            class_confidences = raw_predictions[4:, :]
            cam_confidence = float(np.max(class_confidences)) * 100.0 
            
            if (cam_confidence / 100.0) > 0.40:
                best_match_idx = np.argmax(np.max(class_confidences, axis=0))
                box_coords = raw_predictions[0:4, best_match_idx]
                cx, cy, w, h = box_coords
                human_box = [int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)]
                
                aoi_active = check_intersection(human_box, ART_AOI_BOX)
                
                # Burn blue bounding box and target text inside it
                cv2.rectangle(annotated_frame, (human_box[0], human_box[1]), (human_box[2], human_box[3]), (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"Human Conf: {(cam_confidence/100.0):.2f}", (human_box[0] + 10, human_box[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        print(f"Tracking.. Z: {z_score:4.1f}σ | PIR: {pir_string} | CAM: {cam_confidence:.1f}% ", end='\r')

        # ==========================================
        # 🚨 STEP 4: THE 3 PATHWAY THRESHOLD EVALUATION
        # ==========================================
        Z_THRESHOLD = 2.0           
        DECIBEL_TRIGGER_LIMIT = 80.0 

        cond_camera = (cam_confidence > 80.0) and aoi_active
        cond_audio_spike = z_score > Z_THRESHOLD
        cond_pir_decibel = pir_active and (current_db > DECIBEL_TRIGGER_LIMIT)  # 🚀 LATCHED

        # MULTI-PATHWAY ALARM MASTER GATE EXECUTION
        if cond_camera or cond_audio_spike or cond_pir_decibel:
            
            GPIO.output(BUZZER_PIN, GPIO.HIGH) 
            GPIO.output(LED_PIN, GPIO.HIGH)        

            filename = "None"
            current_time_snap = time.time()
            if current_time_snap - last_capture_time > CAPTURE_COOLDOWN:
                box_color = (0, 0, 255) if aoi_active else (0, 255, 0)
                cv2.rectangle(annotated_frame, (ART_AOI_BOX[0], ART_AOI_BOX[1]), (ART_AOI_BOX[2], ART_AOI_BOX[3]), box_color, 3)
                cv2.putText(annotated_frame, "CRITICAL PERIMETER VIOLATION" if aoi_active else "PERIMETER SECURE", 
                            (ART_AOI_BOX[0] + 10, ART_AOI_BOX[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                
                timestamp_fs = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"/home/user/iot_project/BESTMODEL/alerts/tripwire_breach_{timestamp_fs}.jpg"
                cv2.imwrite(filename, annotated_frame)
                last_capture_time = current_time_snap

            tripped_by = []
            if cond_camera: tripped_by.append("CAMERA")
            if cond_audio_spike: tripped_by.append("AUDIO_Z_SCORE")
            if cond_pir_decibel: tripped_by.append("PIR+DECIBEL")
            pathway_string = " + ".join(tripped_by)

            now = datetime.datetime.now()
            time_string = now.strftime('%d%m%y %H%M%S')

            # COMPILE MQTT METRIC JSON PAYLOAD
            alert_payload = {
                "timestamp": time_string,
                "tripped_pathways": tripped_by,
                "audio": {
                    "z_score": round(float(z_score), 2),
                    "std_deviation": round(float(std_abnormal), 4),
                    "max_delta_db": round(float(current_db), 1),
                    "best_class_1": f"{top1_name.upper()} ({top1_prob:.1f}%)",
                    "best_class_2": f"{top2_name.upper()} ({top2_prob:.1f}%)"
                },
                "pir_detected": bool(pir_active),
                "camera": {
                    "confidence_pct": round(cam_confidence, 1),
                    "aoi_breached": aoi_active,
                    "saved_snapshot": filename
                }
            }

            try:
                mqtt_client.publish(MQTT_TOPIC, json.dumps(alert_payload), qos=1)
            except Exception as net_error:
                print(f"\n⚠️ MQTT Publish failed: {net_error}")

            # 📋 CLEAN TELEMETRY TEXT OUTPUT TO CONSOLE
            print("\n\n================================================")
            print(f"- Time ({time_string}) [TRIPPED BY: {pathway_string}]")
            print(f"- AUDIO(confidence STD: {std_abnormal:.4f}, MAX delta decibel: {current_db:.1f}dB, best class 1: {top1_name.upper()} ({top1_prob:.1f}%), best class 2: {top2_name.upper()} ({top2_prob:.1f}%))")
            print(f"- PIR_DETECT? {pir_string}")
            print(f"- CAMERA Confidence: {cam_confidence:.1f}%")
            print(f"- CAMERA SNAPSHOT (Confidence: {cam_confidence:.1f}%)")
            print("📡 MQTT Network Status Update Sent successfully.")
            print("================================================\n")

            time.sleep(0.5) 

            GPIO.output(BUZZER_PIN, GPIO.LOW)
            GPIO.output(LED_PIN, GPIO.LOW)         
            
            audio_rolling_buffer.clear()
            time.sleep(1.0) 

        time.sleep(0.02)

except KeyboardInterrupt:
    print("\nShutting down engine cleanly.")
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.cleanup()
    stream.stop_stream()
    stream.close()
    p.terminate()
    video_capture.release()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()