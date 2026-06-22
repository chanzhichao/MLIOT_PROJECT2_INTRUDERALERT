import time
import datetime
import collections
import numpy as np
import pyaudio
import cv2
import onnxruntime as ort
import ai_edge_litert.interpreter as litert
import RPi.GPIO as GPIO

# ==========================================
# 1. HARDWARE CONFIGURATION
# ==========================================
BUZZER_PIN = 23  
LED_PIN = 17
PIR_PIN = 24  
CAMERA_INDEX = 0

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(LED_PIN, GPIO.OUT)       
GPIO.setup(PIR_PIN, GPIO.IN)

GPIO.output(BUZZER_PIN, GPIO.LOW) 
GPIO.output(LED_PIN, GPIO.LOW)

# ==========================================
# 2. AUDIO SLIDING WINDOW CONFIGURATION
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
# 3. INITIALIZE AI INTERFERENCE SESSIONS
# ==========================================
print("Loading MobileNetV2 Audio TFLite model...")
model_path = "/home/user/iot_project/Models/mobilenetv2_audio.tflite"
audio_interpreter = litert.Interpreter(model_path=model_path)
audio_interpreter.allocate_tensors()
audio_input_details = audio_interpreter.get_input_details()
audio_output_details = audio_interpreter.get_output_details()
expected_audio_shape = audio_input_details[0]['shape']

print("Loading YOLO11 Camera ONNX model...")
camera_session = ort.InferenceSession("Models/yolo11n.onnx")
camera_input_name = camera_session.get_inputs()[0].name

# Ensure V4L2 backend is specified to prevent Linux frame-buffer lag
video_capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

# ==========================================
# 4. TARGET AREA OF INTEREST & MATH HOOKS
# ==========================================
ART_AOI_BOX = [160, 160, 480, 480] 
last_capture_time = 0
CAPTURE_COOLDOWN = 3.0  

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
        pir_state = GPIO.input(PIR_PIN)
        pir_string = "YES" if pir_state == 1 else "NO"

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

        if len(abnormal_history) >= 30:
            mean_abnormal = np.mean(abnormal_history)
            std_abnormal = np.std(abnormal_history)
            std_abnormal = max(std_abnormal, 0.005)

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

            print(f"Tracking.. Z: {z_score:4.1f}σ | PIR: {pir_string} ", end='\r')

            # ==========================================
            # PRE-EVALUATE HARDWARE CONDITIONS
            # ==========================================
            Z_THRESHOLD = 2.0           
            CAM_THRESHOLD = 75.0        
            DECIBEL_TRIGGER_LIMIT = 80.0 

            cond_audio_spike = z_score > Z_THRESHOLD
            cond_pir_decibel = (pir_state == 1) and (current_db > DECIBEL_TRIGGER_LIMIT)

            # Check if hardware triggers say we need to run camera verification
            if cond_audio_spike or cond_pir_decibel:
                ret, frame = video_capture.read()
                cam_confidence = 0.0
                aoi_active = False

                if ret:
                    annotated_frame = cv2.resize(frame, (640, 640))
                    
                    # Run YOLO11 Real Inference
                    input_data = preprocess_frame(frame)
                    vision_output = camera_session.run(None, {camera_input_name: input_data})
                    raw_predictions = vision_output[0][0] 
                    class_confidences = raw_predictions[4:, :]
                    cam_confidence = float(np.max(class_confidences)) * 100.0 # Standardize to %
                    
                    if (cam_confidence / 100.0) > 0.40:
                        best_match_idx = np.argmax(np.max(class_confidences, axis=0))
                        box_coords = raw_predictions[0:4, best_match_idx]
                        
                        cx, cy, w, h = box_coords
                        human_box = [int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)]
                        
                        # Check dynamic box intersection against AOI boundary
                        aoi_active = check_intersection(human_box, ART_AOI_BOX)
                        
                        # Burn bounding box information onto the snapshot canvas copy
                        cv2.rectangle(annotated_frame, (human_box[0], human_box[1]), (human_box[2], human_box[3]), (255, 0, 0), 2)
                        cv2.putText(annotated_frame, f"Human Conf: {(cam_confidence/100.0):.2f}", (human_box[0], human_box[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Final Evaluation Boolean States
                cond_camera = cam_confidence > CAM_THRESHOLD
                
                # Check your custom Multi-Pathway condition rules
                if cond_camera or cond_audio_spike or cond_pir_decibel:
                    
                    # Fire physical warning pins
                    GPIO.output(BUZZER_PIN, GPIO.HIGH) 
                    GPIO.output(LED_PIN, GPIO.HIGH)        

                    # Handle saving marked image on alarm trip if cooldown allows
                    current_time = time.time()
                    if current_time - last_capture_time > CAPTURE_COOLDOWN:
                        box_color = (0, 0, 255) if aoi_active else (0, 255, 0)
                        
                        # Render tripwire boxes to file copy
                        cv2.rectangle(annotated_frame, (ART_AOI_BOX[0], ART_AOI_BOX[1]), (ART_AOI_BOX[2], ART_AOI_BOX[3]), box_color, 3)
                        cv2.putText(annotated_frame, "CRITICAL PERIMETER VIOLATION" if aoi_active else "PERIMETER SECURE", 
                                    (ART_AOI_BOX[0] + 10, ART_AOI_BOX[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                        
                        timestamp_fs = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                        filename = f"/home/user/iot_project/alerts/tripwire_breach_{timestamp_fs}.jpg"
                        cv2.imwrite(filename, annotated_frame)
                        last_capture_time = current_time

                    # Print out clear, text-only clean logs (No bounding box text parameters)
                    tripped_by = []
                    if cond_camera: tripped_by.append("CAMERA")
                    if cond_audio_spike: tripped_by.append("AUDIO_Z_SCORE")
                    if cond_pir_decibel: tripped_by.append("PIR+DECIBEL")
                    pathway_string = " + ".join(tripped_by)

                    now = datetime.datetime.now()
                    print("\n\n================================================")
                    print(f"- Time ({now.strftime('%d%m%y %H%M%S')}) [TRIPPED BY: {pathway_string}]")
                    print(f"- AUDIO(confidence STD: {std_abnormal:.4f}, MAX delta decibel: {current_db:.1f}dB, best class 1: {top1_name.upper()} ({top1_prob:.1f}%), best class 2: {top2_name.upper()} ({top2_prob:.1f}%))")
                    print(f"- PIR_DETECT? {pir_string}")
                    print(f"- CAMERA Confidence: {cam_confidence:.1f}%")
                    print(f"- CAMERA SNAPSHOT (Confidence: {cam_confidence:.1f}%)")
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