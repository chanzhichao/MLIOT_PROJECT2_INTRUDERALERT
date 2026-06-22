import cv2
import numpy as np
import onnxruntime as ort
import ai_edge_litert.interpreter as litert
import time
import pyaudio
from gpiozero import DigitalInputDevice, LED, Buzzer

# ==========================================
# 1. HARDWARE OUTPUT & INPUT CONFIGURATION
# ==========================================
PIR_SENSOR_PIN = 18
pir_hardware = DigitalInputDevice(PIR_SENSOR_PIN)

# Actuators
led_bulb = LED(17)
buzzer = Buzzer(23)

# Camera Configuration
CAMERA_INDEX = 0
video_capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

# PyAudio Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 1024

audio_interface = pyaudio.PyAudio()
audio_stream = audio_interface.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)

# PIR State Engine Tracking
pir_state = False
last_motion_time = 0
DECAY_DURATION = 2.0

def read_pir_sensor():
    global pir_state, last_motion_time
    if pir_hardware.value == 1:
        pir_state = True
        last_motion_time = time.time()
    else:
        if pir_state and (time.time() - last_motion_time > DECAY_DURATION):
            pir_state = False
    return pir_state

# ==========================================
# 2. RUNTIME CALIBRATIONS & MODEL CORES
# ==========================================
# Audio Spike Settings
audio_amp = 10.0
background_amplitude = 100.0
ALPHA = 0.15

NO_MOTION_THRESHOLD_DB = 12.0  
MOTION_THRESHOLD_DB = 9.00     

# 🎯 DEFINE YOUR VIRTUAL TRIPWIRE BOUNDS (Mapped on 640x640 preprocessed space)
ART_AOI_BOX = [160, 160, 480, 480]  # [xmin, ymin, xmax, ymax]

# 7-Class Audio Model Mapping Array
AUDIO_LABELS = [
    "class1", 
    "class2", 
    "class3", 
    "class4",
    "class5",
    "class6",
    "class7"
]

print("Loading AI Inference Engine Models...")
# Initialize YOLO11 ONNX Bounding Box Session
camera_session = ort.InferenceSession("Models/yolo11n.onnx")
camera_input_name = camera_session.get_inputs()[0].name

# Initialize MobileNetV2 Audio LiteRT Session
audio_interpreter = litert.Interpreter(model_path="Models/mobilenetv2_audio.tflite")
audio_interpreter.allocate_tensors()
audio_input_details = audio_interpreter.get_input_details()
audio_output_details = audio_interpreter.get_output_details()
expected_audio_shape = audio_input_details[0]['shape']

# ==========================================
# 3. HELPER PREPROCESSING FUNCTIONS
# ==========================================
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

# ==========================================
# 4. INITIALIZATION WARMUP TIMING
# ==========================================
print("=" * 70)
print("WARMING UP SENSORS & FLOORS: Calibrating baseline floor...")
warmup_start = time.time()
while time.time() - warmup_start < 3.0:
    try:
        audio_data = audio_stream.read(CHUNK, exception_on_overflow=False)
        signal_samples = np.frombuffer(audio_data, dtype=np.int16)
        background_amplitude = (0.50 * float(np.max(np.abs(signal_samples)))) + (0.50 * background_amplitude)
    except Exception:
        continue
print(f"✅ Baseline Stable: {background_amplitude:.1f} | Master System Fully Armed.")
print("=" * 70)

# ==========================================
# 5. INTEGRATED DECISION MULTI-MODAL LOOP
# ==========================================
frame_counter = 0
vision_score = 0.0
aoi_active = False
human_box = [0, 0, 0, 0]

try:
    while True:
        current_time = time.strftime("%H:%M:%S")
        led_bulb.off()
        buzzer.off()

        # --- 1. SENSOR FUSION ENGINE: PIR + Audio Spike ---
        pir_active = read_pir_sensor()
        try:
            audio_data = audio_stream.read(CHUNK, exception_on_overflow=False)
            signal_samples = np.frombuffer(audio_data, dtype=np.int16)
            raw_mic_amplitude = float(np.max(np.abs(signal_samples)))
        except Exception:
            continue

        safe_mic = max(1.0, raw_mic_amplitude)
        safe_background = max(400.0, background_amplitude)
        db_interval_change = audio_amp * np.log10(safe_mic / safe_background)
        
        active_db_threshold = MOTION_THRESHOLD_DB if pir_active else NO_MOTION_THRESHOLD_DB
        spike_triggered = db_interval_change > active_db_threshold

        if not spike_triggered:
            background_amplitude = (ALPHA * safe_mic) + ((1.0 - ALPHA) * background_amplitude)

        # --- 2. VISION ENGINE: YOLO11 ONNX + AOI Bounds ---
        ret, frame = video_capture.read()
        if not ret:
            print("[Warning] Camera frame dropped.")
            continue

        frame_counter += 1
        if frame_counter % 5 == 0:  # Calculate vision inferences every 5 frames
            input_data = preprocess_frame(frame)
            vision_output = camera_session.run(None, {camera_input_name: input_data})
            raw_predictions = vision_output[0][0]
            
            class_confidences = raw_predictions[4:, :]
            vision_score = float(np.max(class_confidences))
            
            aoi_active = False
            if vision_score > 0.30:
                best_match_idx = np.argmax(np.max(class_confidences, axis=0))
                box_coords = raw_predictions[0:4, best_match_idx]
                
                cx, cy, w, h = box_coords
                human_box = [int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)]
                aoi_active = check_intersection(human_box, ART_AOI_BOX)
            frame_counter = 0

        # --- 3. AUDIO NET ENGINE: MobileNetV2 Classification ---
        # Reshape physical PCM buffers to fit expected model runtime input array shapes
        audio_tensor_input = np.zeros(expected_audio_shape, dtype=np.float32)
        # Fill slice up to model length capability
        filled_length = min(len(signal_samples), audio_tensor_input.size)
        audio_tensor_input.flat[:filled_length] = signal_samples[:filled_length].astype(np.float32) / 32768.0

        audio_interpreter.set_tensor(audio_input_details[0]['index'], audio_tensor_input)
        audio_interpreter.invoke()
        audio_output = audio_interpreter.get_tensor(audio_output_details[0]['index'])[0]
        
        best_audio_idx = int(np.argmax(audio_output))
        audio_confidence = float(audio_output[best_audio_idx])
        audio_class_name = AUDIO_LABELS[best_audio_idx]

        # 🚀 THE CORRECTED BYPASS CONDITION:
        # This explicitly stops Class 0 (Ambient), Class 5 (Footsteps), and Class 6 (Door Slam) from triggering.
        audio_model_triggered = (best_audio_idx != 0) and (best_audio_idx != 5) and (best_audio_idx != 6) and (audio_confidence > 0.65)

        # ==========================================
        # ⚠️ MASTER CASCADING DECISION OR LOGIC
        # ==========================================
        if aoi_active or audio_model_triggered or spike_triggered:
            # Trigger Physical Actuators
            led_bulb.on()
            buzzer.on()

            # Map the visual coordinate data back to original display layout coordinates
            orig_h, orig_w, _ = frame.shape
            scale_x = orig_w / 640.0
            scale_y = orig_h / 640.0

            # Draw Virtual Tripwire AOI (Yellow)
            cv2.rectangle(frame, (int(ART_AOI_BOX[0]*scale_x), int(ART_AOI_BOX[1]*scale_y)), 
                          (int(ART_AOI_BOX[2]*scale_x), int(ART_AOI_BOX[3]*scale_y)), (0, 255, 255), 2)
            
            # Draw Attacker Human Bounding Box if active (Red)
            if vision_score > 0.30:
                cv2.rectangle(frame, (int(human_box[0]*scale_x), int(human_box[1]*scale_y)), 
                              (int(human_box[2]*scale_x), int(human_box[3]*scale_y)), (0, 0, 255), 3)
                cv2.putText(frame, f"Intruder: {vision_score*100:.1f}%", (int(human_box[0]*scale_x), int(human_box[1]*scale_y)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Consolidated Project Telemetry Stream Log Output
            print("\n" + "="*70)
            print("🚨 Master Security Alert Triggered! 🚨")
            print("="*70)
            print(f"Time                        : [{current_time}]")
            print(f"PIR Status                  : {'ACTIVE (Zone Breached)' if pir_active else 'INACTIVE'}")
            print(f"Audio Spike Delta           : {db_interval_change:.3f} dB (Limit: {active_db_threshold:.2f} dB)")
            print(f"Camera Class Confidence     : {vision_score * 100:.1f}%")
            print(f"Camera Virtual AOI Breach   : {'BREACHED' if aoi_active else 'SECURE'}")
            print(f"Audio Classification Event  : {audio_class_name} ({audio_confidence * 100:.1f}%)")
            print("-" * 70)

            # Write high-evidence snapshot image to disk
            snapshot_filename = f"security_breach_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(snapshot_filename, frame)
            print(f"📸 Snapshot preserved locally as: {snapshot_filename}\n")

            time.sleep(1.5)  # Incident lock cooldown period
        else:
            print(f"Scanning Space... [PIR: {int(pir_active)} | Spike: {db_interval_change:.1f}dB | Vision Conf: {vision_score:.2f} | Audio Class: {audio_class_name}]", end="\r")

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nShutting down master security loop system...")
finally:
    led_bulb.off()
    buzzer.off()
    video_capture.release()
    audio_stream.stop_stream()
    audio_stream.close()
    audio_interface.terminate()
    print("🔄 Actuators isolated and safe. Hardware closed down successfully.")