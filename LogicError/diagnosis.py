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
# 2. RUNTIME CALIBRATIONS & NON-BLOCKING TIMERS
# ==========================================
audio_amp = 10.0
background_amplitude = 100.0
ALPHA = 0.15

NO_MOTION_THRESHOLD_DB = 12.0  
MOTION_THRESHOLD_DB = 9.00     

# Virtual Tripwire Bounds (Mapped on 640x640 preprocessed space)
ART_AOI_BOX = [160, 160, 480, 480]  

# Non-Blocking Actuator Parameters
last_alarm_time = 0          
BUZZER_DURATION = 0.5        # Sounds for 0.5 seconds
BUZZER_COOLDOWN = 4.0        # Cannot sound again for 4 seconds
actuators_active = False     

# 7-Class Audio Model Mapping Array
AUDIO_LABELS = [
    "Ambient_Background", 
    "Scream", 
    "Glass_Shatter", 
    "Alarm_Buzzer",
    "Clap",
    "Footsteps",
    "Door_Slam"
]

print("Loading AI Inference Engine Models...")
camera_session = ort.InferenceSession("Models/yolo11n.onnx")
camera_input_name = camera_session.get_inputs()[0].name

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
# 4. INITIALIZATION WARMUP
# ==========================================
print("=" * 80)
print("WARMING UP SENSORS: Calibrating baseline floor...")
warmup_start = time.time()
while time.time() - warmup_start < 3.0:
    try:
        audio_data = audio_stream.read(CHUNK, exception_on_overflow=False)
        signal_samples = np.frombuffer(audio_data, dtype=np.int16)
        background_amplitude = (0.50 * float(np.max(np.abs(signal_samples)))) + (0.50 * background_amplitude)
    except Exception:
        continue
print(f"✅ Baseline Stable: {background_amplitude:.1f} | Master Diagnostics Initialized.")
print("=" * 80)

# ==========================================
# 5. DIAGNOSTIC INFRASTRUCTURE LIVE TRACE LOOP
# ==========================================
frame_counter = 0
vision_score = 0.0
aoi_active = False
human_box = [0, 0, 0, 0]

# Print Matrix Header
print(f"{'TIME':<10} | {'PIR':<3} | {'SPIKE DELTA':<12} | {'VIS CONF':<8} | {'AOI INTR':<8} | {'AUDIO INFERENCE':<22} | {'BUZZ PIN'}")
print("-" * 95)

try:
    while True:
        current_time_str = time.strftime("%H:%M:%S")
        now = time.time()

        # Non-blocking actuator safety manager
        if actuators_active and (now - last_alarm_time > BUZZER_DURATION):
            led_bulb.off()
            buzzer.off()
            actuators_active = False
            try:
                # Flush microphone buffer entirely to remove buzzer noise artifacts
                while audio_stream.get_read_available() > 0:
                    audio_stream.read(CHUNK, exception_on_overflow=False)
            except Exception:
                pass

        # --- 1. SENSOR FUSION DATA CAPTURE ---
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

        # --- 2. VISION DATA CAPTURE ---
        ret, frame = video_capture.read()
        if not ret:
            continue

        frame_counter += 1
        if frame_counter % 5 == 0:  
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

        # --- 3. AUDIO MACHINE LEARNING DATA CAPTURE ---
        audio_tensor_input = np.zeros(expected_audio_shape, dtype=np.float32)
        filled_length = min(len(signal_samples), audio_tensor_input.size)
        audio_tensor_input.flat[:filled_length] = signal_samples[:filled_length].astype(np.float32) / 32768.0

        audio_interpreter.set_tensor(audio_input_details[0]['index'], audio_tensor_input)
        audio_interpreter.invoke()
        audio_output = audio_interpreter.get_tensor(audio_output_details[0]['index'])[0]
        
        best_audio_idx = int(np.argmax(audio_output))
        audio_confidence = float(audio_output[best_audio_idx])
        audio_class_name = AUDIO_LABELS[best_audio_idx]

        # Explicit Class 0 and Class 6 (Door Slam) bypass rule
        audio_model_triggered = (best_audio_idx != 0) and (best_audio_idx != 6) and (audio_confidence > 0.65)

        # Evaluate timing permission
        buzzer_allowed = (now - last_alarm_time > BUZZER_COOLDOWN)
        
        # Determine master trigger flag state
        any_module_triggered = aoi_active or audio_model_triggered or spike_triggered

        # ==========================================
        # EXECUTE MASTER ALARM DISPATCH
        # ==========================================
        buzzer_fired_this_frame = False
        if any_module_triggered:
            if buzzer_allowed and not actuators_active:
                led_bulb.on()
                buzzer.on()
                actuators_active = True
                last_alarm_time = now
                buzzer_fired_this_frame = True
            
            # Continuous back-end recording pipeline
            orig_h, orig_w, _ = frame.shape
            scale_x = orig_w / 640.0
            scale_y = orig_h / 640.0
            cv2.rectangle(frame, (int(ART_AOI_BOX[0]*scale_x), int(ART_AOI_BOX[1]*scale_y)), 
                          (int(ART_AOI_BOX[2]*scale_x), int(ART_AOI_BOX[3]*scale_y)), (0, 255, 255), 2)
            if vision_score > 0.30:
                cv2.rectangle(frame, (int(human_box[0]*scale_x), int(human_box[1]*scale_y)), 
                              (int(human_box[2]*scale_x), int(human_box[3]*scale_y)), (0, 0, 255), 3)
            
            snapshot_filename = f"security_breach_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(snapshot_filename, frame)

        # ==========================================
        # 🎯 REAL-TIME MATRIX DIAGNOSTIC LINE PRINT
        # ==========================================
        # Maps every active sensor array configuration output directly into scannable slots
        spike_status = f"{db_interval_change:>5.2f}/{active_db_threshold:<5.1f}"
        aoi_status = "BREACH" if aoi_active else "CLEAR"
        audio_status = f"{audio_class_name[:12]}({audio_confidence:.2f})"
        buzzer_status = "⚠️ FIRED!" if buzzer_fired_this_frame else ("ON" if actuators_active else "OFF")

        print(f"{current_time_str:<10} | {int(pir_active):<3} | {spike_status:<12} | {vision_score:<8.2f} | {aoi_status:<8} | {audio_status:<22} | {buzzer_status}")

        time.sleep(0.02)  # Controlled thread interval rate

except KeyboardInterrupt:
    print("\nDiagnostics paused safely.")
finally:
    led_bulb.off()
    buzzer.off()
    video_capture.release()
    audio_stream.stop_stream()
    audio_stream.close()
    audio_interface.terminate()
    print("🔄 Systems isolated. Offline.")
    