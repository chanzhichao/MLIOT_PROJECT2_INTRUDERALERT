import numpy as np
import time
import pyaudio
import cv2  # For Image Classification Bounding Boxes
from gpiozero import DigitalInputDevice, LED, Buzzer

# ==========================================
# 1. HARDWARE OUTPUT & INPUT CONFIGURATION
# ==========================================
PIR_SENSOR_PIN = 18
pir_hardware = DigitalInputDevice(PIR_SENSOR_PIN)

# Actuators
led_bulb = LED(17)
buzzer = Buzzer(23)

# PyAudio Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 1024

audio_interface = pyaudio.PyAudio()
audio_stream = audio_interface.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)

# PIR State Engine
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
# 2. CALIBRATION KNOBS & ML THRESHOLDS
# ==========================================
audio_amp = 10.0
background_amplitude = 100.0
ALPHA = 0.15

# Core Thresholds
NO_MOTION_THRESHOLD_DB = 12.0
MOTION_THRESHOLD_DB = 9.00

# ML Inference Configurations
CONFIDENCE_THRESHOLD_CAM = 0.70
CONFIDENCE_THRESHOLD_AUD = 0.65
THREAT_AUDIO_CLASSES = ["Scream", "Groan", "Glass Shatter", "Explosion"]
THREAT_CAMERA_CLASSES = ["person", "vandal", "intruder"]

# Define Area of Interest (Normalized coordinates: y_min, x_min, y_max, x_max)
AOI = [0.2, 0.2, 0.8, 0.8] 

# ==========================================
# 3. MOCK MACHINE LEARNING PIPELINES
# ==========================================
def run_camera_classification_inference():
    """
    PLACEHOLDER: Replace this with your live camera capture loop,
    YOLOv8 inference frame, or MobileNet tensor engine execution.
    """
    # Simulate an empty canvas
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Simulating a mock detection occasionally
    if np.random.random() > 0.98:
        detected_class = "person"
        confidence = np.random.uniform(0.75, 0.95)
        # Mock bounding box inside the frame [x, y, w, h]
        bbox = [150, 120, 200, 300]
        return dummy_frame, detected_class, confidence, bbox
    
    return dummy_frame, "None", 0.0, None

def run_audio_classification_inference(audio_samples):
    """
    PLACEHOLDER: Feed your raw audio array chunks directly into
    your TFLite YAMNet or custom CNN audio classifier model.
    """
    if np.random.random() > 0.99:
        detected_class = np.random.choice(THREAT_AUDIO_CLASSES)
        confidence = np.random.uniform(0.70, 0.92)
        return detected_class, confidence
    return "Background Noise", 0.0

# ==========================================
# 4. INITIALIZATION WARMUP
# ==========================================
print("=" * 70)
print("WARMING UP HARDWARE & INFERENCE PIPELINES...")
warmup_start = time.time()
while time.time() - warmup_start < 3.0:
    try:
        audio_data = audio_stream.read(CHUNK, exception_on_overflow=False)
        signal_samples = np.frombuffer(audio_data, dtype=np.int16)
        background_amplitude = (0.50 * float(np.max(np.abs(signal_samples)))) + (0.50 * background_amplitude)
    except Exception:
        continue
print(f"✅ Baseline Stable: {background_amplitude:.1f} | Master System Armed.")
print("=" * 70)

# ==========================================
# 5. INTEGRATED DECISION MATRIX LOOP
# ==========================================
try:
    while True:
        # Loop Variables
        current_time = time.strftime("%H:%M:%S")
        led_bulb.off()
        buzzer.off()

        # Module 1: PIR & Audio Spike
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

        # Module 2: Camera Inference
        cam_frame, cam_class, cam_conf, cam_bbox = run_camera_classification_inference()
        
        # Check if detection falls within your defined Area of Interest
        cam_triggered = False
        if cam_class in THREAT_CAMERA_CLASSES and cam_conf >= CONFIDENCE_THRESHOLD_CAM:
            if cam_bbox is not None:
                # Basic check: verify if the center point of bounding box falls in AOI bounds
                frame_h, frame_w, _ = cam_frame.shape
                box_cx = cam_bbox[0] + (cam_bbox[2] / 2)
                box_cy = cam_bbox[1] + (cam_bbox[3] / 2)
                
                if (AOI[1] * frame_w <= box_cx <= AOI[3] * frame_w) and (AOI[0] * frame_h <= box_cy <= AOI[2] * frame_h):
                    cam_triggered = True

        # Module 3: Audio Inference
        audio_class, audio_conf = run_audio_classification_inference(signal_samples)
        audio_class_triggered = (audio_class in THREAT_AUDIO_CLASSES) and (audio_conf >= CONFIDENCE_THRESHOLD_AUD)

        # ==========================================
        # ⚠️ MASTER LOGICAL OR DECISION CASCADE
        # ==========================================
        if cam_triggered or audio_class_triggered or spike_triggered:
            # 1. Fire Physical Alarm Hardware Actuators
            led_bulb.on()
            buzzer.on()
            
            # 2. Render Image Instance with Overlays
            if cam_bbox is not None:
                # Draw Area of Interest Boundary Box (Yellow)
                h, w, _ = cam_frame.shape
                cv2.rectangle(cam_frame, (int(AOI[1]*w), int(AOI[0]*h)), (int(AOI[3]*w), int(AOI[2]*h)), (0, 255, 255), 2)
                cv2.putText(cam_frame, "AOI Boundary", (int(AOI[1]*w)+5, int(AOI[0]*h)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Draw Target Classification Bounding Box (Red)
                cv2.rectangle(cam_frame, (cam_bbox[0], cam_bbox[1]), (cam_bbox[0]+cam_bbox[2], cam_bbox[1]+cam_bbox[3]), (0, 0, 255), 2)
                cv2.putText(cam_frame, f"{cam_class}: {cam_conf:.2f}", (cam_bbox[0], cam_bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 3. Output the required Telemetry Profile
            print("\n" + "="*80)
            print("🚨 SECURITY SYSTEM BREACHED - TELEMETRY RECORD DETECTED 🚨")
            print("="*80)
            print(f"Timestamp          : [{current_time}]")
            print(f"PIR Status         : {'ACTIVE (Zone Breached)' if pir_active else 'INACTIVE (Zone Empty)'}")
            print(f"Audio Jump         : {db_interval_change:.3f} dB (Limit Wall: {active_db_threshold:.2f} dB)")
            print(f"Camera Class       : {cam_class} (Confidence: {cam_conf:.2f})")
            print(f"Audio Class        : {audio_class} (Confidence: {audio_conf:.2f})")
            print(f"Trigger Source     : {'CAMERA ' if cam_triggered else ''}{'AUDIO_NET ' if audio_class_triggered else ''}{'SENSOR_FUSION' if spike_triggered else ''}")
            print("="*80)
            
            # Save the image instance context natively to disk for validation
            cv2.imwrite(f"instance_capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg", cam_frame)
            
            time.sleep(1.5)  # Let outputs lock continuous alarm state duration
        else:
            print(f"Scanning Space... [PIR: {int(pir_active)} | Mic Jump: {db_interval_change:.2f} dB | Cam: {cam_class} | Aud: {audio_class}]", end="\r")

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nShutting down master security loop system...")
finally:
    led_bulb.off()
    buzzer.off()
    audio_stream.stop_stream()
    audio_stream.close()
    audio_interface.terminate()
    print("🔄 Actuators isolated. System completely offline.")