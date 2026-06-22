import cv2
import numpy as np
import onnxruntime as ort
import ai_edge_litert.interpreter as litert
import time

# ==========================================
# 1. HARDWARE SETTINGS (PLACEHOLDERS)
# ==========================================
CAMERA_INDEX = 0      # Change this to 1 or 4 if your V4L2 test showed alternative ports
PIR_SENSOR_PIN = 18   

# ==========================================
# 2. INITIALIZE AI MODELS FROM STORAGE
# ==========================================
camera_session = ort.InferenceSession("Models/yolo11n.onnx")
camera_input_name = camera_session.get_inputs()[0].name

audio_interpreter = litert.Interpreter(model_path="Models/mobilenetv2_audio.tflite")
audio_interpreter.allocate_tensors()
audio_input_details = audio_interpreter.get_input_details()
audio_output_details = audio_interpreter.get_output_details()

video_capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

# ==========================================
# 3. HELPER PREPROCESSING FUNCTIONS
# ==========================================
def preprocess_frame(frame):
    # YOLO11 explicitly expects 640x640
    resized = cv2.resize(frame, (640, 640))
    # Convert BGR (OpenCV default) to RGB (YOLO expectation)
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = np.float32(rgb_img) / 255.0
    blob = np.transpose(blob, (2, 0, 1)) # [Channels, Height, Width]
    blob = np.expand_dims(blob, axis=0)  # [1, Channels, Height, Width]
    return blob

def read_pir_sensor(pin):
    return True 

# ==========================================
# 4. SENSOR FUSION CONFIGURATION
# ==========================================
BASE_THRESHOLD = 0.75     
SENSITIVITY_BOOST = 0.20  

W_VISION = 0.6            
W_AUDIO = 0.3             

# ==========================================
# 5. EXECUTION LOOP
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
        
        # YOLO11 raw output is typically shape [1, 84, 8400] 
        # (4 bounding box values + 80 COCO object classes)
        raw_predictions = vision_output[0][0] 
        # Extract rows representing object confidence classes, ignoring the first 4 bounding box items
        class_confidences = raw_predictions[4:, :]
        vision_score = float(np.max(class_confidences)) # Pull the highest detected probability

        # --- Inference 2: Audio (MobileNetV2 LiteRT) ---
        # Dynamically match whatever input shape your audio model's tensor expects
        expected_audio_shape = audio_input_details[0]['shape']
        audio_data = np.random.rand(*expected_audio_shape).astype(np.float32) # Using random spectrum signature for simulation
        
        audio_interpreter.set_tensor(audio_input_details[0]['index'], audio_data)
        audio_interpreter.invoke()
        audio_output = audio_interpreter.get_tensor(audio_output_details[0]['index'])
        audio_score = float(np.max(audio_output)) # Extract maximum classification prediction probability

        # --- Sensor Fusion Decision Math ---
        fused_ai_score = (W_VISION * vision_score) + (W_AUDIO * audio_score)
        
        if pir_active:
            current_threshold = BASE_THRESHOLD - SENSITIVITY_BOOST
        else:
            current_threshold = BASE_THRESHOLD

        print(f"AI Fusion Score: {fused_ai_score:.2f} | Active Threshold: {current_threshold:.2f} (PIR: {pir_active})")

        if fused_ai_score > current_threshold:
            print("🚨 INTRUDER CONFIRMED! 🚨")
            time.sleep(3) 

        time.sleep(0.1)

finally:
    video_capture.release()