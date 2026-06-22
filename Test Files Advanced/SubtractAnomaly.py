import time
import numpy as np
import pyaudio
import ai_edge_litert.interpreter as litert

# ==========================================
# 1. HARDWARE & AUDIO CONFIGURATION
# ==========================================
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000  
CHUNK = 1024  

AUDIO_LABELS = ["class1", "class2", "class3", "class4", "class5", "class6", "class7"]

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK)

# ==========================================
# 2. LOAD TFLITE MODEL
# ==========================================
model_path = "/home/user/iot_project/Models/mobilenetv2_audio.tflite"
audio_interpreter = litert.Interpreter(model_path=model_path)
audio_interpreter.allocate_tensors()

audio_input_details = audio_interpreter.get_input_details()
audio_output_details = audio_interpreter.get_output_details()
expected_audio_shape = audio_input_details[0]['shape']

# ==========================================
# 3. PLAN 1: GLOBAL BASELINE INITIALIZATION
# ==========================================
# This tracks the baseline "noise floor share" of your room
abnormal_baseline = 0.01  
ALPHA = 0.1  # How fast the baseline adapts to a changing room (EMA)

print("🔊 Plan 1: Unified Delta Test Engine Started.")
print("👉 Calibrating room noise floor...")
time.sleep(2)

try:
    while True:
        try:
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            signal_samples = np.frombuffer(raw_data, dtype=np.int16)
        except IOError:
            continue

        downsampled_samples = signal_samples[::3].astype(np.float32) / 32768.0

        audio_tensor_input = np.zeros(expected_audio_shape, dtype=np.float32)
        filled_length = min(len(downsampled_samples), audio_tensor_input.size)
        audio_tensor_input.flat[:filled_length] = downsampled_samples[:filled_length]

        audio_interpreter.set_tensor(audio_input_details[0]['index'], audio_tensor_input)
        audio_interpreter.invoke()
        raw_logits = audio_interpreter.get_tensor(audio_output_details[0]['index'])[0]

        exp_logits = np.exp(raw_logits - np.max(raw_logits))
        audio_probabilities = exp_logits / np.sum(exp_logits)

        # ==========================================
        # PLAN 1: THE UNIFIED BALANCING MATH
        # ==========================================
        # Group baseline background carrier waves (Class 0, 5, 6)
        normal_pool = audio_probabilities[0] + audio_probabilities[5] + audio_probabilities[6]
        abnormal_pool = 1.0 - normal_pool

        # Calculate the absolute mathematical delta shift
        delta_anomaly = abnormal_pool - abnormal_baseline

        # Smoothly update the baseline floor when things are relatively quiet
        if delta_anomaly < 0.10:
            abnormal_baseline = (ALPHA * abnormal_pool) + ((1.0 - ALPHA) * abnormal_baseline)

        # Construct live diagnostics string
        dist_string = " | ".join([f"{AUDIO_LABELS[i][:5]}: {audio_probabilities[i]*100:3.0f}%" for i in range(len(AUDIO_LABELS))])
        
        # 🛠️ LIVE DIAGNOSTICS DISPLAY
        print(f"Δ SHIFT: {delta_anomaly*100:+5.1f}% | BASE_FLOOR: {abnormal_baseline*100:3.0f}% || {dist_string}", end='\r')

        # ==========================================
        # 🚨 PLAN 1: DELTA TRIGGER GATE
        # ==========================================
        # DELTA_THRESHOLD = 0.06 means if the threat classes steal 6% or more of the 
        # total probability pool away from Footsteps/Ambient, drop inside!
        DELTA_THRESHOLD = 0.03 
        
        if delta_anomaly > DELTA_THRESHOLD:
            sorted_indices = np.argsort(audio_probabilities)
            triggered_class_idx = int(sorted_indices[-1])
            
            if triggered_class_idx == 5:
                triggered_class_idx = int(sorted_indices[-2])

            triggered_class_name = AUDIO_LABELS[triggered_class_idx]

            # 1. 🟢 HARDWARE HIGH SIGNAL
            print(f"\n⚡ [HARDWARE] PIN SET TO HIGH (ON)")

            print(f"🔥 [ALARM] UNIFIED DELTA TRIGGERED!")
            print(f"   Detected Class   : {triggered_class_name.upper()}")
            print(f"   Delta Jump       : {delta_anomaly*100:+.1f}% share surge")
            print("   Full Snapshot Distribution:")
            for i, label in enumerate(AUDIO_LABELS):
                print(f"      - {label:<12}: {audio_probabilities[i]*100:5.1f}%")
            print("")

            # 2. ⏳ HOLD THE FLASH
            time.sleep(0.3)

            # 3. 🔴 HARDWARE LOW SIGNAL
            print(f"💤 [HARDWARE] PIN SET TO LOW (OFF)\n")
            time.sleep(0.7)

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping Plan 1 engine cleanly.")
    stream.stop_stream()
    stream.close()
    p.terminate()