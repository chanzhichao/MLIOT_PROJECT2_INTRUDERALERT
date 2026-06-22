import time
import collections
import numpy as np
import pyaudio
import ai_edge_litert.interpreter as litert
import RPi.GPIO as GPIO

# ==========================================
# HARDWARE INITIALIZATION
# ==========================================
BUZZER_PIN = 23  
LED_PIN = 17

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(LED_PIN, GPIO.OUT)       
GPIO.output(BUZZER_PIN, GPIO.LOW) 
GPIO.output(LED_PIN, GPIO.LOW)

# ==========================================
# ROLLING AUDIO CONFIGURATION (1-SECOND RESERVOIR)
# ==========================================
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000  
CHUNK = 2048  # Fast hardware capture slice (~42ms)

# At 16kHz downsampled audio, 1 full second equals exactly 16,000 samples
ONE_SECOND_TOTAL_SAMPLES = 16000

# This rolling queue holds our exact 1-second waveform memory
audio_rolling_buffer = collections.deque(maxlen=ONE_SECOND_TOTAL_SAMPLES)

AUDIO_LABELS = ["class1", "class2", "class3", "class4", "class5", "class6", "class7"]

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK)

# ==========================================
# LOAD TFLITE MODEL
# ==========================================
model_path = "/home/user/iot_project/Models/mobilenetv2_audio.tflite"
audio_interpreter = litert.Interpreter(model_path=model_path)
audio_interpreter.allocate_tensors()

audio_input_details = audio_interpreter.get_input_details()
audio_output_details = audio_interpreter.get_output_details()
expected_audio_shape = audio_input_details[0]['shape']

# Statistical anomaly queues
abnormal_history = collections.deque(maxlen=50)

print("🔊 1-Second Audio Reservoir Isolation Engine Active.")
print("👉 Calibrating background acoustics...")
time.sleep(3)

try:
    while True:
        try:
            # Grab a quick micro-slice of sound from the mic
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            signal_samples = np.frombuffer(raw_data, dtype=np.int16)
        except IOError:
            continue  

        # Downsample 48kHz -> 16kHz via slicing
        downsampled_samples = signal_samples[::3].astype(np.float32) / 32768.0

        # 🚀 THE MAGIC: Append the fresh audio samples into our 1-second rolling queue
        audio_rolling_buffer.extend(downsampled_samples)

        # Wait until the rolling reservoir has collected a full 1-second baseline of sound
        if len(audio_rolling_buffer) < ONE_SECOND_TOTAL_SAMPLES:
            continue

        # Convert the rolling queue into a solid array for the AI model
        full_second_snapshot = np.array(audio_rolling_buffer)

        # Fit the 1-second snapshot smoothly into the expected TFLite input size
        audio_tensor_input = np.zeros(expected_audio_shape, dtype=np.float32)
        filled_length = min(len(full_second_snapshot), audio_tensor_input.size)
        audio_tensor_input.flat[:filled_length] = full_second_snapshot[:filled_length]

        # Run inference on the complete historical second
        audio_interpreter.set_tensor(audio_input_details[0]['index'], audio_tensor_input)
        audio_interpreter.invoke()
        raw_logits = audio_interpreter.get_tensor(audio_output_details[0]['index'])[0]

        exp_logits = np.exp(raw_logits - np.max(raw_logits))
        audio_probabilities = exp_logits / np.sum(exp_logits)

        # Statistical Math Filters
        normal_pool = audio_probabilities[0] + audio_probabilities[5] + audio_probabilities[6]
        abnormal_pool = 1.0 - normal_pool
        
        # Gated memory check to protect history from pollution
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

            dist_string = " | ".join([f"{AUDIO_LABELS[i][:5]}: {audio_probabilities[i]*100:3.0f}%" for i in range(len(AUDIO_LABELS))])
            print(f"Z: {z_score:5.1f}σ | AB_POOL: {abnormal_pool*100:3.0f}% || {dist_string}", end='\r')

            # ==========================================
            # 🚨 TRIGGER CRITERIA
            # ==========================================
            Z_THRESHOLD = 2.0
            if z_score > Z_THRESHOLD and abnormal_pool > 0.01:
                sorted_indices = list(np.argsort(audio_probabilities))
                if 5 in sorted_indices:
                    sorted_indices.remove(5)
                
                top1_idx = sorted_indices[-1]
                top1_name = AUDIO_LABELS[top1_idx]
                top1_prob = audio_probabilities[top1_idx] * 100

                top2_idx = sorted_indices[-2]
                top2_name = AUDIO_LABELS[top2_idx]
                top2_prob = audio_probabilities[top2_idx] * 100

                # Trigger Pin Output
                GPIO.output(BUZZER_PIN, GPIO.HIGH) 
                GPIO.output(LED_PIN, GPIO.HIGH)        
                print(f"\n⚡ [FLAGGED] 1-SECOND SURGE BREAKOUT: {z_score:.2f}σ")
                print(f"🔥 [ALARM] SOUND PROFILE RECOGNIZED!")
                print(f"   🏆 1st Highest : {top1_name.upper()} ({top1_prob:.1f}%)")
                print(f"   🥈 2nd Highest : {top2_name.upper()} ({top2_prob:.1f}%)")
                print("")

                time.sleep(0.5) 

                GPIO.output(BUZZER_PIN, GPIO.LOW)
                GPIO.output(LED_PIN, GPIO.LOW)         
                print(f"💤 [HARDWARE] PINS RESET TO LOW\n")
                
                # Clear buffer on alert so the same crash sound doesn't re-trigger
                audio_rolling_buffer.clear()
                time.sleep(1.0)

        # 🚀 MANDATORY OS COOLING GOVERNOR
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopping audio test engine cleanly.")
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.cleanup()
    stream.stop_stream()
    stream.close()
    p.terminate()