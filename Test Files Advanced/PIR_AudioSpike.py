import numpy as np
import time
import pyaudio
from gpiozero import DigitalInputDevice  # Hardware library for the physical PIR

# ==========================================
# 1. HARDWARE & AUDIO STREAM CONFIGURATION
# ==========================================
PIR_SENSOR_PIN = 18   
pir_hardware = DigitalInputDevice(PIR_SENSOR_PIN)

# PyAudio Buffer Parameters
FORMAT = pyaudio.paInt16     # Pull raw 16-bit signed integers (0 to 32767)
CHANNELS = 1                 # Mono recording input
RATE = 48000                 # 16kHz sampling rate is ideal for lightweight DSP
CHUNK = 1024                 # Read 1024 audio frames per loop iteration (~64ms window)

# Initialize Audio Interface
audio_interface = pyaudio.PyAudio()
audio_stream = audio_interface.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

# PIR Fast Decay tracking variables
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
# 2. REAL HARDWARE CALIBRATION KNOBS
# ==========================================
# Re-tuned base target floor to match your silent room logs
background_amplitude = 100.0  
ALPHA = 0.15  # Learning rate to absorb consistent fan drone

# 🎯 CALIBRATED PHYSICAL DECIBEL HUDRLE LIMITS
NO_MOTION_THRESHOLD_DB = 12.0  # Empty room: Fan turbulence is safely ignored under this wall
MOTION_THRESHOLD_DB = 6.00     # PIR active: Clear of the 5.8 dB hardware pop, ready for talking

last_floor_update_time = "N/A"

# ==========================================
# 3. LIVE PRODUCTION EXECUTION LOOP
# ==========================================
print("=" * 70)
print("LIVE BALANCED HARDWARE PIPELINE ARMED. SCANNING ZONE...")
print("Press Ctrl+C to stop the stream safely.")
print("=" * 70)

try:
    while True:
        # Step 1: Read Live Physical PIR Context
        pir_active = read_pir_sensor()
        
        # Step 2: Read Live Unclipped Mic Amplitude from Hardware Buffer
        try:
            audio_data = audio_stream.read(CHUNK, exception_on_overflow=False)
            signal_samples = np.frombuffer(audio_data, dtype=np.int16)
            raw_mic_amplitude = float(np.max(np.abs(signal_samples)))
        except Exception as audio_err:
            print(f"\n[Warning] Audio buffer sync drop: {audio_err}")
            continue

        # --- Step 3: Direct Logarithmic Decibel Math with Minimum Floor ---
        safe_mic = max(1.0, raw_mic_amplitude)
        
        # 🚀 THE CRITICAL PROTECTION ANCHOR
        # Clamping safe_background at 400.0 stops the decibel math from exploding 
        # when your silent room drops to ~20 units, swallowing the PIR activation click.
        safe_background = max(400.0, background_amplitude)
        
        # Calculate the exact real-time decibel interval jump
        db_interval_change = 20.0 * np.log10(safe_mic / safe_background)

        # --- Step 4: Contextual Threshold Evaluation ---
        active_db_threshold = MOTION_THRESHOLD_DB if pir_active else NO_MOTION_THRESHOLD_DB

        # Check for breach
        alarm_triggered = db_interval_change > active_db_threshold

        # --- Step 5: Fast-Tracking Baseline Adaptation ---
        # Update the background variable normally, while our anchor above keeps math stable
        if not alarm_triggered:
            background_amplitude = (ALPHA * safe_mic) + ((1.0 - ALPHA) * background_amplitude)
            last_floor_update_time = time.strftime("%H:%M:%S")

        # --- Step 6: Console Telemetry Feed ---
        if alarm_triggered:
            print("-" * 70)
            print(f"🚨 ALERT TRIGGERED! 🚨")
            print(f"PIR Sensor Zone     : {'BREACHED (Low Threshold Context)' if pir_active else 'SECURE (High Threshold Context)'}")
            print(f"Live Audio Amplitude: {raw_mic_amplitude:.1f} units")
            print(f"Calculated Baseline : {background_amplitude:.1f} units (Clamped Math Base: {safe_background:.1f})")
            print(f"Measured Jump (dB)  : {db_interval_change:.3f} dB")
            print(f"Active Limit Needed : {active_db_threshold:.3f} dB")
            print("-" * 70)
            
            time.sleep(1.0)  # Prevent print log flooding during sustained alerts
        else:
            print(f"Scanning... [PIR: {int(pir_active)} | Live dB Jump: {db_interval_change:.3f} dB | Floor: {background_amplitude:.1f} | Last Update: {last_floor_update_time}]", end="\r")

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n\nStopping hardware pipelines gracefully...")
finally:
    audio_stream.stop_stream()
    audio_stream.close()
    audio_interface.terminate()
    print("🔄 Audio streams closed. PIR system offline.")