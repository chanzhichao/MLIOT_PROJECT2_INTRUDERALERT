import time
import numpy as np
import pyaudio
import paho.mqtt.client as mqtt

# 📌 Configuration
# 📌 Updated Configuration
MQTT_BROKER = "localhost"
MQTT_AUDIO_TOPIC = "home/sensor/audio"
DEVICE_INDEX = 1        
AUDIO_THRESHOLD = 5000  
RATE = 48000 # 🚀 Changed from 16000 to standard 44100 Hz
CHUNK = 1024

print("Initializing MQTT Client...")
client = mqtt.Client()
client.connect(MQTT_BROKER, 1883, 60)
client.loop_start()

# Setup PyAudio
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=CHUNK
)

print("\n--- Continuous Audio Monitor Active ---")
print("Listening for loud sounds / mic inputs... Running for 20 seconds.")

try:
    end_time = time.time() + 20
    while time.time() < end_time:
        # Read raw binary audio data from the microphone stream
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        # Convert binary data to integers to calculate volume level
        audio_data = np.frombuffer(data, dtype=np.int16)
        peak_volume = np.max(np.abs(audio_data))
        
        # If the volume crosses our threshold, broadcast it!
        if peak_volume > AUDIO_THRESHOLD:
            print(f"🔊 Sound Detected! Volume: {peak_volume}")
            client.publish(MQTT_AUDIO_TOPIC, f"LOUD_SOUND:{peak_volume}")
            
        time.sleep(0.05)

finally:
    # Clean up audio stream resources cleanly
    print("\nClosing audio pipeline...")
    stream.stop_stream()
    stream.close()
    p.terminate()
    client.loop_stop()
    client.disconnect()
    print("Audio Monitor Stopped.")