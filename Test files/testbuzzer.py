import time
from gpiozero import PWMOutputDevice

BUZZER_PIN = 24
buzzer = PWMOutputDevice(BUZZER_PIN, frequency=4000)
print("Beeping passive buzzer at 1kHz...")
try:
    buzzer.value = 0.5
    time.sleep(2)
finally:
    buzzer.off()
    buzzer.close()