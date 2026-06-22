import time
from gpiozero import LED

# 🔴 Connected to the orange wire on your board
LED_PIN = 17  

print("Initializing LED...")
led = LED(LED_PIN)

print("Starting LED test loop (Blinking 5 times)...")

for i in range(5):
    print(f"[{i+1}] LED ON")
    led.on()
    time.sleep(1)
    
    print(f"[{i+1}] LED OFF")
    led.off()
    time.sleep(1)

print("LED test complete!")