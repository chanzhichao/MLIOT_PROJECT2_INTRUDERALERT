import RPi.GPIO as GPIO 
import time 
GPIO.setmode(GPIO.BCM) 
GPIO.setup(23, GPIO.OUT) 
GPIO.output(23, GPIO.HIGH) 
time.sleep(2) 
GPIO.output(23, GPIO.LOW) 
GPIO.cleanup() # this work for my 3 pin buzzer ky006. how to write the code with the gpio python libraries