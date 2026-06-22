import cv2
import time

# Try index 0 (default Pi Cam slot or first USB camera)
CAMERA_INDEX = 0 

print("Initializing camera on index {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX)

# Allow the sensor a second to warm up/adjust exposure
time.sleep(1)

if not cap.isOpened():
    print("❌ Error: Could not open camera stream. Try changing CAMERA_INDEX to 1 or 2.")
else:
    print("✅ Camera interface opened successfully!")
    
    # Try to grab a single frame
    ret, frame = cap.read()
    
    if ret:
        print(f"✅ Success! Grabbed a frame with resolution: {frame.shape[1]}x{frame.shape[0]}")
        # Save it to disk so you can view it in VS Code
        cv2.imwrite("test_capture.jpg", frame)
        print("💾 Frame saved as 'test_capture.jpg' in your project directory.")
    else:
        print("❌ Error: Could not read frame from camera. Stream is open but empty.")

cap.release()