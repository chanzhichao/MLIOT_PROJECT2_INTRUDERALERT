import numpy as np
import time
import ai_edge_litert.interpreter as litert

# ==========================================
# 1. INITIALIZE LITER_T INTERPRETER
# ==========================================
print("Loading MobileNetV2 Audio TFLite Model...")
audio_interpreter = litert.Interpreter(model_path="Models/mobilenetv2_audio.tflite")
audio_interpreter.allocate_tensors()

# Get structural tensor details
audio_input_details = audio_interpreter.get_input_details()
audio_output_details = audio_interpreter.get_output_details()

expected_audio_shape = audio_input_details[0]['shape']
print(f"📊 Model Input Expected Shape : {expected_audio_shape}")
print(f"📊 Model Input Data Type      : {audio_input_details[0]['dtype']}")

# ==========================================
# 2. RUNNING STREAM TEST LOOP
# ==========================================
try:
    print("\nAudio model execution online. Press Ctrl+C to stop.")
    while True:
        # Generate array data matching your model's exact input shape
        audio_data = np.random.rand(*expected_audio_shape).astype(np.float32) 
        
        # Pass input array to the tensor register
        audio_interpreter.set_tensor(audio_input_details[0]['index'], audio_data)
        audio_interpreter.invoke()
        
        # Extract prediction probabilities array
        audio_output = audio_interpreter.get_tensor(audio_output_details[0]['index'])[0]
        
        # Find the index with the highest confidence value
        best_class_index = int(np.argmax(audio_output))
        audio_score = float(audio_output[best_class_index]) 

        # Print out the raw performance metrics
        print(f"Predicted Class Index: {best_class_index} | Confidence: {audio_score:.4f}")
        # Print this out right after allocating tensors to inspect output node metadata
        #print("--- OUTPUT DETAILS MATRIX ---")
        #print(audio_output_details[0])

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n🔄 Audio model testing paused.")