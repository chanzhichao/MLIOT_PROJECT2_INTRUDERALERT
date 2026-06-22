import re

model_path = "/home/user/iot_project/Models/mobilenetv2_audio.tflite"

print("🔍 Scanning binary file for hidden label sets...")

try:
    with open(model_path, "rb") as f:
        data = f.read()

    # Look for groups of readable words separated by line breaks or null bytes
    # This mimics the unix 'strings' command
    found_words = re.findall(b"[a-zA-Z0-9_\-\s]{3,20}", data)
    
    # Clean up and convert to strings
    cleaned_strings = [w.decode('utf-8', errors='ignore').strip() for w in found_words]
    
    # Filter for common audio keyword signatures to pinpoint where the labels hide
    suspect_labels = []
    unique_set = list(dict.fromkeys(cleaned_strings)) # Remove duplicates
    
    for word in unique_set:
        if len(word) > 2:
            suspect_labels.append(word)

    print("\n📦 --- EXTRACTED TEXT FRAGMENTS ---")
    # Print the last 40 readable text fragments found in the model file 
    # (Metadata and labels are almost always appended to the very end of the file)
    for label in suspect_labels[-40:]:
        print(f"👉 {label}")
        
except Exception as e:
    print(f"❌ Failed to parse binary: {e}")