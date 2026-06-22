import os
from tflite_support import metadata

model_path = "/home/user/iot_project/Models/mobilenetv2_audio.tflite"
displayer = metadata.MetadataDisplayer.with_model_file(model_path)

# Create a temporary directory to dump the hidden files
export_dir = "./extracted_metadata"
os.makedirs(export_dir, exist_ok=True)

try:
    # This pulls out any embedded text/label files hidden inside the TFLite file
    file_names = displayer.get_packed_associated_file_list()
    print(f"📦 Found embedded files: {file_names}")
    
    for file_name in file_names:
        displayer.extract_associated_file(file_name, os.path.join(export_dir, file_name))
        print(f"✅ Extracted: {file_name} to {export_dir}")
        
except Exception as e:
    print("❌ No embedded label files found in metadata.")