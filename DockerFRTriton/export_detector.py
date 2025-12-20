import os
import shutil
from pathlib import Path

def main():
    # 1. Define where your unzipped files are and where they should go
    # Adjust 'buffalo_l_temp' if your unzipped folder name is different
    source_folder = Path("buffalo_l_temp") 
    target_path = Path("model_repository/face_detector/1/model.onnx")
    
    # det_10g is the RetinaFace model from the buffalo_l pack
    source_file = source_folder / "det_10g.onnx"

    print(f"[detector] Looking for source at: {source_file}")

    if not source_file.exists():
        print(f"[detector] ERROR: {source_file} not found!")
        print("Please ensure you unzipped buffalo_l.zip into a folder named 'buffalo_l_temp'")
        return

    # 2. Create the target directory structure
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. "Export" by copying the file into the repo
    print(f"[detector] Exporting to {target_path}...")
    shutil.copy(source_file, target_path)
    
    print("[detector] Success! Detector deployed to model_repository.")

if __name__ == "__main__":
    main()