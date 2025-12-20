import numpy as np
import cv2
import tritonclient.http as httpclient
from typing import Any
from pathlib import Path

# --- Constants ---
TRITON_HTTP_PORT = 8000
MODEL_NAME = "fr_model"
MODEL_INPUT_NAME = "input.1"
MODEL_OUTPUT_NAME = "683"
DET_MODEL_NAME = "face_detector"
DET_MODEL_INPUT_NAME = "input.1"
DET_MODEL_OUTPUT_NAMES = ["448", "451", "454"]

class TritonService:
    def __init__(self, client=None, url=f"localhost:{TRITON_HTTP_PORT}"):
        if client is not None:
            self.client = client
        else:
            self.client = httpclient.InferenceServerClient(url=url)

    def _infer(self, model_name, inputs, output_names):
        outputs = [httpclient.InferRequestedOutput(name) for name in output_names]
        response = self.client.infer(model_name, inputs, outputs=outputs)
        
        results = []
        for name in output_names:
            arr = response.as_numpy(name)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            results.append(arr)
        return results

    def get_face_detection(self, image_640):
        inputs = [httpclient.InferInput(DET_MODEL_INPUT_NAME, image_640.shape, "FP32")]
        inputs[0].set_data_from_numpy(image_640.astype(np.float32))
        return self._infer(DET_MODEL_NAME, inputs, DET_MODEL_OUTPUT_NAMES)
    
    def get_embedding(self, face_112):
        # Ensure face_112 is (1, 3, 112, 112)
        # If it's coming in as (112, 112, 3), we MUST transpose it
        if face_112.ndim == 3:
            face_112 = face_112.transpose(2, 0, 1)
            face_112 = np.expand_dims(face_112, axis=0)
        elif face_112.ndim == 4 and face_112.shape[-1] == 3:
            # If it's (1, 112, 112, 3) -> (1, 3, 112, 112)
            face_112 = face_112.transpose(0, 3, 1, 2)

        inputs = [httpclient.InferInput(MODEL_INPUT_NAME, face_112.shape, "FP32")]
        
        face_data = face_112.astype(np.float32)
        if face_data.max() <= 1.1: # Use 1.1 to be safe with float rounding
            face_data = face_data * 255.0

        face_preprocessed = (face_data - 127.5) / 128.0
        
        inputs[0].set_data_from_numpy(face_preprocessed)
        results = self._infer(MODEL_NAME, inputs, [MODEL_OUTPUT_NAME])
        return results[0].flatten() # Flatten to ensure a 1D vector for np.dot

# --- Compatibility stubs for app.py ---
def create_triton_client(url):
    return httpclient.InferenceServerClient(url=url)

def prepare_model_repository(repo_path: Path): pass
def start_triton_server(repo_path: Path): return None 
def stop_triton_server(handle): pass