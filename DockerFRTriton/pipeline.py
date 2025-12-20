import numpy as np
import cv2
from triton_service import TritonService

class FacePipeline:
    def __init__(self, client=None):
        self.triton = TritonService(client=client)
        self.target_size = 640
        self.fr_size = 112

    def _preprocess_detector(self, img):
        img_resized = cv2.resize(img, (self.target_size, self.target_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_input = img_rgb.transpose(2, 0, 1)[np.newaxis, :]
        return img_input.astype(np.float32)

    def _postprocess_detector(self, raw_output):
        scores, boxes, landms = raw_output
        scores_1d = scores[:, 1] if (scores.ndim > 1 and scores.shape[1] > 1) else scores.flatten()
        
        idx = np.where(scores_1d > 0.5)[0]
        if len(idx) == 0: return None, None

        best_idx = idx[np.argmax(scores_1d[idx])]
        kps = landms[best_idx].reshape(5, 2)
        return boxes[best_idx], kps

    def _align_face(self, img, kps):
        """
        The missing method! 
        Standard InsightFace reference points for a 112x112 output crop.
        """
        dst_points = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]
        ], dtype=np.float32)
        
        M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), dst_points)
        return cv2.warpAffine(img, M, (self.fr_size, self.fr_size))

    def get_single_embedding(self, img):
        # 1. Detection
        # _preprocess_detector already resizes to 640x640
        input_640 = self._preprocess_detector(img)
        raw_det = self.triton.get_face_detection(input_640)
        _, kps = self._postprocess_detector(raw_det)
        
        if kps is None: 
            raise ValueError("No face detected")

        # 2. Alignment - THE FIX IS HERE
        # You MUST align using the SAME image size the landmarks were found on.
        # We resize the original 'img' to 640 to match the 'kps' coordinates.
        img_for_alignment = cv2.resize(img, (640, 640))
        face_aligned = self._align_face(img_for_alignment, kps)
        
        # 3. DEBUG: Check if the crop is actually a face
        # If you can, check this file. If it's not a clear face, similarity will stay 0.99
        cv2.imwrite("check_this_crop.jpg", face_aligned) 

        # 4. Color Conversion (BGR -> RGB)
        face_rgb = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB)
        
        # 5. Prepare for Triton (CHW format)
        face_input = face_rgb.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
        
        # 6. Inference & Normalization
        emb = self.triton.get_embedding(face_input)
        norm = np.linalg.norm(emb)
        
        if norm < 1e-6:
            return np.zeros(512)
        return (emb / norm).flatten()

# --- Functions for app.py ---
def run_inference(client, image_bytes: bytes):
    pipeline = FacePipeline(client)
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return pipeline.get_single_embedding(img)

def calculate_face_similarity(client, image_a_bytes: bytes, image_b_bytes: bytes):
    pipeline = FacePipeline(client)
    
    emb1 = pipeline.get_single_embedding(cv2.imdecode(np.frombuffer(image_a_bytes, np.uint8), cv2.IMREAD_COLOR))
    emb2 = pipeline.get_single_embedding(cv2.imdecode(np.frombuffer(image_b_bytes, np.uint8), cv2.IMREAD_COLOR))
    
    # Cosine Similarity = (A dot B) / (||A|| * ||B||)
    # Since we already normalize in get_single_embedding, np.dot is enough,
    # but let's be 100% safe:
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    dot_product = np.dot(emb1, emb2)
    similarity = dot_product / (norm1 * norm2)
    
    return float(similarity)