"""
Utility stubs for the face recognition project.

Each function is intentionally left unimplemented so that students can
fill in the logic as part of the coursework.
"""

from typing import Any, List
import cv2
import numpy as np
from retinaface import RetinaFace
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# Initialize InsightFace globally (once)
embedder = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
embedder.prepare(ctx_id=0, det_size=(640, 640))
_ = embedder.models['recognition']  # force load recognition model


def detect_faces(image: Any) -> List[Any]:
    """
    Detect faces within the provided image.

    Parameters can be raw image bytes or a decoded image object, depending on
    the student implementation. Expected to return a list of face regions
    (e.g., bounding boxes or cropped images).
    """
    if not isinstance(image, np.ndarray):
        image = decode_image(image)
    # Call RetinaFace
    faces = RetinaFace.detect_faces(image)
    if faces is None or not isinstance(faces, dict):
        return []
    # Extract all faces
    results = []
    for key, face in faces.items():
        x1, y1, x2, y2 = face["facial_area"]
        face_img = image[y1:y2, x1:x2]
        face = [face_img, face["landmarks"]]
        results.append(face)
    # Return list of cropped face images
    return results


def compute_face_embedding(face_image: np.ndarray) -> np.ndarray:
    """
    Compute a numerical embedding vector for the provided face image.
    Uses the recognition model from InsightFace directly on an aligned 112x112 crop.
    """
    if not isinstance(face_image, np.ndarray):
        raise ValueError("face_image must be a numpy array")

    # Convert to RGB (InsightFace models expect RGB)
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (112, 112))

    # Get the recognition model (ArcFace)
    model = embedder.models.get('recognition', None)
    if model is None:
        raise RuntimeError("Recognition model not loaded in embedder")

    # ArcFaceONNX expects a batch of faces (N, 112, 112, 3) in RGB
    embedding = model.get_feat(face_resized)

    if embedding is None or embedding.shape[0] == 0:
        raise ValueError("Failed to compute embedding")

    # Normalize the embedding vector
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

def detect_face_keypoints(face_image: Any) -> Any:
    """
    Identify facial keypoints (landmarks) for alignment or analysis.

    The return type can be tailored to the chosen keypoint detection library.
    """
    return face_image[1]


def warp_face(image: Any, homography_matrix: Any) -> Any:
    """
    Warp the provided face image using the supplied homography matrix.

    Typically used to align faces prior to embedding extraction.
    """
    if homography_matrix is None:
        raise ValueError("Facial keypoints required for warping")
    # Convert dict of facial keypoints to numpy array (source points)
    src_points = np.array([
        homography_matrix["right_eye"],
        homography_matrix["left_eye"],
        homography_matrix["nose"],
        homography_matrix["mouth_left"],
        homography_matrix["mouth_right"]
    ], dtype=np.float32)
    # Standard 5-point reference for a 112x112 aligned face (InsightFace standard)
    dst_points = np.array([
        [38.2946, 51.6963],   # right_eye
        [73.5318, 51.5014],   # left_eye
        [56.0252, 71.7366],   # nose
        [41.5493, 92.3655],   # mouth_left
        [70.7299, 92.2041]    # mouth_right
    ], dtype=np.float32)
    # Compute affine transform
    M, _ = cv2.estimateAffinePartial2D(src_points, dst_points)
    if M is None:
        raise ValueError("Failed to compute affine transform for face alignment")
    # Warp face to 112x112
    aligned_face = cv2.warpAffine(image, M, (112, 112))
    return aligned_face

def antispoof_check(face_image: Any) -> float:
    """
    Perform an anti-spoofing check and return a confidence score.

    A higher score should indicate a higher likelihood that the face is real.
    """
    raise NotImplementedError("Student implementation required for face anti-spoofing")


def calculate_face_similarity(image_a: Any, image_b: Any) -> float:
    """
    End-to-end pipeline that returns a similarity score between two faces.

    This function should:
      1. Detect faces in both images.
      2. Align faces using keypoints and homography warping.
      3. (Run anti-spoofing checks to validate face authenticity. - If you want)
      4. Generate embeddings and compute a similarity score.

    The images provided by the API arrive as raw byte strings; convert or decode
    them as needed for downstream processing.
    """
    imga = decode_image(image_a)
    imgb = decode_image(image_b)
    # Start by detecting the faces in both images
    faces_a = detect_faces(imga)
    faces_b = detect_faces(imgb)

    if(len(faces_a) == 0 or len(faces_b) == 0):
        raise ValueError("No faces detected in one or both images")
    if(len(faces_a) > 1 or len(faces_b) > 1):
        raise ValueError("Multiple faces detected in one or both images")
    
    kpsa = detect_face_keypoints(faces_a[0])
    if kpsa is None:
        raise ValueError("Could not detect keypoints for face alignment")
    face_a = warp_face(imga, kpsa) 

    kpsb = detect_face_keypoints(faces_b[0])
    if kpsb is None:
        raise ValueError("Could not detect keypoints for face alignment")
    face_b = warp_face(imgb, kpsb)  

    emb_a = compute_face_embedding(face_a)
    emb_b = compute_face_embedding(face_b)

    sim = float(np.dot(emb_a.flatten(), emb_b.flatten()))
    return sim  # Placeholder similarity score


def decode_image(image_bytes: bytes) -> np.ndarray:
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img
    