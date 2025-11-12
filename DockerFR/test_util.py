import cv2
import numpy as np
from util import detect_faces, calculate_face_similarity, warp_face, compute_face_embedding

def test_face_detection():
    # Load two test images
    img1 = cv2.imread("testimg1.jpg")  # Using the test image from RetinaFace
    if img1 is None:
        print("Error: Could not load test image 1")
        return

    # Test face detection
    faces = detect_faces(img1)
    print(f"Found {len(faces)} faces in test image")
    
    #Draw detected faces on image for visualization
    cv2.imwrite("detected_faces.jpg", faces[0][0])

def test_face_similarity():
    # Test with same image (should give similarity close to 1.0)
    with open("./testimages/testimg2.jpg", "rb") as f:
        img_bytes = f.read()
    with open("./testimages/testimgWonyoung2.jpg", "rb") as f:
        img_bytes2 = f.read()
    
    result = calculate_face_similarity(img_bytes, img_bytes2)
    print("Similarity test result:", result)

def test_single_embedding(img_path: str):
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: could not read {img_path}")
        return

    # Detect face
    faces = detect_faces(img)
    if len(faces) == 0:
        print("No face detected!")
        return
    
    face_img, landmarks = faces[0]

    # Align the face
    aligned = warp_face(face_img, landmarks)

    # Compute embedding
    embedding = compute_face_embedding(aligned)

    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    print(f"First 5 values: {embedding[:5]}")

def compare_two_faces(img1_path: str, img2_path: str):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Detect & align both
    face1 = detect_faces(img1)[0]
    face2 = detect_faces(img2)[0]
    aligned1 = warp_face(face1[0], face1[1])
    aligned2 = warp_face(face2[0], face2[1])

    # Compute embeddings
    emb1 = compute_face_embedding(aligned1)
    emb2 = compute_face_embedding(aligned2)

    # Cosine similarity
    sim = float(np.dot(emb1.flatten(), emb2.flatten()))
    print(f"Cosine similarity: {sim:.4f}")

if __name__ == "__main__":
    # print("Testing face detection...")
    # test_face_detection()

    test_single_embedding("testimg1.jpg")

    print("\nTesting face similarity...")
    test_face_similarity()