from deepface import DeepFace
import numpy as np
import cv2

# Load FaceNet model (same as train_model.py)
model = DeepFace.build_model("Facenet")

def extract_embedding(image_path):
    # Read image
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Image not found or cannot be opened: {image_path}")

    # Generate embedding (DeepFace handles detection internally)
    embedding_data = DeepFace.represent(
        img_path=image_path,
        model_name="Facenet",
        enforce_detection=True
    )

    embedding = np.array(embedding_data[0]["embedding"])

    return embedding
