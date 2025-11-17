from fastapi import FastAPI, UploadFile, File
from test_model import extract_embedding
import numpy as np

app = FastAPI()

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.get("/")
def home():
    return {"message": "Face Authentication API is running!"}

@app.post("/verify/")
async def verify_faces(img1: UploadFile = File(...), img2: UploadFile = File(...)):
    # Save uploaded images temporarily
    file1_path = "temp_img1.jpg"
    file2_path = "temp_img2.jpg"

    with open(file1_path, "wb") as f:
        f.write(await img1.read())

    with open(file2_path, "wb") as f:
        f.write(await img2.read())

    # Extract embeddings
    emb1 = extract_embedding(file1_path)
    emb2 = extract_embedding(file2_path)

    # Calculate similarity
    similarity_score = cosine_similarity(emb1, emb2)

    # Decide result
    threshold = 0.65  # Higher score = more similar
    result = "Same Person" if similarity_score > threshold else "Different Person"

    return {
        "similarity_score": float(similarity_score),
        "verification_result": result
    }
