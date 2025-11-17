from deepface import DeepFace
import pickle

def load_model_and_save():
    # Load the pre-trained FaceNet model
    model = DeepFace.build_model("Facenet")
    
    # Save the loaded model into a file
    with open("face_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("Model saved.")

if __name__ == "__main__":
    load_model_and_save()
