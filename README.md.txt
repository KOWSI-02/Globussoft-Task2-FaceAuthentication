Task 2 Face Authentication App

This project contains:

1. A FastAPI application for face authentication

2. A training script to extract FaceNet embeddings

3. A testing script to verify face comparison

4. Sample images for verification

   How to Run

1. Install requirements:
   pip install -r requirements.txt

2. Train the model:
   python train_model.py

3. Test the model:
   python test_model.py

4. Start the API:
   uvicorn app:app --reload
