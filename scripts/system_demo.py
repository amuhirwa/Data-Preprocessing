import numpy as np
from PIL import Image
import librosa
import os
import pandas as pd
from multi_modal_model_trainer import ModelTrainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from predictor import ModelPredictor

# Set the project base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Initialize the predictor
predictor = ModelPredictor()

# Prediction functions
def facial_recognition(image_path):
    predicted_member, probabilities = predictor.predict_from_image(image_path)
    
    # Threshold for recognition (adjust as needed, e.g., 0.9)
    if max(probabilities) > 0.8:
        return True, predicted_member  # Return True and matched member
    return False, None

def voice_verification(audio_path):
    predicted_label = predictor.predict_from_audio(audio_path)
    return predicted_label

def product_recommendation(customer_id):
    prediction = predictor.predict_product(customer_id)
    return prediction

# System simulation
def simulate_system():
    print("=== User Identity and Product Recommendation System ===")
    face_image = input("Enter path to face image (e.g., assets/images/member_neutral.jpg): ")
    recognition_result, matched_member = facial_recognition(face_image)
    if recognition_result:
        print(f"Face recognized as {matched_member}. Proceeding to voice verification...")
    else:
        print("Access denied: Unrecognized face.")
        return
    print("-" * 50)
    
    voice_audio = input("Enter path to voice sample: ")
    if voice_verification(voice_audio) == matched_member:
        print("*" * 50)
        print("Voice verified. Proceeding to product recommendation...")
    else:
        print("*" * 50)
        print("Access denied: Unrecognized voice.")
        return
    
    print("-" * 50)
    customer_id = input("Enter customer ID (e.g., A178): ")
    recommendation = product_recommendation(customer_id)
    print(f"Product recommendation: {recommendation}")

if __name__ == "__main__":
    simulate_system()
