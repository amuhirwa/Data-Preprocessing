import sys
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
def run_system(face_image=None, voice_audio=None, customer_id=None):
    print("=== User Identity and Product Recommendation System ===")
    if not face_image:
        face_image = input("Enter path to face image (e.g., assets/images/member_neutral.jpg): ")
    recognition_result, matched_member = facial_recognition(face_image)
    if recognition_result:
        print(f"Face recognized. Making call to get product recommendation...")
    else:
        print("Access denied: Unrecognized face.")
        return
    
    print("-" * 50)
    if not customer_id:
        customer_id = input("Enter customer ID (e.g., A181): ")
    recommendation = product_recommendation(customer_id)

    print("-" * 50)
    
    if not voice_audio:
        voice_audio = input("Enter path to voice sample to proceed to get the product recommendation: ")
    if voice_verification(voice_audio).lower() == matched_member.lower():
        print("*" * 50)
        print(f"Voice verified. Welcome {matched_member}! Proceeding to display product recommendation...")
    else:
        print("*" * 50)
        print("Access denied: Unrecognized or incorrect voice.")
        return
    
    print(f"Product recommendation: {recommendation}")

def simulate_all():
    print("=== SIMULATING UNAUTHORIZED ATTEMPT UNKNOWN FACE ===")
    run_system(face_image=BASE_DIR / "assets/images/unauthorized_image.jpg", voice_audio=BASE_DIR / "assets/audios/michael_confirm.m4a", customer_id="A181")
    print("\n" + "="*50 + "\n")
    print("=== SIMULATING UNAUTHORIZED ATTEMPT MISMATCH FACE AND VOICE ===")
    run_system(face_image=BASE_DIR / "assets/images/michael_neutral.jpg", voice_audio=BASE_DIR / "assets/audios/amandine_approve.m4a", customer_id="A181")
    print("=== SIMULATING UNKNOWN CUSTOMER ID ===")
    try:
        run_system(face_image=BASE_DIR / "assets/images/afsa_neutral.jpg", voice_audio=BASE_DIR / "assets/audios/afsa_approve.m4a", customer_id="A1")
    except Exception as e:
        print(e)
    print("\n" + "="*50 + "\n")
    print("=== SIMULATING AUTHORIZED ATTEMPT ===")
    run_system(face_image=BASE_DIR / "assets/images/michael_neutral.jpg", voice_audio=BASE_DIR / "assets/audios/michael_confirm.m4a", customer_id="A181")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "simulate":
        simulate_all()
    else:
        run_system()
