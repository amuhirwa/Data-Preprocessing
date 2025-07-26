import numpy as np
from PIL import Image
import librosa
import os
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# Set the project base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Comment out model loading for now (uncomment and replace paths when models are available)
# face_model = tf.keras.models.load_model('path_to_your_face_model.h5')  # Placeholder
# voice_model = RandomForestClassifier()  # Placeholder

# Load and train the product recommendation model from Task 4 notebook
product_model = RandomForestClassifier(
    class_weight='balanced',
    max_depth=10,
    min_samples_leaf=1,
    n_estimators=200,
    random_state=42
)
merged_df = pd.read_csv(BASE_DIR / 'data' / 'merged_customer_data.csv')
features = ['engagement_score', 'purchase_interest_score', 'review_sentiment', 'social_media_platform', 'engagement_level']
X = pd.get_dummies(merged_df[features], columns=['review_sentiment', 'social_media_platform', 'engagement_level'], drop_first=True)
le = LabelEncoder()
y = le.fit_transform(merged_df['product'])
product_model.fit(X, y)  # Train with the saved dataset

# Load image_features.csv for facial recognition
image_features_df = pd.read_csv(BASE_DIR / 'data' / 'image_features.csv')
known_embeddings = np.array([np.array(eval(emb)) for emb in image_features_df['embedding']])
known_members = image_features_df['member'].values

# Feature extraction functions (replace with your implementations when available)
def load_image_features(image_path):
    # Adapted from image_processing.py: Preprocess image for facial recognition
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Temporary normalization
    # Placeholder: Replace with actual embedding extraction using ResNet18
    return img_array.flatten()  # Will be replaced with model embedding

def load_audio_features(audio_path):
    # Placeholder: Replace with your audio preprocessing
    pass

# Prediction functions (replace with your model prediction logic when available)
def facial_recognition(image_path):
    # Placeholder: Simulate facial recognition using image_features.csv
    if not os.path.exists(image_path):
        return False
    
    # Extract features from the input image (placeholder)
    input_features = load_image_features(image_path)
    input_features = input_features.reshape(1, -1)
    
    # Compare with known embeddings using cosine similarity (placeholder logic)
    similarities = cosine_similarity(input_features, known_embeddings)
    best_match_idx = np.argmax(similarities)
    similarity_score = similarities[0, best_match_idx]
    
    # Threshold for recognition (adjust as needed, e.g., 0.9)
    if similarity_score > 0.9:
        return True, known_members[best_match_idx]  # Return True and matched member
    return False, None

def voice_verification(audio_path):
    # Placeholder: Simulate with file existence
    if os.path.exists(audio_path):
        return True
    return False

def product_recommendation(customer_id):
    # Extract features for the given customer_id
    customer_data = merged_df[merged_df['customer_id'] == customer_id]
    if customer_data.empty:
        return "No recommendation available"
    
    X_customer = pd.get_dummies(customer_data[features], columns=['review_sentiment', 'social_media_platform', 'engagement_level'], drop_first=True)
    X_customer = X_customer.reindex(columns=X.columns, fill_value=0)  # Align columns
    prediction = product_model.predict(X_customer)
    return le.inverse_transform(prediction)[0]

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
    
    voice_audio = input("Enter path to voice sample: ")
    if voice_verification(voice_audio):
        print("Voice verified. Proceeding to product recommendation...")
    else:
        print("Access denied: Unrecognized voice.")
        return
    
    customer_id = input("Enter customer ID (e.g., A178): ")
    recommendation = product_recommendation(customer_id)
    print(f"Product recommendation: {recommendation}")

# Unauthorized attempt simulation
def simulate_unauthorized():
    print("\n=== Unauthorized Attempt Simulation ===")
    face_image = input("Enter path to unauthorized face image: ")
    voice_audio = input("Enter path to unauthorized voice sample: ")
    
    recognition_result, matched_member = facial_recognition(face_image)
    if not recognition_result:
        print("Access denied: Unrecognized face.")
    else:
        print(f"Face recognized as {matched_member}. Proceeding to voice verification...")
        if not voice_verification(voice_audio):
            print("Access denied: Unrecognized voice.")
        else:
            print("Unexpected success in unauthorized case!")
            customer_id = input("Enter unauthorized customer ID (e.g., UNKNOWN): ")
            recommendation = product_recommendation(customer_id)
            print(f"Product recommendation (should not occur): {recommendation}")

if __name__ == "__main__":
    simulate_system()
    simulate_unauthorized()