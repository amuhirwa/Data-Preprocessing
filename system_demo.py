import os
import argparse
import joblib
import numpy as np
import pandas as pd
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from audio_processing import display_audio_samples, process_audio_augmentations
from sklearn.metrics.pairwise import cosine_similarity

class SystemDemo:
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.image_scaler = None
        self.image_pca = None
        self.audio_scaler = None
        self.feature_columns = {} 
        self.load_models()

    def load_models(self):
        """Load pre-trained models and scalers."""
        try:
            self.models['face_recognition'] = joblib.load('models/face_recognition_model.pkl')
            self.models['voice_verification'] = joblib.load('models/voice_verification_model.pkl')
            self.models['product_recommendation'] = joblib.load('models/product_recommendation_model.pkl')
            self.label_encoders['face'] = joblib.load('models/face_label_encoder.pkl')
            self.label_encoders['voice'] = joblib.load('models/voice_label_encoder.pkl')
            self.label_encoders['product'] = joblib.load('models/product_label_encoder.pkl')
            self.image_scaler = joblib.load('models/image_scaler.pkl')
            self.image_pca = joblib.load('models/image_pca.pkl')
            self.audio_scaler = joblib.load('models/audio_scaler.pkl')
        except FileNotFoundError as e:
            print(f"❌ Error: {e}. Ensure all model files are in the 'models/' directory.")
            exit(1)

    def predict_from_image(self, image_path):
        """Predict user identity from image."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base_model = models.resnet18(weights='DEFAULT')
        embedding_model = torch.nn.Sequential(*list(base_model.children())[:-1])
        embedding_model.to(device)
        embedding_model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image!")

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        def extract_histogram(img):
            if len(img.shape) == 2:
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                return hist.flatten()
            else:
                hist = []
                for i in range(3):
                    h = cv2.calcHist([img], [i], None, [256], [0, 256])
                    hist.extend(h.flatten())
                return np.array(hist)

        hist = extract_histogram(img)
        hist = np.pad(hist, (0, max(0, 768 - len(hist))), constant_values=0)

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = embedding_model(input_tensor).cpu().numpy().flatten()

        combined = np.concatenate([emb, hist])
        X_scaled = self.image_scaler.transform([combined])
        X_reduced = self.image_pca.transform(X_scaled)
        y_pred = self.models['face_recognition'].predict(X_reduced)
        probabilities = self.models['face_recognition'].predict_proba(X_reduced)[0]
        predicted_label = self.label_encoders['face'].inverse_transform(y_pred)[0]
        return predicted_label, probabilities

    def predict_from_audio(self, audio_path):
        """Predict user identity from audio."""
        y, sr = display_audio_samples(audio_path, 'None - none', show_plots=False)
        phrase_features = process_audio_augmentations(y, sr, "michael", "approve")
        exclude = ['member', 'phrase', 'augmentation']
        filtered_features = {k: v for k, v in phrase_features[0].items() if k not in exclude}
        X = np.array(list(filtered_features.values())).reshape(1, -1)
        X_scaled = self.audio_scaler.transform(X)
        y_pred = self.models['voice_verification'].predict(X_scaled)
        predicted_label = self.label_encoders['voice'].inverse_transform(y_pred)[0]
        return predicted_label
    
    
    def predict_product(self, user_id):
       """Generate product recommendation."""
       try:
           trans_df = pd.read_csv('C:\\Users\\hp\\Documents\\Data-Preprocessing\\customer_transactions.csv')
           social_df = pd.read_csv('C:\\Users\\hp\\Documents\\Data-Preprocessing\\customer_social_profiles.csv')
           print(f"✅ Loaded transactions with {len(trans_df)} records")
           print(f"✅ Loaded social profiles with {len(social_df)} records")
       except FileNotFoundError as e:
           print(f"❌ Error: {e}. Using dummy data.")
           data = {
               'customer_id': ['A178'],
               'social_media_platform': ['LinkedIn'],
               'engagement_score': [74.0],
               'purchase_interest_score': [4.9],
               'review_sentiment': ['Positive'],
               'purchase_amount': [408.0],
               'purchase_month': [1],
               'customer_rating': [2.3]
           }
           merged_df = pd.DataFrame(data)
       else:
           # Map customer_id_legacy to customer_id in trans_df
           trans_df['customer_id'] = 'A' + trans_df['customer_id_legacy'].astype(str).str.zfill(3)
           # Rename customer_id_new to customer_id in social_df for consistency
           social_df = social_df.rename(columns={'customer_id_new': 'customer_id'})
           # Merge on customer_id
           merged_df = pd.merge(social_df, trans_df, on='customer_id', how='left')
           # Derive purchase_month from purchase_date
           merged_df['purchase_month'] = pd.to_datetime(merged_df['purchase_date']).dt.month.fillna(1)
           # Retain customer_rating and select relevant columns
           merged_df = merged_df[['customer_id', 'social_media_platform', 'engagement_score',
                              'purchase_interest_score', 'review_sentiment', 'purchase_amount',
                              'purchase_month', 'customer_rating']]

       # Filter data for the specific user index
       if user_id < len(merged_df):
           user_data = merged_df.iloc[user_id:user_id+1].copy()
           print(f"Using data for customer: {user_data['customer_id'].iloc[0]}")
       else:
           print(f"❌ Error: User index {user_id} out of range. Using first record.")
           user_data = merged_df.iloc[0:1].copy()

       # 🔥 CRITICAL FIX: Drop customer_id BEFORE one-hot encoding (same as training)
       user_data = user_data.drop(columns=['customer_id'])
       print("Columns before one-hot encoding:", user_data.columns.tolist())

       # One-hot encode categorical variables to match training
       user_data = pd.get_dummies(user_data, columns=['social_media_platform', 'review_sentiment'], drop_first=True)
       print("After one-hot encoding columns:", user_data.columns.tolist())

       # 🔥 USE STORED TRAINING FEATURES instead of hardcoded list
       if 'product' not in self.feature_columns:
           raise ValueError("Model not trained yet or feature columns not stored")
    
       expected_features = self.feature_columns['product']
       print(f"Expected features from training: {expected_features}")

       # Add missing columns with 0
       for feat in expected_features:
           if feat not in user_data.columns:
               user_data[feat] = 0
               print(f"Added missing feature: {feat}")

       # Remove extra columns that weren't in training
       extra_cols = [col for col in user_data.columns if col not in expected_features]
       if extra_cols:
           user_data = user_data.drop(columns=extra_cols)
           print(f"Removed extra columns: {extra_cols}")

       # Reindex to ensure order matches training
       X = user_data.reindex(columns=expected_features, fill_value=0)
       print("Final X columns:", X.columns.tolist())
       print("Final X shape:", X.shape)

       try:
               y_pred = self.models['product_recommendation'].predict(X)
               product = self.label_encoders['product'].inverse_transform(y_pred)[0]
               print(f"✅ Recommended product: {product}")
               return product
       except Exception as e:
            print(f"❌ Product recommendation error: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Simulate User Authentication and Product Recommendation")
    parser.add_argument('--image', type=str, required=True, help="Path to face image")
    parser.add_argument('--audio', type=str, required=True, help="Path to audio sample")
    args = parser.parse_args()

    demo = SystemDemo()

    # Step 1: Face Recognition
    print("Step 1: Performing Face Recognition...")
    predicted_user, probabilities = demo.predict_from_image(args.image)
    threshold = 0.9
    if max(probabilities) > threshold and predicted_user in demo.label_encoders['face'].classes_:
        print(f"Face recognized as {predicted_user}. Proceeding to voice verification.")
    else:
        print("Access Denied: Unauthorized face detected.")
        return

    # Step 2: Voice Verification
    print("Step 2: Performing Voice Verification...")
    voice_user = demo.predict_from_audio(args.audio)
    if voice_user == predicted_user:
        print(f"Voice verified as {voice_user}. Proceeding to product recommendation.")
    else:
        print("Access Denied: Unauthorized voice detected.")
        return

    # Step 3: Product Recommendation
    print("Step 3: Generating Product Recommendation...")
    user_id = list(demo.label_encoders['face'].classes_).index(predicted_user)
    print(f"Generating recommendation for user index: {user_id}")
    recommendation = demo.predict_product(user_id)
    print(f"Recommended product: {recommendation}")

if __name__ == "__main__":
    main()