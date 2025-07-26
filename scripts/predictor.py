import joblib
import torch
import numpy as np
import cv2
import pandas as pd
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from audio_processing import display_audio_samples, process_audio_augmentations

# Define base path to load models and data
BASE_DIR = Path(__file__).resolve().parent.parent

class ModelPredictor:
    def __init__(self):
        # Load face recognition model and components
        self.face_model = joblib.load(BASE_DIR / 'models/face_recognition_model.pkl')
        self.face_scaler = joblib.load(BASE_DIR / 'models/image_scaler.pkl')
        self.face_pca = joblib.load(BASE_DIR / 'models/image_pca.pkl')
        self.face_encoder = joblib.load(BASE_DIR / 'models/face_label_encoder.pkl')

        # Load voice verification model and components
        self.voice_model = joblib.load(BASE_DIR / 'models/voice_verification_model.pkl')
        self.voice_scaler = joblib.load(BASE_DIR / 'models/audio_scaler.pkl')
        self.voice_encoder = joblib.load(BASE_DIR / 'models/voice_label_encoder.pkl')

        # Load product recommendation model and encoder
        self.product_model = joblib.load(BASE_DIR / 'models/product_recommendation_model.pkl')
        self.product_encoder = joblib.load(BASE_DIR / 'models/product_label_encoder.pkl')

        # Load pre-trained ResNet for facial embeddings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base = models.resnet18(weights='DEFAULT')
        self.embedding_model = torch.nn.Sequential(*list(base.children())[:-1]).to(self.device).eval()

        # Define image preprocessing steps
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_from_image(self, image_path):
        """
        Predict the identity from a face image.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image!")

        # Compute color histogram
        hist = []
        for i in range(3):
            h = cv2.calcHist([img], [i], None, [256], [0, 256])
            hist.extend(h.flatten())
        hist = np.pad(hist, (0, 768 - len(hist)), constant_values=0)

        # Extract embedding using ResNet
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.embedding_model(input_tensor).cpu().numpy().flatten()

        # Combine and transform features
        combined = np.concatenate([emb, hist])
        scaled = self.face_scaler.transform([combined])
        reduced = self.face_pca.transform(scaled)

        # Predict and decode identity
        pred = self.face_model.predict(reduced)
        probabilities = self.face_model.predict_proba(reduced)[0]
        predicted_label = self.face_encoder.inverse_transform(pred)[0]
        return predicted_label, probabilities

    def predict_from_audio(self, audio_path):
        """
        Predict the identity from a voice sample.
        """
        y, sr = display_audio_samples(audio_path, f'None - none', show_plots=False)
        features = process_audio_augmentations(y, sr, "None", "none", verbose=False)[0]

        # Filter out non-numerical fields
        exclude = ['member', 'phrase', 'augmentation']
        filtered = [v for k, v in features.items() if k not in exclude]
        scaled = self.voice_scaler.transform([filtered])

        pred = self.voice_model.predict(scaled)
        return self.voice_encoder.inverse_transform(pred)[0]

    def predict_product(self, customer_id):
        """
        Predict a recommended product based on customer_id.
        """
        df = pd.read_csv(BASE_DIR / 'data' / 'merged_customer_data.csv')

        # Preprocess like training
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        df['purchase_month'] = df['purchase_date'].dt.month
        df.drop(columns=['purchase_date'], inplace=True)
        df = pd.get_dummies(df, columns=['social_media_platform', 'review_sentiment'], drop_first=True)

        # Validate customer exists
        if customer_id not in df['customer_id_new'].values:
            raise ValueError(f"Customer ID '{customer_id}' not found.")

        # Prepare input row
        customer_row = df[df['customer_id_new'] == customer_id]
        X = customer_row.drop(columns=['customer_id_new', 'product_category'])

        # Align features with model input
        model_cols = self.product_model.feature_names_in_
        for col in model_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[model_cols]

        # Predict and decode product label
        pred = self.product_model.predict(X)
        return self.product_encoder.inverse_transform(pred)[0]
